"""
EMG hand-gesture classification pipeline — core logic (validated standalone).

Signal parameters are modeled on real public sEMG datasets:
- Ozdemir, M.A. et al. (2022). Dataset for multi-channel surface electromyography
  (sEMG) signals of hand gestures. Data in Brief. DOI: 10.1016/j.dib.2022.107921
  (4-ch sEMG, Butterworth bandpass 5-500 Hz, 50 Hz notch, multiple gestures)
- Ninapro database (Atzori et al.), the standard EMG gesture-recognition benchmark.

NOTE: the EMG data here is SIMULATED and clearly labeled. Surface EMG is well-modeled
as a zero-mean stochastic signal whose amplitude is modulated by muscle activation; we
generate it that way per-gesture, add power-line interference and baseline drift so the
preprocessing steps are meaningful, then run a standard feature-extraction + ML pipeline.
The pipeline is identical for real data (see swap instructions in the notebook).
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, welch

FS = 1000          # sampling rate (Hz)
N_CH = 8           # EMG channels (forearm electrode array)
GESTURES = ["rest", "fist", "open_hand", "wrist_flexion", "wrist_extension", "pinch"]
RNG = np.random.default_rng(7)

# Each gesture activates a characteristic pattern across the 8 channels (0..1 intensity).
ACTIVATION = {
    "rest":            np.array([0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05]),
    "fist":            np.array([0.9, 0.85,0.8, 0.7, 0.6, 0.65,0.75,0.8 ]),
    "open_hand":       np.array([0.3, 0.4, 0.85,0.9, 0.8, 0.3, 0.25,0.2 ]),
    "wrist_flexion":   np.array([0.85,0.9, 0.3, 0.2, 0.25,0.3, 0.4, 0.35]),
    "wrist_extension": np.array([0.25,0.2, 0.35,0.4, 0.8, 0.85,0.8, 0.45]),
    "pinch":           np.array([0.75,0.7, 0.7, 0.6, 0.5, 0.65,0.7, 0.7 ]),
}


def _emg_burst(n, intensity):
    """One channel's EMG over n samples: zero-mean noise scaled by activation intensity."""
    base = RNG.normal(0, 1, n)
    # shape the spectrum a little (EMG energy concentrated ~50-150 Hz) via simple smoothing
    k = np.ones(3)/3
    base = np.convolve(base, k, mode="same")
    return base * (0.02 + intensity)   # 0.02 = baseline muscle tone


def generate_emg_trial(gesture, duration_s=2.0):
    """Generate one multi-channel EMG trial (n_ch x n_samples) for a gesture."""
    n = int(duration_s * FS)
    t = np.arange(n) / FS
    act = ACTIVATION[gesture].copy()
    # subject/trial-level gain + per-channel electrode-shift variation (realistic overlap)
    trial_gain = RNG.uniform(0.6, 1.4)
    act = act * trial_gain * RNG.uniform(0.65, 1.35, size=N_CH)
    raw = np.zeros((N_CH, n))
    for ch in range(N_CH):
        raw[ch] = _emg_burst(n, max(act[ch], 0.02))
    # channel cross-talk: each channel picks up a fraction of its neighbours (realistic)
    crosstalk = np.zeros_like(raw)
    for ch in range(N_CH):
        for d in (-1, 1):
            nb = ch + d
            if 0 <= nb < N_CH:
                crosstalk[ch] += 0.15 * raw[nb]
    raw = raw + crosstalk
    sig = np.zeros((N_CH, n))
    for ch in range(N_CH):
        powerline = 0.15 * np.sin(2*np.pi*50*t + RNG.uniform(0, 2*np.pi))   # 50 Hz mains
        drift = 0.2 * np.sin(2*np.pi*0.5*t + RNG.uniform(0, 2*np.pi))       # motion artifact
        noise = RNG.normal(0, 0.08, n)                                      # sensor noise
        sig[ch] = raw[ch] + powerline + drift + noise
    return sig


def build_dataset(trials_per_gesture=60, duration_s=2.0):
    """Return list of (signal[n_ch,n], gesture_label)."""
    data = []
    for g in GESTURES:
        for _ in range(trials_per_gesture):
            data.append((generate_emg_trial(g, duration_s), g))
    RNG.shuffle(data)
    return data


# ---------- preprocessing ----------
def bandpass(sig, lo=20, hi=450, fs=FS, order=4):
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype="band")
    return filtfilt(b, a, sig, axis=-1)

def notch(sig, f0=50, fs=FS, Q=30):
    b, a = iirnotch(f0/(fs/2), Q)
    return filtfilt(b, a, sig, axis=-1)

def preprocess(sig):
    return notch(bandpass(sig))


# ---------- feature extraction (canonical EMG features) ----------
def _zero_crossings(x, thr=1e-3):
    s = np.sign(x)
    return int(np.sum((s[:-1]*s[1:] < 0) & (np.abs(x[:-1]-x[1:]) > thr)))

def _waveform_length(x):
    return float(np.sum(np.abs(np.diff(x))))

def _mean_freq(x, fs=FS):
    f, P = welch(x, fs=fs, nperseg=min(256, len(x)))
    return float(np.sum(f*P)/np.sum(P)) if np.sum(P) > 0 else 0.0

def extract_features(sig):
    """sig: n_ch x n. Returns a flat feature vector (per-channel time + freq features)."""
    feats = []
    for ch in sig:
        rms = float(np.sqrt(np.mean(ch**2)))
        mav = float(np.mean(np.abs(ch)))
        wl  = _waveform_length(ch)
        zc  = _zero_crossings(ch)
        mnf = _mean_freq(ch)
        feats += [rms, mav, wl, zc, mnf]
    return np.array(feats)

FEATURE_NAMES = []
for ch in range(N_CH):
    for f in ["RMS", "MAV", "WL", "ZC", "MNF"]:
        FEATURE_NAMES.append(f"ch{ch+1}_{f}")


if __name__ == "__main__":
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline

    print("Generating dataset...")
    data = build_dataset(trials_per_gesture=60)
    print(f"trials: {len(data)} ({len(GESTURES)} gestures x 60)")

    X = np.array([extract_features(preprocess(sig)) for sig, _ in data])
    y = np.array([g for _, g in data])
    print("feature matrix:", X.shape)

    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10))
    rf  = RandomForestClassifier(n_estimators=200, random_state=0)
    print("SVM 5-fold accuracy:", round(cross_val_score(svm, X, y, cv=5).mean(), 3))
    print("RF  5-fold accuracy:", round(cross_val_score(rf,  X, y, cv=5).mean(), 3))
