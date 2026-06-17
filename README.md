# EMG Hand-Gesture Classification — Signal Processing + Machine Learning

A full surface-EMG (sEMG) pipeline for **hand-gesture recognition** — the computational core of
**myoelectric prosthetic control** and **EMG-based neurorehabilitation**. Raw signal → filtering →
feature extraction → machine-learning classification, built in Python (SciPy + scikit-learn).

> **Author:** Isaac Glenu — Biomedical Engineering, University of Waterloo

---

## What it does
1. **Simulates** realistic 8-channel sEMG for 6 hand gestures (rest, fist, open hand, wrist flexion,
   wrist extension, pinch), with power-line interference, motion-artifact drift, channel cross-talk,
   and sensor noise.
2. **Preprocesses** — 20–450 Hz Butterworth bandpass + 50 Hz notch (the standard EMG filter chain),
   with a power-spectrum plot showing *why* those choices.
3. **Extracts features** — the canonical EMG set per channel: RMS, MAV, waveform length, zero
   crossings, mean frequency (8 channels × 5 = 40-D feature vector).
4. **Classifies** — RBF-kernel SVM and Random Forest, 5-fold cross-validated.
5. **Evaluates** — confusion matrix + feature importances.

## Key results
- ~85–90% cross-validated accuracy across 6 gestures (realistic for surface EMG).
- Filtering removes 50 Hz mains hum and low-frequency drift, isolating the muscle-activation band.
- Errors concentrate where they should — between **mechanically similar gestures (fist ↔ pinch)** —
  matching real myoelectric systems.

## ⚠️ Data provenance
The EMG signals are **simulated** and clearly labeled as such. Surface EMG is well-modeled as a
zero-mean stochastic signal amplitude-modulated by muscle activation, generated here per gesture with
realistic artifacts so the preprocessing is meaningful. Signal parameters (sampling rate, filter
bands, gesture set, channels) are modeled on **real public datasets**, so the pipeline runs on real
data with minimal changes:

- Ozdemir, M. A., Kisa, D. H., Guren, O., & Akan, A. (2022). *Dataset for multi-channel surface
  electromyography (sEMG) signals of hand gestures.* **Data in Brief, 41, 107921.**
  DOI: [10.1016/j.dib.2022.107921](https://doi.org/10.1016/j.dib.2022.107921) (CSV files; Butterworth
  bandpass 5–500 Hz + 50 Hz notch).
- Ninapro database (Atzori et al.) — the standard EMG gesture-recognition benchmark.

**To run on real data:** load each trial as an `n_channels × n_samples` array with a gesture label and
feed it into `preprocess()` → `extract_features()` exactly as in the notebook.

## Run it
```bash
pip install numpy scipy scikit-learn matplotlib jupyter
jupyter notebook emg_gesture_classification.ipynb
```

## Files
- `emg_gesture_classification.ipynb` — the full pipeline (runs top-to-bottom).
- `figures/` — raw-vs-filtered signal, power spectrum, feature heatmap, confusion matrix, feature
  importances.

## Possible next steps
- Swap in a real dataset (Ozdemir 2022 / Ninapro) — same `preprocess` / `extract_features`.
- Add sliding-window segmentation for real-time control + report per-window latency.
- Test cross-subject generalization (train on some subjects, test on unseen) — the hard,
  research-relevant version of the problem.

---
*Simulated data, clearly labeled; parameters modeled on Ozdemir et al. (2022) and the Ninapro benchmark.*
