# FlightCue-ML

Machine learning pipeline for [FlightCue](https://github.com/samahm02/FlightCue),
an Android app that detects aircraft takeoff and landing using only the phone's
built-in accelerometer and barometer.

## What it does

The pipeline takes raw flight recordings, extracts features using a multi-grid
windowing strategy, and trains two GRU models (one for takeoff and one for
landing). The final models are exported to ONNX with z-score standardisation baked into the graph,
so the Android app can pass raw feature vectors directly with no extra
preprocessing.

## How it's built

The feature extraction code here is the reference implementation. The Android
Kotlin pipeline is verified against it using an end-to-end parity test that
checks all 154 features match within 1e-5 tolerance across 50 windows. Any
changes to feature computation need to be reflected in the app and re-verified.

Training accounts for the extreme class imbalance in flight event detection.
Sequence models use a two-phase negative sampling strategy where the model
first learns from a limited negative set before seeing the full imbalance.
Evaluation uses PR-AUC as the primary metric alongside hit rate and detection
delay, measured under the same trigger logic used in the deployed app.

## Thesis

Developed as part of a master's thesis at the University of Oslo, covering
dataset collection, model training, and evaluation against the prior
rule-based flyDetect system.
