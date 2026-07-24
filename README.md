# Ping-Pong Point-Outcome Classifier from Player Reaction

Deep-learning pipeline that decides whether a table-tennis player **won or lost** the last point by analyzing the **3-second emotional/body reaction** captured immediately after each rally ends. Instead of watching the ball or the score, the model watches the *player*.

Built on **MediaPipe Holistic** for whole-body keypoint extraction (pose + face + both hands, 454 features / frame) and a **Bi-directional LSTM** over 90-frame sequences (3 s at 30 fps) for binary classification: **Gewonnen (G) vs Verloren (V)** — German labels for "Won" vs "Lost".

The repository also ships a **model-explainability tool** that ranks the top 30 features driving each prediction and labels them semantically (which facial region, which hand joint, which pose landmark), so the model's decisions can be inspected — not just measured.

---

## Motivation

A table-tennis point ends in milliseconds, but the player's body tells you the outcome long before the scoreboard updates. Winners raise their arms, loosen their shoulders, and their facial micro-expressions are distinctly different from losers, who tend to slump, tighten their grip, or briefly look away. This project asks: *can a model learn that signal purely from a 3-second post-point clip, with no ball, no scoreboard, and no audio?*

The pipeline treats each rally-end as a short **spatio-temporal sequence of body language**, uses MediaPipe to strip everything except the person's skeleton and face mesh, and trains an LSTM to classify the outcome from that skeleton stream alone.

---

## Core Features

- **454-feature per-frame body representation** — MediaPipe Holistic gives us:
  - 7 upper-body pose landmarks × (x, y, z, visibility) = 28
  - 100 face landmarks × (x, y, z) = 300
  - 21 right-hand landmarks × (x, y, z) = 63
  - 21 left-hand landmarks × (x, y, z) = 63
  - **Total: 454 features / frame**, 90 frames / sample → tensor `(90, 454)` per rally.
- **On-the-fly data augmentation** during keypoint extraction: horizontal flip (50 %), HSV value jitter, small rotations (±15°), and Gaussian noise — dramatically increases effective dataset size and improves generalization to different lighting and player positions.
- **Bi-directional-ready LSTM classifier** (128 units → 64 dense → 2 softmax) with **Dropout (0.4)** on both hidden layers, `categorical_crossentropy`, and Adam.
- **Class-balanced training** via `sklearn.utils.class_weight` so the model is not biased when one class (won/lost) is over-represented in the dataset.
- **Reproducible splits & seeds** — deterministic `PYTHONHASHSEED = 42`, and matching seeds for `random`, `numpy`, and `tensorflow`, plus a fixed 60/20/20 train/val/test partition.
- **Full evaluation report** — accuracy, precision, recall, F1, AUC-ROC, and confusion matrix — automatically saved as a multi-page **PDF** plus a **PNG** confusion-matrix and a per-sample **CSV** with predicted probabilities for every test file.
- **`ModelCheckpoint`** to keep only the best-`val_loss` weights.
- **Perturbation-based feature-importance analysis** — for each of the 454 features, we add a small ε and measure `|Δ softmax|`. The top 30 features are then **semantically labelled** (e.g. `L_thumb_tip_z`, `Face_Brow_R_y`, `right_shoulder_v`) and rendered as a color-coded bar chart, so you can literally see whether the model is focusing on the face, the hands, or the shoulders.
- **Body-region color-coding** in the importance plot: face (red), right hand (green), left hand (orange), pose (blue) — makes it trivial to spot whether wins are being detected via smiles or via arm-raises.

---

## Pipeline

```
raw_videos/*.mp4        ─┐
                          │  (1) extract_keypoints.py
                          │      MediaPipe Holistic + augmentation
                          ▼
AUGMENTED_KEYPOINTS/*.npy  ← 90-frame × 454-feature sequences
                          │
                          │  (2) train_lstm.py
                          │      60/20/20 split, LSTM, class weights
                          ▼
lstm_best_model.h5         ← best-checkpoint weights
+ Desktop/KI/*.pdf, *.png, *.csv
                          │
                          │  (3) feature_importance.py
                          │      perturbation-based ranking + labels
                          ▼
important_features_custom.png
```

Naming convention: raw video files begin with **`G_`** for won points and **`V_`** for lost points (Gewonnen / Verloren). The label parser lives in `label_map = {'G': 0, 'V': 1}`.

---

## Repository Layout

```
pingpong-reaction-classifier/
├── src/
│   ├── extract_keypoints.py     # Stage 1 — MediaPipe Holistic + augmentation
│   ├── train_lstm.py            # Stage 2 — LSTM training + full eval report
│   └── feature_importance.py    # Stage 3 — perturbation-based explainability
├── examples/
│   └── mediapipe_hands_demo.py  # Standalone webcam hand-tracking demo (early prototype)
├── docs/
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## Data Layout

Expected input / output directories (created relative to wherever you run the scripts):

```
raw_videos/            # Input: <label>_<anything>.mp4  →  G_match17_pt3.mp4, V_match17_pt4.mp4, …
AUGMENTED_KEYPOINTS/   # Output of Stage 1 — 90×454 .npy sequences
features_data/         # Rename/symlink AUGMENTED_KEYPOINTS to this for Stage 2/3
lstm_best_model.h5     # Saved by Stage 2 checkpoint
~/Desktop/KI/          # Reports (PDF/PNG/CSV) — path is hard-coded, adjust if needed
```

> **Note:** the training and explainability scripts write reports into `~/Desktop/KI/`. Edit `desktop_path` / `output_path` inside the scripts if you want them elsewhere.

---

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/mahdi1993bayat-ui/pingpong-reaction-classifier.git
cd pingpong-reaction-classifier

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

`tensorflow`, `mediapipe`, and `opencv-python` are the heavy dependencies. On Apple Silicon use `tensorflow-macos` instead of `tensorflow`.

---

## Usage

**Stage 1 — Extract keypoints from raw videos**

Put your rally clips in `raw_videos/` with the `G_` / `V_` prefix convention, then:

```bash
python src/extract_keypoints.py
```

Output goes to `AUGMENTED_KEYPOINTS/*.npy` (one file per clip, shape `(90, 454)`).

**Stage 2 — Train the LSTM**

Rename or symlink `AUGMENTED_KEYPOINTS/` to `features_data/`, then:

```bash
python src/train_lstm.py
```

You'll get `lstm_best_model.h5` plus a full evaluation bundle in `~/Desktop/KI/`:
- `lstm_eval_report.pdf` — confusion matrix, accuracy & loss curves, metric summary
- `lstm_confusion_matrix.png` — high-res confusion matrix
- `lstm_prediction_results.csv` — per-sample true label, predicted label, class probabilities

**Stage 3 — See which features the model relies on**

```bash
python src/feature_importance.py
```

Produces `~/Desktop/KI/important_features_custom.png` — a color-coded bar chart of the top 30 features driving predictions, labelled by body region.

---

## Model & Training Details

| | |
|---|---|
| Architecture | `LSTM(128, return_sequences=False) → Dropout(0.4) → Dense(64, relu) → Dropout(0.4) → Dense(2, softmax)` |
| Input shape | `(90, 454)` — 90 frames × 454 features |
| Loss | `categorical_crossentropy` |
| Optimizer | Adam (default LR) |
| Epochs | 140 |
| Batch size | 32 |
| Class weights | `sklearn.utils.class_weight.compute_class_weight('balanced', …)` |
| Checkpoint | best `val_loss` → `lstm_best_model.h5` |
| Splits | 60 % train / 20 % val / 20 % test, shuffled once |
| Seeds | `PYTHONHASHSEED=42`, `random`, `numpy`, `tensorflow` all seeded 42 |

---

## Broader Project Context

This LSTM classifier is **one of several models** built for the same problem. The full research effort also included:
- **YOLO**-based person / paddle detection variants,
- **MSN**-style self-supervised representations,
- and other hybrid architectures.

Only the LSTM stage is published in this repository; the other stages are held separately.

---

## Contributors

This is a **joint personal project** developed together by two authors:

- **Mahdi Bayat** — [@mahdi1993bayat-ui](https://github.com/mahdi1993bayat-ui) — data collection, model training, evaluation pipeline
- **Fazelehsadat Shirazian** — [@fazelehshirazian-ship-it](https://github.com/fazelehshirazian-ship-it) — co-author, data collection and experimental design

Both authors contributed equally to the research design, data annotation, and iterative model refinement across the different architectures explored during the project.

A mirror of this repository is also maintained at [@fazelehshirazian-ship-it/pingpong-reaction-classifier](https://github.com/fazelehshirazian-ship-it/pingpong-reaction-classifier).

---

## License

Released under the MIT License. See `LICENSE`.
