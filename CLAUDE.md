# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project for emotion recognition using the SEED-IV (SJTU Emotion EEG Dataset for Four Emotions) dataset. The project implements machine learning models to classify four emotion states: neutral (0), sad (1), fear (2), and happy (3) using multimodal EEG and eye-tracking features.

## Dataset Structure

### Data Organization
- **Sessions**: 3 sessions per subject, each with different emotion label sequences
- **Trials**: 24 trials per session
- **Features**:
  - EEG: 62 channels, 5 frequency bands (delta, theta, alpha, beta, gamma), extracted using differential entropy (DE) with LDS smoothing
  - Eye tracking: 31 features per window
- **Time windows**: 4-second sliding windows for feature extraction

### Directory Layout
```
├── eeg_feature_smooth/{1,2,3}/    # Preprocessed EEG features per session
├── eye_feature_smooth/{1,2,3}/    # Preprocessed eye features per session
├── eeg_raw_data/{1,2,3}/          # Raw EEG data per session
├── eye_raw_data/{1,2,3}/          # Raw eye data per session
├── seed_iv/                        # Contains similar structure (redundant copy)
└── Channel Order.xlsx              # EEG channel arrangement documentation
```

### Data File Format
Subject files are named: `{subject_id}_{date}.mat` (e.g., `1_20160518.mat`)

Within each .mat file:
- **EEG keys**: `de_LDS1` through `de_LDS24` (one per trial)
  - Shape: `(62 channels, W windows, 5 frequency bands)`
- **Eye keys**: `eye_1` through `eye_24` (one per trial)
  - Shape: `(31 features, W windows)`

### Session Labels (Fixed for all subjects)
```python
session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
```

## Core Architecture

### Feature Construction
The project combines EEG and eye features into a unified feature vector per time window:
- **EEG features**: 62 channels × 5 bands = 310 features
- **Eye features**: 31 features
- **Total**: 341-dimensional feature vector per window

Feature extraction process:
1. Load trial data from .mat files
2. Reshape EEG from `(62, W, 5)` → `(W, 310)` by flattening channels and bands
3. Transpose eye data from `(31, W)` → `(W, 31)`
4. Concatenate horizontally to create `(W, 341)` feature matrix
5. Each window inherits the trial's emotion label

### Training Scripts

**LOSO.py**: Leave-One-Subject-Out cross-validation within a single session
- Trains on N-1 subjects, tests on the held-out subject
- Iterates through all subjects as test set
- Reports per-fold and average accuracy

**session1train_2test.py**: Cross-session generalization experiment
- Trains on all subjects from session 1
- Tests on all subjects from session 2
- Evaluates model's ability to generalize across different sessions

### Data Loading Functions
Both scripts use shared loading utilities:
- `load_subject_session(session, subject_file)`: Loads one subject's data for one session
  - Returns: `X (N_windows, 341)`, `y (N_windows,)`, `groups (N_windows,)`
  - Groups array contains subject IDs for stratification
- `load_session_all_subjects(session)`: Aggregates all subjects in a session
  - Discovers .mat files from `eeg_feature_smooth/{session}/` directory
  - Returns concatenated data from all subjects

## Running the Code

### Execute LOSO Cross-Validation
```bash
python LOSO.py
```
Currently configured to run session 1 (see line 181). Modify the session parameter to test other sessions.

### Execute Cross-Session Experiment
```bash
python session1train_2test.py
```
Trains on session 1, tests on session 2. Modify lines 79-80 to change source/target sessions.

### Inspect Data Structure
```bash
python check_mat.py
```
Utility script to examine .mat file keys and shapes (currently hardcoded to check `eye_feature_smooth/1/1_20160518.mat`).

## Model Configuration

Default RandomForest parameters:
- `n_estimators=500`: Number of trees
- `random_state=42`: Reproducibility seed
- `n_jobs=-1`: Use all CPU cores
- `class_weight="balanced_subsample"`: Handle class imbalance per bootstrap sample

## Dependencies

Required Python packages:
- `numpy`: Array operations
- `scipy`: MATLAB file I/O (`scipy.io.loadmat`)
- `scikit-learn`: RandomForestClassifier and metrics

## Important Implementation Details

1. **Subject ID Parsing**: Extracted from filename prefix (e.g., `"1_20160518.mat"` → subject 1)
2. **Window Alignment**: Code validates that EEG and eye data have matching window counts per trial
3. **Label Indexing**: Trial labels are 0-indexed in arrays but trials are numbered 1-24 (subtract 1 when indexing)
4. **Feature Ordering**: Always EEG features first (0:310), then eye features (310:341)
