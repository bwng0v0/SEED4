import os
import time
import logging
from datetime import datetime

import numpy as np
import scipy.io as sio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ====== labels from ReadMe (trial 1..24) ======
session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]

# label mapping: 0 neutral, 1 sad, 2 fear, 3 happy
LABELS_BY_SESSION = {1: session1_label, 2: session2_label, 3: session3_label}


def setup_logger(log_dir="logs", name_prefix="rf_loso"):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{name_prefix}_{ts}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers in notebooks/reruns

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    logging.info(f"Logging to: {log_path}")
    return log_path


def _cm_to_str(cm: np.ndarray) -> str:
    # pretty print confusion matrix without extra deps
    return "\n".join(["[" + " ".join([f"{x:6d}" for x in row]) + "]" for row in cm])


def load_subject_session(
    session: int,
    subject_file: str,
    eeg_root="eeg_feature_smooth",
    eye_root="eye_feature_smooth",
    eeg_key_prefix="de_LDS",
):
    """
    Load ONE subject's ONE session data.

    Expects trial keys:
      EEG: de_LDS1..de_LDS24 each (62, W, 5)
      EYE: eye_1..eye_24     each (31, W)

    Returns:
      X: (num_windows_total, 341)   [EEG(310) + EYE(31)]
      y: (num_windows_total,)
      groups: (num_windows_total,)  subject_id repeated
    """
    eeg_path = os.path.join(eeg_root, str(session), subject_file)
    eye_path = os.path.join(eye_root, str(session), subject_file)

    if not os.path.exists(eeg_path):
        raise FileNotFoundError(f"EEG file not found: {eeg_path}")
    if not os.path.exists(eye_path):
        raise FileNotFoundError(f"Eye file not found: {eye_path}")

    eeg = sio.loadmat(eeg_path)
    eye = sio.loadmat(eye_path)

    # subject id: "1_20160518.mat" -> 1
    try:
        subj_id = int(subject_file.split("_")[0])
    except Exception as e:
        raise ValueError(f"Cannot parse subject id from filename: {subject_file}") from e

    labels = LABELS_BY_SESSION[session]  # length 24

    X_list, y_list, g_list = [], [], []

    for i in range(1, 25):
        eeg_key = f"{eeg_key_prefix}{i}"  # de_LDS1...
        eye_key = f"eye_{i}"              # eye_1...

        if eeg_key not in eeg:
            raise KeyError(f"Missing key '{eeg_key}' in {eeg_path}")
        if eye_key not in eye:
            raise KeyError(f"Missing key '{eye_key}' in {eye_path}")

        Xe = eeg[eeg_key]  # (62, W, 5)
        Xo = eye[eye_key]  # (31, W)

        if Xe.ndim != 3:
            raise ValueError(f"{eeg_key} expected 3D (62,W,5) but got {Xe.shape}")
        if Xo.ndim != 2:
            raise ValueError(f"{eye_key} expected 2D (31,W) but got {Xo.shape}")

        # alignment check: same W
        if Xe.shape[1] != Xo.shape[1]:
            raise ValueError(f"Window mismatch trial {i}: EEG {Xe.shape} vs EYE {Xo.shape}")

        W = Xe.shape[1]

        # EEG (62,W,5) -> (W, 310)
        Xe_w = np.transpose(Xe, (1, 0, 2)).reshape(W, -1)

        # EYE (31,W) -> (W,31)
        Xo_w = Xo.T

        # concat -> (W,341)
        Xw = np.concatenate([Xe_w, Xo_w], axis=1)

        yi = labels[i - 1]                   # trial label
        yv = np.full(W, yi, dtype=np.int64)  # replicate to each window
        gv = np.full(W, subj_id, dtype=np.int64)

        X_list.append(Xw)
        y_list.append(yv)
        g_list.append(gv)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.concatenate(g_list, axis=0)
    return X, y, groups


def load_session_all_subjects(session: int, eeg_root="eeg_feature_smooth", eye_root="eye_feature_smooth"):
    """
    Load ALL subjects for a given session.

    Returns:
      X: (N,341), y: (N,), groups: (N,)
    """
    eeg_folder = os.path.join(eeg_root, str(session))
    eye_folder = os.path.join(eye_root, str(session))

    if not os.path.isdir(eeg_folder):
        raise FileNotFoundError(f"EEG session folder not found: {eeg_folder}")
    if not os.path.isdir(eye_folder):
        raise FileNotFoundError(f"Eye session folder not found: {eye_folder}")

    files = [f for f in os.listdir(eeg_folder) if f.endswith(".mat")]
    files.sort()
    if len(files) == 0:
        raise RuntimeError(f"No .mat files found in {eeg_folder}")

    Xs, ys, gs = [], [], []
    for f in files:
        eye_path = os.path.join(eye_folder, f)
        if not os.path.exists(eye_path):
            raise FileNotFoundError(f"Eye file missing for {f}: {eye_path}")

        X, y, g = load_subject_session(session, f, eeg_root=eeg_root, eye_root=eye_root)
        Xs.append(X); ys.append(y); gs.append(g)

    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    g_all = np.concatenate(gs, axis=0)
    return X_all, y_all, g_all


def load_all_sessions_all_subjects(sessions=(1, 2, 3), eeg_root="eeg_feature_smooth", eye_root="eye_feature_smooth"):
    """
    Load ALL subjects across multiple sessions and concatenate.

    Returns:
      X: (N,341), y: (N,), groups: (N,)  (groups = subject_id)
    """
    Xs, ys, gs = [], [], []
    for s in sessions:
        X, y, g = load_session_all_subjects(s, eeg_root=eeg_root, eye_root=eye_root)
        Xs.append(X); ys.append(y); gs.append(g)

    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    g_all = np.concatenate(gs, axis=0)
    return X_all, y_all, g_all


def loso_cross_subject_all_sessions(
    sessions=(1, 2, 3),
    n_estimators=700,
    max_features=0.3,
    random_state=4742,
    min_samples_leaf=10,
    eeg_root="eeg_feature_smooth",
    eye_root="eye_feature_smooth",
):
    """
    (B) sessions 1+2+3을 합쳐서 하나의 큰 데이터로 LOSO.
    Fold마다 RF 1개씩 학습됨(평가용).
    """
    t0 = time.time()
    logging.info("========== EXP START ==========")
    logging.info(f"Mode: LOSO cross-subject on merged sessions {sessions}")
    logging.info(f"RF params: n_estimators={n_estimators}, max_features={max_features}, "
                 f"min_samples_leaf={min_samples_leaf}, random_state={random_state}")
    logging.info(f"Data roots: eeg_root={eeg_root}, eye_root={eye_root}")

    X, y, groups = load_all_sessions_all_subjects(sessions=sessions, eeg_root=eeg_root, eye_root=eye_root)
    logging.info(f"Loaded data: X={X.shape}, y={y.shape}, groups={groups.shape}")
    logging.info(f"Class counts: {dict(zip(*np.unique(y, return_counts=True)))}")
    logging.info(f"Subjects: {len(np.unique(groups))} -> {list(map(int, np.unique(groups)))}")

    subject_ids = np.unique(groups)
    fold_accs = []

    for test_subj in subject_ids:
        fold_t0 = time.time()
        test_mask = (groups == test_subj)
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        logging.info(f"----- FOLD START | test_subject={int(test_subj)} | "
                     f"train_windows={X_train.shape[0]} | test_windows={X_test.shape[0]} -----")

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=random_state,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        rf.fit(X_train, y_train)

        pred = rf.predict(X_test)
        acc = accuracy_score(y_test, pred)
        fold_accs.append(acc)

        cm = confusion_matrix(y_test, pred)
        report = classification_report(y_test, pred, digits=4)

        logging.info(f"FOLD RESULT | test_subject={int(test_subj)} | window_acc={acc:.4f}")
        logging.info("Confusion matrix:\n" + _cm_to_str(cm))
        logging.info("Classification report:\n" + report)
        logging.info(f"----- FOLD END | test_subject={int(test_subj)} | "
                     f"elapsed={time.time()-fold_t0:.1f}s -----")

    mean_acc = float(np.mean(fold_accs))
    std_acc  = float(np.std(fold_accs))

    logging.info("========== EXP END ==========")
    logging.info(f"FINAL | sessions={sessions} | mean_window_acc={mean_acc:.4f} ± {std_acc:.4f}")
    logging.info(f"Total elapsed: {time.time()-t0:.1f}s")

    return mean_acc, std_acc


def train_final_single_model_all_sessions(
    sessions=(1, 2, 3),
    n_estimators=700,
    max_features=0.3,
    random_state=4742,
    min_samples_leaf=10,
    eeg_root="eeg_feature_smooth",
    eye_root="eye_feature_smooth",
):
    """
    LOSO는 평가용이라 fold별 모델이 생김.
    '단일 모델'이 필요하면 전체 데이터로 한 번 더 학습해서 이 모델을 쓰면 됨.
    """
    t0 = time.time()
    logging.info("========== TRAIN FINAL MODEL ==========")
    logging.info(f"Training on ALL data | sessions={sessions}")
    logging.info(f"RF params: n_estimators={n_estimators}, max_features={max_features}, "
                 f"min_samples_leaf={min_samples_leaf}, random_state={random_state}")

    X, y, _ = load_all_sessions_all_subjects(sessions=sessions, eeg_root=eeg_root, eye_root=eye_root)
    logging.info(f"Loaded ALL data: X={X.shape}, y={y.shape}")
    logging.info(f"Class counts: {dict(zip(*np.unique(y, return_counts=True)))}")

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        random_state=random_state,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rf.fit(X, y)

    logging.info(f"[OK] Final RF trained. elapsed={time.time()-t0:.1f}s")
    return rf


if __name__ == "__main__":
    setup_logger()

    # (B) Evaluate with LOSO on merged sessions
    loso_cross_subject_all_sessions(
        sessions=(1, 2, 3),
        n_estimators=700,
        max_features=0.3,
        min_samples_leaf=10,
        random_state=4742,
    )

    # Train ONE final model on ALL data (for deployment / single model usage)
    final_rf = train_final_single_model_all_sessions(
        sessions=(1, 2, 3),
        n_estimators=700,
        max_features=0.3,
        min_samples_leaf=10,
        random_state=4742,
    )
    logging.info("[OK] Trained final single RF model on ALL sessions+subjects.")
