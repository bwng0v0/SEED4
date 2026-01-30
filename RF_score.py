import os
import numpy as np
import scipy.io as sio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


# ====== labels from ReadMe (trial 1..24) ======
session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]

# label mapping: 0 neutral, 1 sad, 2 fear, 3 happy
LABELS_BY_SESSION = {1: session1_label, 2: session2_label, 3: session3_label}

# fixed class set for SEED-IV
CLASSES = np.array([0, 1, 2, 3], dtype=int)


def compute_multiclass_metrics(y_true, y_pred, y_proba, classes=CLASSES):
    """
    Returns:
      acc
      f1_macro, f1_weighted
      auc_ovr_macro, auc_ovr_weighted (or np.nan if cannot compute)
    """
    acc = accuracy_score(y_true, y_pred)

    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # ROC-AUC needs: y_true in one-vs-rest binarized form + per-class probabilities
    # Guard: if a fold is missing some classes in y_true, roc_auc_score can error.
    auc_macro = np.nan
    auc_weighted = np.nan
    try:
        y_true_bin = label_binarize(y_true, classes=classes)  # (n_samples, n_classes)
        # y_proba should be aligned to `classes` order (we'll enforce below)
        auc_macro = roc_auc_score(
            y_true_bin, y_proba, average="macro", multi_class="ovr"
        )
        auc_weighted = roc_auc_score(
            y_true_bin, y_proba, average="weighted", multi_class="ovr"
        )
    except Exception:
        # common when only 1 class appears in y_true (or other degenerate cases)
        pass

    return acc, f1_macro, f1_weighted, auc_macro, auc_weighted


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

    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0), np.concatenate(gs, axis=0)


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

    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0), np.concatenate(gs, axis=0)


def loso_cross_subject_all_sessions(
    sessions=(1, 2, 3),
    n_estimators=700,
    max_features=0.3,
    random_state=4742,
    min_samples_leaf=10,
):
    """
    (B) sessions 1+2+3을 합쳐서 하나의 큰 데이터로 LOSO.
    Fold마다 RF 1개씩 학습됨(평가용).
    """
    X, y, groups = load_all_sessions_all_subjects(sessions=sessions)

    subject_ids = np.unique(groups)

    # collect per-fold metrics
    accs, f1_macros, f1_weights, auc_macros, auc_weights = [], [], [], [], []

    for test_subj in subject_ids:
        test_mask = (groups == test_subj)
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

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

        # --- proba aligned to CLASSES order ---
        proba = rf.predict_proba(X_test)  # columns follow rf.classes_
        # align to fixed [0,1,2,3]
        aligned = np.zeros((X_test.shape[0], len(CLASSES)), dtype=float)
        for j, c in enumerate(rf.classes_):
            idx = int(np.where(CLASSES == c)[0][0])
            aligned[:, idx] = proba[:, j]

        acc, f1m, f1w, aucm, aucw = compute_multiclass_metrics(
            y_true=y_test, y_pred=pred, y_proba=aligned, classes=CLASSES
        )

        accs.append(acc); f1_macros.append(f1m); f1_weights.append(f1w)
        auc_macros.append(aucm); auc_weights.append(aucw)

        print(f"\n=== LOSO | sessions={sessions} | test_subject={test_subj} ===")
        print(f"Accuracy        : {acc:.4f}")
        print(f"F1 (macro)      : {f1m:.4f}")
        print(f"F1 (weighted)   : {f1w:.4f}")
        if np.isnan(aucm) or np.isnan(aucw):
            print("ROC-AUC (OvR)   : nan (fold에 특정 클래스가 없어서 계산 불가할 수 있음)")
        else:
            print(f"ROC-AUC OvR macro   : {aucm:.4f}")
            print(f"ROC-AUC OvR weighted: {aucw:.4f}")

        print("Confusion matrix:\n", confusion_matrix(y_test, pred))
        print("Report:\n", classification_report(y_test, pred, digits=4))

    def mean_std(x):
        x = np.array(x, dtype=float)
        return float(np.nanmean(x)), float(np.nanstd(x))

    mean_acc, std_acc = mean_std(accs)
    mean_f1m, std_f1m = mean_std(f1_macros)
    mean_f1w, std_f1w = mean_std(f1_weights)
    mean_aucm, std_aucm = mean_std(auc_macros)
    mean_aucw, std_aucw = mean_std(auc_weights)

    print("\n========================")
    print(f"ALL sessions {sessions} LOSO")
    print(f"Accuracy      : {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"F1 (macro)    : {mean_f1m:.4f} ± {std_f1m:.4f}")
    print(f"F1 (weighted) : {mean_f1w:.4f} ± {std_f1w:.4f}")
    print(f"ROC-AUC OvR macro   : {mean_aucm:.4f} ± {std_aucm:.4f} (nan 제외 평균)")
    print(f"ROC-AUC OvR weighted: {mean_aucw:.4f} ± {std_aucw:.4f} (nan 제외 평균)")
    print("========================")
    return (mean_acc, std_acc), (mean_f1m, std_f1m), (mean_f1w, std_f1w), (mean_aucm, std_aucm), (mean_aucw, std_aucw)


def train_final_single_model_all_sessions(
    sessions=(1, 2, 3),
    n_estimators=700,
    max_features=0.3,
    random_state=4742,
    min_samples_leaf=10,
):
    """
    LOSO는 평가용이라 fold별 모델이 생김.
    '단일 모델'이 필요하면 전체 데이터로 한 번 더 학습해서 이 모델을 쓰면 됨.
    """
    X, y, _ = load_all_sessions_all_subjects(sessions=sessions)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        random_state=random_state,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rf.fit(X, y)
    return rf


if __name__ == "__main__":
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
    print("\n[OK] Trained final single RF model on ALL sessions+subjects.")
