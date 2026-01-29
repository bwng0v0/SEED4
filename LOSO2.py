import os
import numpy as np
import scipy.io as sio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ====== labels from ReadMe (trial 1..24) ======
session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]

# label mapping: 0 neutral, 1 sad, 2 fear, 3 happy
LABELS_BY_SESSION = {1: session1_label, 2: session2_label, 3: session3_label}

def load_subject_session(
    session: int,
    subject_file: str,
    eeg_root="eeg_feature_smooth",
    eye_root="eye_feature_smooth",
    eeg_key_prefix="de_LDS",
):
    """
    Returns:
      X:        (num_windows_total, 341)
      y:        (num_windows_total,)
      groups:   (num_windows_total,)  # subject id
      trial_id: (num_windows_total,)  # trial index 1..24 (for trial-level voting)
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

    X_list, y_list, g_list, t_list = [], [], [], []

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

        yi = labels[i - 1]
        yv = np.full(W, yi, dtype=np.int64)
        gv = np.full(W, subj_id, dtype=np.int64)
        tv = np.full(W, i, dtype=np.int64)  # trial id 1..24

        X_list.append(Xw)
        y_list.append(yv)
        g_list.append(gv)
        t_list.append(tv)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.concatenate(g_list, axis=0)
    trial_id = np.concatenate(t_list, axis=0)
    return X, y, groups, trial_id


def load_session_all_subjects(
    session: int,
    eeg_root="eeg_feature_smooth",
    eye_root="eye_feature_smooth",
):
    """
    Returns:
      X: (N,341), y: (N,), groups: (N,), trial_id: (N,)
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

    Xs, ys, gs, ts = [], [], [], []
    for f in files:
        eye_path = os.path.join(eye_folder, f)
        if not os.path.exists(eye_path):
            raise FileNotFoundError(f"Eye file missing for {f}: {eye_path}")

        X, y, g, t = load_subject_session(session, f, eeg_root=eeg_root, eye_root=eye_root)
        Xs.append(X); ys.append(y); gs.append(g); ts.append(t)

    return (
        np.concatenate(Xs, axis=0),
        np.concatenate(ys, axis=0),
        np.concatenate(gs, axis=0),
        np.concatenate(ts, axis=0),
    )


def majority_vote_trial(pred, y_true, subj_ids, trial_ids):
    """
    trial 단위 majority voting:
      - 같은 (subject, trial) 묶음에서 pred를 다수결
      - y_true는 해당 묶음 내 모두 동일하다고 가정 (SEED-IV 라벨 방식)
    Returns:
      trial_acc, y_trial_true, y_trial_pred
    """
    keys = np.stack([subj_ids, trial_ids], axis=1)  # (N,2)

    # unique group index
    uniq_keys, inv = np.unique(keys, axis=0, return_inverse=True)

    y_trial_true = np.zeros(len(uniq_keys), dtype=np.int64)
    y_trial_pred = np.zeros(len(uniq_keys), dtype=np.int64)

    for k in range(len(uniq_keys)):
        idx = np.where(inv == k)[0]
        # true label: (should be constant)
        y_trial_true[k] = y_true[idx[0]]
        # pred majority
        vals, cnts = np.unique(pred[idx], return_counts=True)
        y_trial_pred[k] = vals[np.argmax(cnts)]

    trial_acc = accuracy_score(y_trial_true, y_trial_pred)
    return trial_acc, y_trial_true, y_trial_pred


def loso_cross_subject(
    session: int,
    n_estimators=500,
    random_state=42,
    use_scaler=True,
    print_detail=False,
):
    X, y, groups, trial_id = load_session_all_subjects(session)

    subject_ids = np.unique(groups)
    fold_window_accs = []
    fold_trial_accs = []

    for test_subj in subject_ids:
        test_mask = (groups == test_subj)
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        # ✅ Train-only standardization (no leakage)
        if use_scaler:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample"
        )
        rf.fit(X_train, y_train)

        pred = rf.predict(X_test)

        # window-level acc
        win_acc = accuracy_score(y_test, pred)
        fold_window_accs.append(win_acc)

        # trial-level acc (majority voting within this test subject)
        test_trial_id = trial_id[test_mask]
        test_groups   = groups[test_mask]
        trial_acc, y_tr, p_tr = majority_vote_trial(pred, y_test, test_groups, test_trial_id)
        fold_trial_accs.append(trial_acc)

        print(f"\n=== LOSO | session={session} | test_subject={test_subj} ===")
        print(f"Window-Acc: {win_acc:.4f} | Trial-Acc(vote): {trial_acc:.4f}")

        if print_detail:
            print("Confusion matrix (window):\n", confusion_matrix(y_test, pred))
            print("Report (window):\n", classification_report(y_test, pred, digits=4, zero_division=0))
            print("Confusion matrix (trial):\n", confusion_matrix(y_tr, p_tr))
            print("Report (trial):\n", classification_report(y_tr, p_tr, digits=4, zero_division=0))

    print("\n========================")
    print(f"Session {session} LOSO Window mean acc: {np.mean(fold_window_accs):.4f} ± {np.std(fold_window_accs):.4f}")
    print(f"Session {session} LOSO Trial  mean acc: {np.mean(fold_trial_accs):.4f} ± {np.std(fold_trial_accs):.4f}")
    print("========================")


# 실행
loso_cross_subject(session=1, use_scaler=True, print_detail=False)
