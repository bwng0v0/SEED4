import os
import numpy as np
import scipy.io as sio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ====== labels from ReadMe (trial 1..24) ======
session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]

# label mapping (ReadMe): 0 neutral, 1 sad, 2 fear, 3 happy
LABELS_BY_SESSION = {1: session1_label, 2: session2_label, 3: session3_label}

def load_subject_session(session: int, subject_file: str,
                         eeg_root="eeg_feature_smooth",
                         eye_root="eye_feature_smooth",
                         eeg_key_prefix="de_LDS"):
    """
    Returns:
      X: (num_windows_total, 341)
      y: (num_windows_total,)
      groups: (num_windows_total,)  # subject id for grouping (optional)
    """
    eeg_path = os.path.join(eeg_root, str(session), subject_file)
    eye_path = os.path.join(eye_root, str(session), subject_file)

    eeg = sio.loadmat(eeg_path)
    eye = sio.loadmat(eye_path)

    # subject id from filename: "1_20160518.mat" -> 1
    subj_id = int(subject_file.split("_")[0])

    X_list, y_list, g_list = [], [], []
    labels = LABELS_BY_SESSION[session]  # length 24

    for i in range(1, 25):
        eeg_key = f"{eeg_key_prefix}{i}"   # e.g., de_LDS1
        eye_key = f"eye_{i}"              # eye_1

        Xe = eeg[eeg_key]   # (62, W, 5)
        Xo = eye[eye_key]   # (31, W)

        # alignment check
        assert Xe.shape[1] == Xo.shape[1], f"Window mismatch at trial {i}: {Xe.shape} vs {Xo.shape}"
        W = Xe.shape[1]

        # build per-window samples
        # EEG: (62,W,5) -> (W, 62*5)
        Xe_w = np.transpose(Xe, (1, 0, 2)).reshape(W, -1)  # (W, 310)
        # EYE: (31,W) -> (W,31)
        Xo_w = Xo.T  # (W,31)

        Xw = np.concatenate([Xe_w, Xo_w], axis=1)          # (W,341)

        yi = labels[i-1]
        yv = np.full(W, yi, dtype=np.int64)

        X_list.append(Xw)
        y_list.append(yv)
        g_list.append(np.full(W, subj_id, dtype=np.int64))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.concatenate(g_list, axis=0)
    return X, y, groups

def load_session_all_subjects(session: int):
    folder = os.path.join("eeg_feature_smooth", str(session))
    files = [f for f in os.listdir(folder) if f.endswith(".mat")]
    files.sort()
    Xs, ys, gs = [], [], []
    for f in files:
        X, y, g = load_subject_session(session, f)
        Xs.append(X); ys.append(y); gs.append(g)
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)

# ====== Example: Cross-session train(1) -> test(2) ======
X_train, y_train, _ = load_session_all_subjects(1)
X_test,  y_test,  _ = load_session_all_subjects(2)

rf = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample"
)
rf.fit(X_train, y_train)

pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print("\nConfusion matrix:\n", confusion_matrix(y_test, pred))
print("\nReport:\n", classification_report(y_test, pred, digits=4))
