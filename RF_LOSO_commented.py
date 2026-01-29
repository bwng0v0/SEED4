import os
import numpy as np
import scipy.io as sio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# [PART 0] 세션별 trial 라벨 정의 (ReadMe의 trial 1..24 라벨)
# - SEED-IV는 세션마다 24개 trial(영상)에 대한 감정 라벨이 고정되어 있음
# - 각 trial은 여러 개의 sliding window(W개)로 잘려 feature가 존재
# - 여기서 라벨은 "trial 단위" 라벨이며, 이후 window 단위로 복제해서 붙일 것임
# ============================================================

session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]

# label mapping: 0 neutral, 1 sad, 2 fear, 3 happy
LABELS_BY_SESSION = {1: session1_label, 2: session2_label, 3: session3_label}


# ============================================================
# [PART 1] 단일 subject + 단일 session 데이터를 로드하고 전처리/결합하는 함수
# - 입력: session 번호, subject 파일명(예: "1_20160518.mat")
# - EEG 파일: eeg_feature_smooth/{session}/{subject_file}
# - Eye 파일: eye_feature_smooth/{session}/{subject_file}
#
# - 각 trial마다:
#   EEG 키: de_LDS1..de_LDS24   shape (62, W, 5)
#   Eye 키: eye_1..eye_24       shape (31, W)
#
# - 목표: trial별 window를 모두 이어붙여서
#   X: (총 window 수, 341) = EEG 310 + Eye 31
#   y: (총 window 수,)     = trial 라벨을 window 수만큼 복제
#   groups: (총 window 수,) = subject id를 window 수만큼 복제 (LOSO용)
# ============================================================

def load_subject_session(
    session: int,
    subject_file: str,
    eeg_root="eeg_feature_smooth",
    eye_root="eye_feature_smooth",
    eeg_key_prefix="de_LDS",
):
    """
    Load ONE subject's ONE session data from:
      EEG: {eeg_root}/{session}/{subject_file}
      EYE: {eye_root}/{session}/{subject_file}

    Expects trial keys:
      EEG: de_LDS1..de_LDS24 each (62, W, 5)
      EYE: eye_1..eye_24     each (31, W)

    Returns:
      X: (num_windows_total, 341)   [EEG(310) + EYE(31)]
      y: (num_windows_total,)
      groups: (num_windows_total,)  subject_id repeated
    """

    # ------------------------------------------------------------
    # (1) 파일 경로 구성 및 존재 여부 확인
    # ------------------------------------------------------------
    eeg_path = os.path.join(eeg_root, str(session), subject_file)
    eye_path = os.path.join(eye_root, str(session), subject_file)

    if not os.path.exists(eeg_path):
        raise FileNotFoundError(f"EEG file not found: {eeg_path}")
    if not os.path.exists(eye_path):
        raise FileNotFoundError(f"Eye file not found: {eye_path}")

    # ------------------------------------------------------------
    # (2) .mat 파일 로드 (dict 형태로 key -> ndarray 접근)
    # ------------------------------------------------------------
    eeg = sio.loadmat(eeg_path)
    eye = sio.loadmat(eye_path)

    # ------------------------------------------------------------
    # (3) subject id 파싱 (파일명 "1_20160518.mat" -> 1)
    #     LOSO에서 그룹(피험자) 단위 분할을 위해 필요
    # ------------------------------------------------------------
    try:
        subj_id = int(subject_file.split("_")[0])
    except Exception as e:
        raise ValueError(f"Cannot parse subject id from filename: {subject_file}") from e

    # ------------------------------------------------------------
    # (4) 해당 session의 trial 라벨 배열 가져오기 (길이 24)
    #     trial i의 라벨 = labels[i-1]
    # ------------------------------------------------------------
    labels = LABELS_BY_SESSION[session]  # length 24

    X_list, y_list, g_list = [], [], []

    # ------------------------------------------------------------
    # (5) trial 1..24를 순회하면서 EEG/eye feature를 window 단위로 펼치고 결합
    # ------------------------------------------------------------
    for i in range(1, 25):
        eeg_key = f"{eeg_key_prefix}{i}"  # ex) de_LDS1
        eye_key = f"eye_{i}"              # ex) eye_1

        # 키 존재 체크 (데이터셋/파일 누락 조기 발견)
        if eeg_key not in eeg:
            raise KeyError(f"Missing key '{eeg_key}' in {eeg_path}")
        if eye_key not in eye:
            raise KeyError(f"Missing key '{eye_key}' in {eye_path}")

        # 원본 feature 로드
        Xe = eeg[eeg_key]  # expected (62, W, 5)
        Xo = eye[eye_key]  # expected (31, W)

        # shape sanity check
        if Xe.ndim != 3:
            raise ValueError(f"{eeg_key} expected 3D (62,W,5) but got {Xe.shape}")
        if Xo.ndim != 2:
            raise ValueError(f"{eye_key} expected 2D (31,W) but got {Xo.shape}")

        # --------------------------------------------------------
        # (6) EEG와 Eye의 window 개수 W가 같은지 확인 (alignment 검증)
        #     둘이 같은 시간창으로 잘려있다는 가정 확인
        # --------------------------------------------------------
        if Xe.shape[1] != Xo.shape[1]:
            raise ValueError(f"Window mismatch trial {i}: EEG {Xe.shape} vs EYE {Xo.shape}")

        W = Xe.shape[1]  # trial i의 window 개수

        # --------------------------------------------------------
        # (7) EEG reshape:
        #     (62, W, 5) -> transpose해서 window를 앞축으로 -> (W, 62, 5)
        #     이후 (W, 310)로 펼침 (62*5=310)
        # --------------------------------------------------------
        Xe_w = np.transpose(Xe, (1, 0, 2)).reshape(W, -1)

        # --------------------------------------------------------
        # (8) Eye reshape:
        #     (31, W) -> transpose -> (W, 31)
        # --------------------------------------------------------
        Xo_w = Xo.T

        # --------------------------------------------------------
        # (9) EEG + Eye 결합:
        #     (W, 310) + (W, 31) -> (W, 341)
        # --------------------------------------------------------
        Xw = np.concatenate([Xe_w, Xo_w], axis=1)

        # --------------------------------------------------------
        # (10) 라벨/그룹을 window 수만큼 복제
        #      trial 라벨을 그 trial의 모든 window에 동일하게 붙임
        # --------------------------------------------------------
        yi = labels[i - 1]  # trial i의 감정 라벨
        yv = np.full(W, yi, dtype=np.int64)
        gv = np.full(W, subj_id, dtype=np.int64)

        # trial 단위 결과를 리스트에 누적
        X_list.append(Xw)
        y_list.append(yv)
        g_list.append(gv)

    # ------------------------------------------------------------
    # (11) 24개 trial을 모두 이어붙여 subject-session 전체 데이터 생성
    # ------------------------------------------------------------
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.concatenate(g_list, axis=0)
    return X, y, groups


# ============================================================
# [PART 2] 단일 session의 "모든 subject"를 로드해서 하나의 큰 데이터셋으로 합치는 함수
# - session 폴더 안의 .mat 파일 목록을 읽어 모든 subject를 순회
# - 각 subject에 대해 PART 1(load_subject_session) 호출
# - 최종적으로 session 전체:
#   X: (전체 window 수, 341), y: (전체 window 수,), groups: (전체 window 수,)
# ============================================================

def load_session_all_subjects(
    session: int,
    eeg_root="eeg_feature_smooth",
    eye_root="eye_feature_smooth",
):
    """
    Load ALL subjects for a given session.
    Uses EEG folder listing as the canonical file list, and assumes
    the same filenames exist in the eye folder.

    Returns:
      X: (N,341), y: (N,), groups: (N,)
    """
    # ------------------------------------------------------------
    # (1) session 폴더 존재 확인
    # ------------------------------------------------------------
    eeg_folder = os.path.join(eeg_root, str(session))
    eye_folder = os.path.join(eye_root, str(session))

    if not os.path.isdir(eeg_folder):
        raise FileNotFoundError(f"EEG session folder not found: {eeg_folder}")
    if not os.path.isdir(eye_folder):
        raise FileNotFoundError(f"Eye session folder not found: {eye_folder}")

    # ------------------------------------------------------------
    # (2) EEG 폴더의 .mat 파일 리스트를 subject 리스트로 사용
    #     (eye 폴더도 동일 파일명이 있다고 가정)
    # ------------------------------------------------------------
    files = [f for f in os.listdir(eeg_folder) if f.endswith(".mat")]
    files.sort()

    if len(files) == 0:
        raise RuntimeError(f"No .mat files found in {eeg_folder}")

    # ------------------------------------------------------------
    # (3) 각 subject 파일을 로드해서 누적 결합
    # ------------------------------------------------------------
    Xs, ys, gs = [], [], []
    for f in files:
        # eye 파일 존재 확인 (짝이 안 맞으면 즉시 에러)
        eye_path = os.path.join(eye_folder, f)
        if not os.path.exists(eye_path):
            raise FileNotFoundError(f"Eye file missing for {f}: {eye_path}")

        # subject-session 로드 (PART 1)
        X, y, g = load_subject_session(session, f, eeg_root=eeg_root, eye_root=eye_root)
        Xs.append(X)
        ys.append(y)
        gs.append(g)

    # session 전체로 합치기
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0), np.concatenate(gs, axis=0)


# ============================================================
# [PART 3] LOSO(Leave-One-Subject-Out) cross-subject 평가 함수
# - 같은 session 안에서:
#   1) 전체 subject 데이터를 로드한 뒤
#   2) subject_id(groups) 기준으로
#      한 명을 test로, 나머지를 train으로 반복
# - 모델: RandomForestClassifier
# - 각 fold마다 accuracy / confusion matrix / classification report 출력
# - 마지막에 mean ± std 출력
# ============================================================

def loso_cross_subject(session: int, n_estimators=500, random_state=42):
    # ------------------------------------------------------------
    # (1) session 전체 데이터 로드 (모든 subject)
    # ------------------------------------------------------------
    X, y, groups = load_session_all_subjects(session)

    # ------------------------------------------------------------
    # (2) subject 목록을 얻고, subject별로 반복해서 LOSO 수행
    # ------------------------------------------------------------
    subject_ids = np.unique(groups)
    fold_accs = []

    for test_subj in subject_ids:
        # test subject와 train subject 분리
        test_mask = (groups == test_subj)
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        # --------------------------------------------------------
        # (3) 랜덤포레스트 학습
        # - n_estimators: 트리 개수
        # - class_weight="balanced_subsample": 각 트리 학습 시 클래스 불균형 보정
        # --------------------------------------------------------
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample"
        )
        rf.fit(X_train, y_train)

        # --------------------------------------------------------
        # (4) 예측 및 평가 지표 출력
        # --------------------------------------------------------
        pred = rf.predict(X_test)
        acc = accuracy_score(y_test, pred)
        fold_accs.append(acc)

        print(f"\n=== LOSO | session={session} | test_subject={test_subj} ===")
        print("Accuracy:", acc)
        print("Confusion matrix:\n", confusion_matrix(y_test, pred))
        print("Report:\n", classification_report(y_test, pred, digits=4))

    # ------------------------------------------------------------
    # (5) LOSO 전체 fold 평균/표준편차 출력
    # ------------------------------------------------------------
    print("\n========================")
    print(f"Session {session} LOSO mean acc: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
    print("========================")


# ============================================================
# [PART 4] 실행 엔트리 (실험 시작점)
# - session=1에서 LOSO cross-subject 실험 수행
# ============================================================

loso_cross_subject(session=1)
