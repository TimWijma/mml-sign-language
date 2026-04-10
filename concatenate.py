import sys
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

CROP_SIZE   = 112     # each crop is CROP_SIZE x CROP_SIZE px
TARGET_FPS  = 12.0    # output frame rate
MAX_FRAMES  = 512     # hard cap
FACE_MARGIN = 0.35    # expansion around detected face bbox
HAND_MARGIN = 0.45    # expansion around detected hand bbox
PAD_COLOR   = (20, 20, 20)

def safe_crop_bbox(frame, xmin, xmax, ymin, ymax, margin, size):
    fh, fw = frame.shape[:2]
    xmin = max(0.0, min(1.0, xmin))
    xmax = max(0.0, min(1.0, xmax))
    ymin = max(0.0, min(1.0, ymin))
    ymax = max(0.0, min(1.0, ymax))
    if xmax <= xmin or ymax <= ymin:
        return np.full((size, size, 3), PAD_COLOR, dtype=np.uint8)
    bw = (xmax - xmin) * fw
    bh = (ymax - ymin) * fh
    cx = (xmin + xmax) * 0.5 * fw
    cy = (ymin + ymax) * 0.5 * fh
    half = max(bw, bh) * (0.5 + margin)
    x1, x2 = int(cx - half), int(cx + half)
    y1, y2 = int(cy - half), int(cy + half)
    pl = max(0, -x1);  x1 = max(0, x1)
    pt = max(0, -y1);  y1 = max(0, y1)
    pr = max(0, x2 - fw); x2 = min(fw, x2)
    pb = max(0, y2 - fh); y2 = min(fh, y2)
    if x2 <= x1 or y2 <= y1:
        return np.full((size, size, 3), PAD_COLOR, dtype=np.uint8)
    patch = frame[y1:y2, x1:x2]
    if any([pl, pt, pr, pb]):
        patch = cv2.copyMakeBorder(patch, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=PAD_COLOR)
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)


def safe_crop_center(frame, cx, cy, half_norm, size):
    fh, fw = frame.shape[:2]
    half = max(4.0, half_norm * max(fw, fh))
    x1, x2 = int(cx * fw - half), int(cx * fw + half)
    y1, y2 = int(cy * fh - half), int(cy * fh + half)
    pl = max(0, -x1);  x1 = max(0, x1)
    pt = max(0, -y1);  y1 = max(0, y1)
    pr = max(0, x2 - fw); x2 = min(fw, x2)
    pb = max(0, y2 - fh); y2 = min(fh, y2)
    if x2 <= x1 or y2 <= y1:
        return np.full((size, size, 3), PAD_COLOR, dtype=np.uint8)
    patch = frame[y1:y2, x1:x2]
    if any([pl, pt, pr, pb]):
        patch = cv2.copyMakeBorder(patch, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=PAD_COLOR)
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)


def blank(size):
    return np.full((size, size, 3), PAD_COLOR, dtype=np.uint8)


def process_frame(frame, results):
    lm = results
    mp_pose = mp.solutions.pose.PoseLandmark

    nose_x = nose_y = shoulder_y = shoulder_w = None
    if lm.pose_landmarks:
        pl = lm.pose_landmarks.landmark
        nose_x, nose_y = pl[mp_pose.NOSE].x, pl[mp_pose.NOSE].y
        ls, rs = pl[mp_pose.LEFT_SHOULDER], pl[mp_pose.RIGHT_SHOULDER]
        shoulder_y = (ls.y + rs.y) / 2.0
        shoulder_w = abs(ls.x - rs.x) + 1e-6
    if shoulder_w is None:
        shoulder_w = 0.3

    # Face crop
    if lm.face_landmarks:
        xs = [p.x for p in lm.face_landmarks.landmark]
        ys = [p.y for p in lm.face_landmarks.landmark]
        face_crop = safe_crop_bbox(frame, min(xs), max(xs), min(ys), max(ys), FACE_MARGIN, CROP_SIZE)
    elif nose_y is not None:
        half_norm = max(0.10, shoulder_w * 0.55)
        face_crop = safe_crop_center(frame, nose_x, nose_y, half_norm, CROP_SIZE)
    else:
        face_crop = blank(CROP_SIZE)

    # Hand crops
    def hand_data(hand_lm):
        if hand_lm is None:
            return blank(CROP_SIZE), 0.5, 0.5
        xs = [p.x for p in hand_lm.landmark]
        ys = [p.y for p in hand_lm.landmark]
        crop = safe_crop_bbox(frame, min(xs), max(xs), min(ys), max(ys), HAND_MARGIN, CROP_SIZE)
        return crop, hand_lm.landmark[0].x, hand_lm.landmark[0].y

    left_crop, _, _ = hand_data(lm.left_hand_landmarks)
    right_crop, _, _ = hand_data(lm.right_hand_landmarks)

    mosaic = np.concatenate([left_crop, face_crop, right_crop], axis=1)

    return mosaic

def resample(video_path: str):
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        sys.exit(f"File not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"Cannot open: {video_path}")

    src_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride    = max(1, round(src_fps / TARGET_FPS))
    out_w     = CROP_SIZE * 3
    out_h     = CROP_SIZE

    video_out = video_path.parent / f"{video_path.stem}_resampled.mp4"

    print(f"Input  : {video_path.name}  ({src_w}x{src_h} @ {src_fps:.1f}fps, {src_total} frames)")
    print(f"Output : {video_out.name}  ({out_w}x{out_h} @ {TARGET_FPS}fps, stride={stride})")

    writer = cv2.VideoWriter(
        str(video_out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        TARGET_FPS,
        (out_w, out_h),
    )

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_idx = kept = 0

    while kept < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            mosaic = process_frame(frame, results)
            writer.write(mosaic)
            kept += 1
        frame_idx += 1

    cap.release()
    writer.release()
    holistic.close()

    print(f"Done   : {kept} frames")
    print(f"  -> {video_out}")


def find_video(script_dir: Path) -> str:
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"):
        matches = [m for m in sorted(script_dir.glob(ext)) if "_resampled" not in m.stem]
        if matches:
            return str(matches[0])
    sys.exit(f"No video file found in {script_dir}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        resample(sys.argv[1])
    else:
        resample(find_video(Path(__file__).parent))