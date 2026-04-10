import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

video_path = "testing_video2.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video file: {video_path}")

# change these to adjust the output crop size
OUTPUT_WIDTH = 500
OUTPUT_HEIGHT = 600


def _landmark_xy(landmarks, idx):
    if landmarks is None:
        return None
    lm = landmarks.landmark[idx]
    return lm.x, lm.y


def _compute_crop_center(results):
    # Use head (nose) and elbows to center the crop.
    nose = _landmark_xy(results.pose_landmarks, mp_holistic.PoseLandmark.NOSE)
    left_elbow = _landmark_xy(results.pose_landmarks, mp_holistic.PoseLandmark.LEFT_ELBOW)
    right_elbow = _landmark_xy(results.pose_landmarks, mp_holistic.PoseLandmark.RIGHT_ELBOW)

    points = [p for p in [nose, left_elbow, right_elbow] if p is not None]
    if not points:
        return None

    avg_x = sum(p[0] for p in points) / len(points)
    avg_y = sum(p[1] for p in points) / len(points)
    return avg_x, avg_y


def _crop_to_fixed_size(image, center_xy):
    h, w = image.shape[:2]
    if center_xy is None:
        cx, cy = w // 2, h // 2
    else:
        cx = int(center_xy[0] * w)
        cy = int(center_xy[1] * h)

    half_w = OUTPUT_WIDTH // 2
    half_h = OUTPUT_HEIGHT // 2

    left = max(0, cx - half_w)
    right = min(w, cx + half_w)
    top = max(0, cy - half_h)
    bottom = min(h, cy + half_h)

    crop = image[top:bottom, left:right]
    if crop.shape[0] == OUTPUT_HEIGHT and crop.shape[1] == OUTPUT_WIDTH:
        return crop

    # Pad with black to keep fixed output size when near borders.
    pad_top = max(0, half_h - cy)
    pad_left = max(0, half_w - cx)
    pad_bottom = OUTPUT_HEIGHT - (pad_top + crop.shape[0])
    pad_right = OUTPUT_WIDTH - (pad_left + crop.shape[1])
    return cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))



with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))

        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        center_xy = _compute_crop_center(results)
        cropped = _crop_to_fixed_size(image, center_xy)

        cv2.imshow('Full Body Detection', image)
        cv2.imshow('Cropped (TikTok)', cropped)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()