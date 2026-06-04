import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32)
]

options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='pose_landmarker_lite.task'),
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    detection_result = detector.detect(mp_image)

    if detection_result.pose_landmarks:
        for landmarks in detection_result.pose_landmarks:
            h, w, _ = frame.shape
            pts = [(int(l.x * w), int(l.y * h)) for l in landmarks]

            for start_idx, end_idx in POSE_CONNECTIONS:
                if start_idx < len(pts) and end_idx < len(pts):
                    cv2.line(frame, pts[start_idx], pts[end_idx], (0, 255, 0), 2)

            for pt in pts:
                cv2.circle(frame, pt, 3, (0, 0, 255), -1)

    cv2.imshow("MediaPipe Pose", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
