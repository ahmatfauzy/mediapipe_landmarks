import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Buat HandLandmarker
options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# Buka kamera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Konversi ke RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Deteksi tangan
    detection_result = detector.detect(mp_image)

    # Gambar landmark jika terdeteksi
    if detection_result.hand_landmarks:
        for landmarks in detection_result.hand_landmarks:
            for landmark in landmarks:
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    # Tampilkan hasil
    cv2.imshow("MediaPipe Hands", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp

# Inisialisasi Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)  # refine_landmarks=True untuk lebih presisi

# Buka kamera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Konversi ke RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Proses dengan Face Mesh
    results = face_mesh.process(frame_rgb)

    # Jika ada wajah terdeteksi
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Gambar hanya kontur wajah (bukan semua 468 titik)
            mp_draw.draw_landmarks(
                frame, 
                face_landmarks, 
                mp_face_mesh.FACEMESH_CONTOURS,  # Hanya gambar kontur
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
            )

    # Tampilkan hasil
    cv2.imshow("Face Mesh - Hanya Kontur Wajah", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp

# Inisialisasi Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Buka kamera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Konversi ke RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Proses dengan Face Mesh
    results = face_mesh.process(frame_rgb)

    # Jika ada wajah terdeteksi
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Gambar hanya mata dan bibir
            mp_draw.draw_landmarks(
                frame, 
                face_landmarks, 
                mp_face_mesh.FACEMESH_LIPS,  # Hanya bibir
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
            )
            mp_draw.draw_landmarks(
                frame, 
                face_landmarks, 
                mp_face_mesh.FACEMESH_LEFT_EYE,  # Mata kiri
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
            )
            mp_draw.draw_landmarks(
                frame, 
                face_landmarks, 
                mp_face_mesh.FACEMESH_RIGHT_EYE,  # Mata kanan
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
            )

    # Tampilkan hasil
    cv2.imshow("Face Mesh - Hanya Mata & Bibir", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
