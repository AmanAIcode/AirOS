import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

# Download hand landmarker model if not exists
model_path = "models/hand_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading hand tracking model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Model downloaded!")

# Hand connections for drawing skeleton
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

def draw_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape
    points = {}

    # Draw dots on each landmark
    for i, lm in enumerate(hand_landmarks):
        x = int(lm.x * w)
        y = int(lm.y * h)
        points[i] = (x, y)
        cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

    # Draw skeleton lines
    for connection in HAND_CONNECTIONS:
        start = points.get(connection[0])
        end = points.get(connection[1])
        if start and end:
            cv2.line(frame, start, end, (0, 255, 0), 2)

    return points

# Setup MediaPipe Hand Landmarker
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

# Open webcam
cap = cv2.VideoCapture(0)

print("Hand Tracking Started! Press Q to quit.")

while True:
    success, frame = cap.read()

    if not success:
        print("Cannot read webcam!")
        break

    # Flip frame horizontally (mirror)
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect hands
    results = detector.detect(mp_image)

    # Draw results
    hand_count = 0
    if results.hand_landmarks:
        hand_count = len(results.hand_landmarks)
        for hand_landmark in results.hand_landmarks:
            points = draw_landmarks(frame, hand_landmark)

            # Green circle on index fingertip (landmark 8)
            if 8 in points:
                cv2.circle(frame, points[8], 12, (0, 255, 0), -1)
                cv2.putText(frame, f'Index: {points[8]}',
                           (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.8, (0, 255, 0), 2)

    # Show hand count
    cv2.putText(frame, f'Hands: {hand_count}',
               (10, 80),
               cv2.FONT_HERSHEY_SIMPLEX,
               1, (255, 0, 0), 2)

    # Show title
    cv2.putText(frame, 'AirOS - Hand Tracking v1.0',
               (10, frame.shape[0] - 20),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.7, (255, 255, 255), 2)

    # Display frame
    cv2.imshow('AirOS Hand Tracking', frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Hand Tracking Stopped.")