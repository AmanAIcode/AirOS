import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request

# Download model if not exists
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

def get_finger_states(hand_landmarks):
    """Check which fingers are up or down"""
    fingers = []
    tips = [4, 8, 12, 16, 20]      # Fingertip landmarks
    mids = [3, 6, 10, 14, 18]      # Middle joint landmarks

    # Thumb (special case — checks left/right)
    if hand_landmarks[4].x < hand_landmarks[3].x:
        fingers.append(1)  # Thumb up
    else:
        fingers.append(0)  # Thumb down

    # Other 4 fingers (checks if tip is above middle joint)
    for tip, mid in zip(tips[1:], mids[1:]):
        if hand_landmarks[tip].y < hand_landmarks[mid].y:
            fingers.append(1)  # Finger up
        else:
            fingers.append(0)  # Finger down

    return fingers  # [thumb, index, middle, ring, pinky]


def detect_gesture(hand_landmarks):
    """Detect gesture from finger states"""
    fingers = get_finger_states(hand_landmarks)
    thumb, index, middle, ring, pinky = fingers

    # Calculate pinch distance (thumb tip to index tip)
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    pinch_dist = ((thumb_tip.x - index_tip.x)**2 +
                  (thumb_tip.y - index_tip.y)**2) ** 0.5

    # --- Gesture Rules ---

    # Fist — all fingers down
    if fingers == [0, 0, 0, 0, 0]:
        return "FIST ✊"

    # Open Palm — all fingers up
    if fingers == [1, 1, 1, 1, 1]:
        return "OPEN PALM 🖐️"

    # Index finger up only
    if fingers == [0, 1, 0, 0, 0]:
        return "POINTING ☝️"

    # Two fingers up (peace)
    if fingers == [0, 1, 1, 0, 0]:
        return "PEACE ✌️"

    # Three fingers up
    if fingers == [0, 1, 1, 1, 0]:
        return "THREE FINGERS 🤟"

    # Thumbs up
    if fingers == [1, 0, 0, 0, 0]:
        return "THUMBS UP 👍"

    # Thumbs down
    if thumb == 0 and index == 0 and middle == 0 and ring == 0 and pinky == 0:
        return "THUMBS DOWN 👎"

    # Rock sign
    if fingers == [0, 1, 0, 0, 1]:
        return "ROCK 🤘"

    # OK sign — pinch detected
    if pinch_dist < 0.05:
        return "PINCH 🤏"

    # Pinky up only
    if fingers == [0, 0, 0, 0, 1]:
        return "PINKY 🤙"

    # All fingers down except pinky and index
    if fingers == [0, 1, 0, 0, 1]:
        return "ROCK 🤘"

    return "UNKNOWN"


def draw_landmarks(frame, hand_landmarks):
    """Draw hand skeleton on frame"""
    h, w, _ = frame.shape
    points = {}

    for i, lm in enumerate(hand_landmarks):
        x = int(lm.x * w)
        y = int(lm.y * h)
        points[i] = (x, y)
        cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

    for connection in HAND_CONNECTIONS:
        start = points.get(connection[0])
        end = points.get(connection[1])
        if start and end:
            cv2.line(frame, start, end, (0, 255, 0), 2)

    return points


# Setup MediaPipe
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

print("Gesture Detection Started! Press Q to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip and convert
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect hands
    results = detector.detect(mp_image)

    if results.hand_landmarks:
        for i, hand_landmark in enumerate(results.hand_landmarks):

            # Draw skeleton
            points = draw_landmarks(frame, hand_landmark)

            # Detect gesture
            gesture = detect_gesture(hand_landmark)

            # Show gesture name on screen
            y_pos = 50 + (i * 60)
            cv2.putText(frame, f'Hand {i+1}: {gesture}',
                       (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 255, 255), 2)

            # Highlight index fingertip
            if 8 in points:
                cv2.circle(frame, points[8], 12, (0, 255, 0), -1)

    # Title
    cv2.putText(frame, 'AirOS - Gesture Detection v2.0',
               (10, frame.shape[0] - 20),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.7, (255, 255, 255), 2)

    cv2.imshow('AirOS Gesture Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Gesture Detection Stopped.")