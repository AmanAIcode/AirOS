import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import time
from collections import deque

model_path = "models/hand_landmarker.task"

# Drawing settings
draw_color = (0, 0, 255)
brush_size = 8
eraser_size = 40
mode = "OFF"

# Smoothing buffer — stores last N finger positions
SMOOTH_BUFFER = 5
pos_buffer = deque(maxlen=SMOOTH_BUFFER)

# Color palette
colors = {
    'Red':    (0, 0, 255),
    'Green':  (0, 255, 0),
    'Blue':   (255, 0, 0),
    'Yellow': (0, 255, 255),
    'Purple': (255, 0, 255),
    'White':  (255, 255, 255),
    'Orange': (0, 165, 255),
}
color_names = list(colors.keys())
current_color_idx = 0

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

def get_finger_states(hand_landmarks):
    """
    Improved finger detection with better accuracy
    Uses multiple reference points for each finger
    """
    fingers = []
    h_lm = hand_landmarks

    # --- THUMB ---
    # Compare thumb tip x vs thumb IP joint x
    # Also check wrist position for left/right hand
    thumb_tip = h_lm[4]
    thumb_ip  = h_lm[3]
    thumb_mcp = h_lm[2]
    index_mcp = h_lm[5]

    # Better thumb detection using angle
    if thumb_tip.x < thumb_ip.x and thumb_tip.x < index_mcp.x:
        fingers.append(1)  # Thumb extended left
    elif thumb_tip.x > thumb_ip.x and thumb_tip.x > index_mcp.x:
        fingers.append(1)  # Thumb extended right
    else:
        fingers.append(0)

    # --- INDEX FINGER ---
    # Tip must be significantly above PIP joint
    tip_ids  = [8,  12, 16, 20]
    pip_ids  = [6,  10, 14, 18]
    mcp_ids  = [5,   9, 13, 17]

    for tip_id, pip_id, mcp_id in zip(tip_ids, pip_ids, mcp_ids):
        tip = h_lm[tip_id]
        pip = h_lm[pip_id]
        mcp = h_lm[mcp_id]

        # Finger length for dynamic threshold
        finger_length = abs(mcp.y - pip.y)
        threshold = finger_length * 0.3

        # Finger is UP if tip is above PIP by threshold
        if tip.y < pip.y - threshold:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers  # [thumb, index, middle, ring, pinky]


def get_smooth_position(ix, iy, pos_buffer):
    """Smooth finger position using moving average"""
    pos_buffer.append((ix, iy))
    avg_x = int(sum(p[0] for p in pos_buffer) / len(pos_buffer))
    avg_y = int(sum(p[1] for p in pos_buffer) / len(pos_buffer))
    return avg_x, avg_y


def get_pinch_distance(hand_landmarks):
    """Improved pinch detection"""
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    # Normalize by hand size for better accuracy
    wrist = hand_landmarks[0]
    middle_mcp = hand_landmarks[9]
    hand_size = ((wrist.x - middle_mcp.x)**2 +
                 (wrist.y - middle_mcp.y)**2) ** 0.5
    raw_dist = ((thumb_tip.x - index_tip.x)**2 +
                (thumb_tip.y - index_tip.y)**2) ** 0.5
    # Normalized distance
    if hand_size > 0:
        return raw_dist / hand_size
    return raw_dist


def draw_landmarks(frame, hand_landmarks, show_skeleton=True):
    h, w, _ = frame.shape
    points = {}
    for i, lm in enumerate(hand_landmarks):
        x = int(lm.x * w)
        y = int(lm.y * h)
        points[i] = (x, y)
        if show_skeleton:
            cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)
    if show_skeleton:
        for connection in HAND_CONNECTIONS:
            start = points.get(connection[0])
            end = points.get(connection[1])
            if start and end:
                cv2.line(frame, start, end, (0, 255, 0), 1)
    return points


def draw_ui(frame, mode, draw_color, brush_size,
            color_names, current_color_idx):
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)

    for i, name in enumerate(color_names):
        color = colors[name]
        x = 10 + i * 55
        cv2.rectangle(frame, (x, 8), (x+45, 50), color, -1)
        if i == current_color_idx:
            cv2.rectangle(frame, (x, 8), (x+45, 50),
                         (255, 255, 255), 3)

    if mode == "DRAW":
        mode_color = (0, 255, 0)
    elif mode == "ERASE":
        mode_color = (0, 0, 255)
    else:
        mode_color = (150, 150, 150)

    cv2.putText(frame, f'Mode: {mode}',
               (w-200, 35),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.7, mode_color, 2)

    cv2.putText(frame, f'Brush: {brush_size}',
               (w-350, 35),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.7, (255, 255, 255), 2)

    cv2.rectangle(frame, (0, h-40), (w, h), (30, 30, 30), -1)
    cv2.putText(frame,
               '4FIN=Draw ON/OFF | PINKY=Erase | PEACE=Color | 3FIN=Bigger | THUMB=Smaller | PALM=Clear | S=Save | Q=Quit',
               (10, h-12),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.4, (200, 200, 200), 1)
    return frame


# Setup MediaPipe with better settings
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.6,  # Slightly lower = detects faster
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5         # Lower = smoother tracking
)
detector = vision.HandLandmarker.create_from_options(options)

# Open webcam with better resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

ret, first_frame = cap.read()
first_frame = cv2.flip(first_frame, 1)
canvas = np.zeros_like(first_frame)

print("Air Drawing Started! (Improved Accuracy)")
print("4 FINGERS  = Toggle Draw ON/OFF")
print("INDEX ALONE = Draw when mode is ON")
print("PINKY       = Erase mode")
print("PEACE       = Next Color")
print("3 FINGERS   = Bigger brush")
print("THUMBS UP   = Smaller brush")
print("OPEN PALM   = Clear canvas")
print("S = Save | Q = Quit")

prev_x, prev_y = 0, 0
gesture_cooldown = 0
pos_buffer = deque(maxlen=SMOOTH_BUFFER)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas.shape != frame.shape:
        canvas = np.zeros_like(frame)

    # Better preprocessing — improve contrast
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )
    results = detector.detect(mp_image)

    gesture_cooldown = max(0, gesture_cooldown - 1)

    if results.hand_landmarks:
        for hand_landmark in results.hand_landmarks:

            show_skeleton = mode != "DRAW"
            points = draw_landmarks(frame, hand_landmark,
                                   show_skeleton)
            fingers = get_finger_states(hand_landmark)

            # Get raw index fingertip
            index_tip = hand_landmark[8]
            raw_ix = int(index_tip.x * w)
            raw_iy = int(index_tip.y * h)

            # Smooth the position
            ix, iy = get_smooth_position(raw_ix, raw_iy,
                                         pos_buffer)

            # Skip UI area
            if iy < 65 or iy > h - 45:
                prev_x, prev_y = 0, 0
                continue

            # Show cursor
            if mode == "DRAW":
                # Neon glow effect on cursor
                cv2.circle(frame, (ix, iy),
                          brush_size + 4,
                          tuple(c//3 for c in draw_color), -1)
                cv2.circle(frame, (ix, iy),
                          brush_size, draw_color, -1)
            elif mode == "ERASE":
                cv2.circle(frame, (ix, iy),
                          eraser_size, (255, 255, 255), 2)
                cv2.circle(frame, (ix, iy),
                          eraser_size - 4, (100, 100, 100), 1)
            else:
                cv2.circle(frame, (ix, iy),
                          8, (150, 150, 150), -1)

            # FOUR FINGERS = Toggle Draw ON/OFF
            if fingers == [0,1,1,1,1] and gesture_cooldown == 0:
                if mode == "OFF" or mode == "ERASE":
                    mode = "DRAW"
                    pos_buffer.clear()
                    print("Draw mode ON ✅")
                else:
                    mode = "OFF"
                    pos_buffer.clear()
                    print("Draw mode OFF ⛔")
                gesture_cooldown = 25

            # PINKY ONLY = Erase mode
            elif fingers == [0,0,0,0,1] and gesture_cooldown == 0:
                mode = "ERASE"
                pos_buffer.clear()
                print("Erase mode ON 🔴")
                gesture_cooldown = 20

            # PEACE = Next color
            elif fingers == [0,1,1,0,0] and gesture_cooldown == 0:
                current_color_idx = (current_color_idx + 1) % len(color_names)
                draw_color = colors[color_names[current_color_idx]]
                mode = "DRAW"
                print(f"Color: {color_names[current_color_idx]} ✅")
                gesture_cooldown = 20

            # THREE FINGERS = Bigger brush
            elif fingers == [0,1,1,1,0] and gesture_cooldown == 0:
                brush_size = min(brush_size + 2, 40)
                mode = "DRAW"
                print(f"Brush size: {brush_size}")
                gesture_cooldown = 20

            # THUMBS UP = Smaller brush
            elif fingers == [1,0,0,0,0] and gesture_cooldown == 0:
                brush_size = max(brush_size - 2, 2)
                mode = "DRAW"
                print(f"Brush size: {brush_size}")
                gesture_cooldown = 20

            # OPEN PALM = Clear canvas
            elif fingers == [1,1,1,1,1] and gesture_cooldown == 0:
                canvas = np.zeros_like(frame)
                mode = "DRAW"
                pos_buffer.clear()
                print("Canvas cleared! 🗑️")
                gesture_cooldown = 30

            # DRAW with index finger only
            if mode == "DRAW" and fingers == [0,1,0,0,0]:
                if prev_x != 0 and prev_y != 0:
                    # Interpolate between points for smoother lines
                    dist = ((ix - prev_x)**2 + (iy - prev_y)**2)**0.5
                    if dist < 100:  # Ignore huge jumps
                        cv2.line(canvas,
                                (prev_x, prev_y),
                                (ix, iy),
                                draw_color, brush_size,
                                lineType=cv2.LINE_AA)  # Anti-aliased line
                prev_x, prev_y = ix, iy

            # ERASE
            elif mode == "ERASE":
                cv2.circle(canvas, (ix, iy),
                          eraser_size, (0, 0, 0), -1)
                prev_x, prev_y = ix, iy

            else:
                prev_x, prev_y = 0, 0

    else:
        prev_x, prev_y = 0, 0
        pos_buffer.clear()

    # Blend canvas with frame
    drawing_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, drawing_mask = cv2.threshold(
        drawing_gray, 10, 255, cv2.THRESH_BINARY)
    drawing_mask_inv = cv2.bitwise_not(drawing_mask)
    frame_bg = cv2.bitwise_and(
        frame, frame, mask=drawing_mask_inv)
    drawing_fg = cv2.bitwise_and(
        canvas, canvas, mask=drawing_mask)
    combined = cv2.add(frame_bg, drawing_fg)

    combined = draw_ui(combined, mode, draw_color,
                      brush_size, color_names,
                      current_color_idx)

    cv2.imshow('AirOS Air Drawing', combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        filename = f'assets/drawing_{int(time.time())}.png'
        cv2.imwrite(filename, canvas)
        print(f"Saved: {filename} ✅")
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Air Drawing Stopped.")