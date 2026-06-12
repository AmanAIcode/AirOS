import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import time
from collections import deque
import pyautogui

model_path = "models/hand_landmarker.task"

# Keyboard layout
KEYS = [
    ['Q','W','E','R','T','Y','U','I','O','P'],
    ['A','S','D','F','G','H','J','K','L'],
    ['Z','X','C','V','B','N','M','⌫'],
    ['SPACE','CLEAR','ENTER']
]

# Keyboard settings
KEY_WIDTH  = 65
KEY_HEIGHT = 65
KEY_MARGIN = 8
START_X    = 30
START_Y    = 200

# Colors
COLOR_KEY_BG      = (40, 40, 40)
COLOR_KEY_HOVER   = (0, 180, 0)
COLOR_KEY_PRESSED = (0, 255, 0)
COLOR_KEY_TEXT    = (255, 255, 255)
COLOR_KEY_BORDER  = (100, 100, 100)

# State
typed_text    = ""
hovered_key   = None
pressed_key   = None
dwell_start   = {}
DWELL_TIME    = 1.2  # seconds to dwell before keypress
last_pressed  = None
last_press_time = 0
COOLDOWN      = 0.5  # seconds between keypresses

# Smoothing
SMOOTH_BUFFER = 5
pos_buffer = deque(maxlen=SMOOTH_BUFFER)

# Word suggestions (simple)
COMMON_WORDS = [
    "the", "and", "for", "are", "but", "not", "you", "all",
    "can", "had", "her", "was", "one", "our", "out", "day",
    "get", "has", "him", "his", "how", "its", "may", "new",
    "now", "old", "see", "two", "way", "who", "did", "did",
    "hello", "world", "python", "project", "airos", "hand",
    "gesture", "drawing", "typing", "computer", "screen"
]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

def get_finger_states(hand_landmarks):
    fingers = []
    h_lm = hand_landmarks
    thumb_tip  = h_lm[4]
    thumb_ip   = h_lm[3]
    index_mcp  = h_lm[5]

    if thumb_tip.x < thumb_ip.x and thumb_tip.x < index_mcp.x:
        fingers.append(1)
    elif thumb_tip.x > thumb_ip.x and thumb_tip.x > index_mcp.x:
        fingers.append(1)
    else:
        fingers.append(0)

    tip_ids = [8,  12, 16, 20]
    pip_ids = [6,  10, 14, 18]
    mcp_ids = [5,   9, 13, 17]

    for tip_id, pip_id, mcp_id in zip(tip_ids, pip_ids, mcp_ids):
        tip = h_lm[tip_id]
        pip = h_lm[pip_id]
        mcp = h_lm[mcp_id]
        finger_length = abs(mcp.y - pip.y)
        threshold = finger_length * 0.3
        if tip.y < pip.y - threshold:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

def get_pinch_distance(hand_landmarks):
    thumb_tip  = hand_landmarks[4]
    index_tip  = hand_landmarks[8]
    wrist      = hand_landmarks[0]
    middle_mcp = hand_landmarks[9]
    hand_size  = ((wrist.x - middle_mcp.x)**2 +
                  (wrist.y - middle_mcp.y)**2) ** 0.5
    raw_dist   = ((thumb_tip.x - index_tip.x)**2 +
                  (thumb_tip.y - index_tip.y)**2) ** 0.5
    if hand_size > 0:
        return raw_dist / hand_size
    return raw_dist

def get_smooth_position(ix, iy, pos_buffer):
    pos_buffer.append((ix, iy))
    avg_x = int(sum(p[0] for p in pos_buffer) / len(pos_buffer))
    avg_y = int(sum(p[1] for p in pos_buffer) / len(pos_buffer))
    return avg_x, avg_y

def draw_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape
    points = {}
    for i, lm in enumerate(hand_landmarks):
        x = int(lm.x * w)
        y = int(lm.y * h)
        points[i] = (x, y)
        cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)
    for connection in HAND_CONNECTIONS:
        start = points.get(connection[0])
        end   = points.get(connection[1])
        if start and end:
            cv2.line(frame, start, end, (0, 255, 0), 1)
    return points

def get_key_rect(row_idx, col_idx, key):
    """Get rectangle coords for a key"""
    if key == 'SPACE':
        x = START_X
        w = KEY_WIDTH * 5 + KEY_MARGIN * 4
    elif key == 'CLEAR':
        x = START_X + KEY_WIDTH * 5 + KEY_MARGIN * 5
        w = KEY_WIDTH * 2 + KEY_MARGIN
    elif key == 'ENTER':
        x = START_X + KEY_WIDTH * 7 + KEY_MARGIN * 7
        w = KEY_WIDTH * 3 + KEY_MARGIN * 2
    else:
        x = START_X + col_idx * (KEY_WIDTH + KEY_MARGIN)
        w = KEY_WIDTH

    y = START_Y + row_idx * (KEY_HEIGHT + KEY_MARGIN)
    return x, y, w, KEY_HEIGHT

def draw_keyboard(frame, hovered_key, pressed_key,
                  dwell_start, typed_text):
    h, w, _ = frame.shape

    # Background panel
    kb_h = len(KEYS) * (KEY_HEIGHT + KEY_MARGIN) + 20
    overlay = frame.copy()
    cv2.rectangle(overlay,
                 (START_X - 10, START_Y - 10),
                 (w - 20, START_Y + kb_h),
                 (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    for row_idx, row in enumerate(KEYS):
        for col_idx, key in enumerate(row):
            x, y, kw, kh = get_key_rect(row_idx, col_idx, key)

            # Key color
            if key == pressed_key:
                bg_color = COLOR_KEY_PRESSED
            elif key == hovered_key:
                # Dwell progress
                if key in dwell_start:
                    elapsed = time.time() - dwell_start[key]
                    progress = min(elapsed / DWELL_TIME, 1.0)
                    g = int(100 + 155 * progress)
                    bg_color = (0, g, 0)
                else:
                    bg_color = COLOR_KEY_HOVER
            else:
                bg_color = COLOR_KEY_BG

            # Draw key
            cv2.rectangle(frame, (x, y),
                         (x + kw, y + kh),
                         bg_color, -1)
            cv2.rectangle(frame, (x, y),
                         (x + kw, y + kh),
                         COLOR_KEY_BORDER, 1)

            # Key text
            font_scale = 0.5 if len(key) > 2 else 0.7
            text_size = cv2.getTextSize(
                key, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, 2)[0]
            text_x = x + (kw - text_size[0]) // 2
            text_y = y + (kh + text_size[1]) // 2
            cv2.putText(frame, key,
                       (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale,
                       COLOR_KEY_TEXT, 2)

    # Typed text display box
    cv2.rectangle(frame, (START_X - 10, START_Y - 80),
                 (w - 20, START_Y - 15),
                 (20, 20, 20), -1)
    cv2.rectangle(frame, (START_X - 10, START_Y - 80),
                 (w - 20, START_Y - 15),
                 (100, 100, 100), 1)

    # Show typed text
    display_text = typed_text[-50:] if len(typed_text) > 50 else typed_text
    cv2.putText(frame, display_text + "|",
               (START_X, START_Y - 30),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.8, (0, 255, 0), 2)

    # Word suggestions
    if typed_text:
        last_word = typed_text.split()[-1].lower() if typed_text.split() else ""
        suggestions = [w for w in COMMON_WORDS
                      if w.startswith(last_word) and w != last_word][:3]
        if suggestions:
            sx = START_X
            cv2.putText(frame, "Suggestions: ",
                       (sx, START_Y - 55),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (150, 150, 150), 1)
            for i, sug in enumerate(suggestions):
                cv2.putText(frame, f"[{sug}]",
                           (sx + 110 + i * 120, START_Y - 55),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 200, 255), 1)

    return frame

def process_keypress(key, typed_text):
    """Handle key press logic"""
    if key == '⌫':
        typed_text = typed_text[:-1]
    elif key == 'SPACE':
        typed_text += ' '
    elif key == 'CLEAR':
        typed_text = ''
    elif key == 'ENTER':
        typed_text += '\n'
        # Type into active app
        try:
            pyautogui.hotkey('enter')
        except:
            pass
    else:
        typed_text += key
        # Type into active app
        try:
            pyautogui.typewrite(key.lower(), interval=0.01)
        except:
            pass
    return typed_text

# Setup MediaPipe
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Air Typing Started!")
print("HOVER finger over key for 1.2 seconds to type")
print("Q key on keyboard to quit")

pos_buffer = deque(maxlen=SMOOTH_BUFFER)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame  = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )
    results = detector.detect(mp_image)

    hovered_key  = None
    current_time = time.time()

    if results.hand_landmarks:
        for hand_landmark in results.hand_landmarks:
            points  = draw_landmarks(frame, hand_landmark)
            fingers = get_finger_states(hand_landmark)
            pinch   = get_pinch_distance(hand_landmark)

            # Get smooth index fingertip
            index_tip = hand_landmark[8]
            raw_ix = int(index_tip.x * w)
            raw_iy = int(index_tip.y * h)
            ix, iy = get_smooth_position(raw_ix, raw_iy,
                                         pos_buffer)

            # Draw cursor
            cv2.circle(frame, (ix, iy), 10, (0, 255, 255), -1)
            cv2.circle(frame, (ix, iy), 12, (255, 255, 255), 2)

            # Check which key is hovered
            for row_idx, row in enumerate(KEYS):
                for col_idx, key in enumerate(row):
                    x, y, kw, kh = get_key_rect(
                        row_idx, col_idx, key)
                    if x < ix < x + kw and y < iy < y + kh:
                        hovered_key = key
                        break

            # Dwell typing — hover for DWELL_TIME seconds
            if hovered_key:
                if hovered_key not in dwell_start:
                    dwell_start[hovered_key] = current_time
                else:
                    elapsed = current_time - dwell_start[hovered_key]
                    if elapsed >= DWELL_TIME:
                        # Check cooldown
                        if (current_time - last_press_time > COOLDOWN):
                            typed_text = process_keypress(
                                hovered_key, typed_text)
                            pressed_key    = hovered_key
                            last_pressed   = hovered_key
                            last_press_time = current_time
                            dwell_start    = {}
                            print(f"Typed: {hovered_key}")
            else:
                # Clear dwell for keys not hovered
                dwell_start = {k: v for k, v in
                              dwell_start.items()
                              if k == hovered_key}

            # Pinch to type instantly
            if pinch < 0.15 and hovered_key:
                if current_time - last_press_time > COOLDOWN:
                    typed_text = process_keypress(
                        hovered_key, typed_text)
                    pressed_key     = hovered_key
                    last_press_time = current_time
                    dwell_start     = {}
                    print(f"Typed: {hovered_key}")
    else:
        pos_buffer.clear()
        dwell_start = {}

    # Reset pressed key visual after short time
    if pressed_key and current_time - last_press_time > 0.2:
        pressed_key = None

    # Draw keyboard
    frame = draw_keyboard(frame, hovered_key,
                         pressed_key, dwell_start,
                         typed_text)

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 40), (30, 30, 30), -1)
    cv2.putText(frame, 'AirOS - Air Typing v4.0  |  Hover finger on key to type  |  ESC = Quit',
               (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.55, (200, 200, 200), 1)

    cv2.imshow('AirOS Air Typing', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
print(f"Final typed text: {typed_text}")
print("Air Typing Stopped.")