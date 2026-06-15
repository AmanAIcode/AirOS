import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import subprocess
import sys
import os
from collections import deque

model_path = "models/hand_landmarker.task"

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# ─── Menu Items ───────────────────────────────────────
MENU_ITEMS = [
    ("✍️  Air Drawing",         "air_drawing.py"),
    ("⌨️  Air Typing",           "air_typing.py"),
    ("🖥️  Desktop Control",      "desktop_control.py"),
    ("🎵  Media Control",        "media_control.py"),
    ("🤖  AI Assistant",         "ai_assistant.py"),
    ("📊  Presentation Control", "presentation_control.py"),
    ("❌  Quit AirOS",           None),
]

SMOOTH_BUFFER = 6
pos_buffer = deque(maxlen=SMOOTH_BUFFER)

# ─── Finger Detection ─────────────────────────────────
def get_finger_states(lm):
    fingers = []
    thumb_tip = lm[4]
    thumb_ip  = lm[3]
    wrist     = lm[0]
    index_mcp = lm[5]

    hand_dir = wrist.x - index_mcp.x
    if hand_dir > 0:
        thumb_open = thumb_tip.x > thumb_ip.x + 0.03
    else:
        thumb_open = thumb_tip.x < thumb_ip.x - 0.03
    fingers.append(1 if thumb_open else 0)

    tips = [8,  12, 16, 20]
    dips = [7,  11, 15, 19]
    pips = [6,  10, 14, 18]
    mcps = [5,   9, 13, 17]

    for tip, dip, pip, mcp in zip(tips, dips, pips, mcps):
        tip_lm = lm[tip]
        pip_lm = lm[pip]
        mcp_lm = lm[mcp]
        dip_lm = lm[dip]
        seg1 = abs(mcp_lm.y - pip_lm.y)
        seg2 = abs(pip_lm.y - dip_lm.y)
        threshold = (seg1 + seg2) * 0.3
        if tip_lm.y < pip_lm.y - threshold:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

def is_pinch(lm):
    thumb_tip  = lm[4]
    index_tip  = lm[8]
    wrist      = lm[0]
    mid_mcp    = lm[9]
    hand_size  = ((wrist.x - mid_mcp.x)**2 +
                  (wrist.y - mid_mcp.y)**2) ** 0.5
    dist       = ((thumb_tip.x - index_tip.x)**2 +
                  (thumb_tip.y - index_tip.y)**2) ** 0.5
    if hand_size > 0:
        return dist / hand_size < 0.18
    return dist < 0.05

def is_index_only(fingers):
    return fingers == [0,1,0,0,0]

# ─── Smoothing ────────────────────────────────────────
def smooth_pos(x, y, buf):
    buf.append((x, y))
    ax = int(sum(p[0] for p in buf) / len(buf))
    ay = int(sum(p[1] for p in buf) / len(buf))
    return ax, ay

def draw_hand(frame, lm):
    h, w, _ = frame.shape
    pts = {}
    for i, p in enumerate(lm):
        x = int(p.x * w)
        y = int(p.y * h)
        pts[i] = (x, y)
        cv2.circle(frame,(x,y),4,(0,255,255),-1)
    for a, b in HAND_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame,pts[a],pts[b],(0,200,0),1)
    return pts

# ─── UI ────────────────────────────────────────────────
def draw_menu(frame, hovered_idx, dwell_progress):
    h, w, _ = frame.shape

    overlay = frame.copy()
    cv2.rectangle(overlay,(0,0),(w,h),(10,10,30),-1)
    cv2.addWeighted(overlay,0.6,frame,0.4,0,frame)

    # Title
    cv2.putText(frame,'AirOS',
               (w//2-120,80),
               cv2.FONT_HERSHEY_SIMPLEX,
               2.0,(0,255,255),4)
    cv2.putText(frame,'Gesture Controlled Operating System',
               (w//2-220,120),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.7,(150,150,150),1)

    # Menu items
    item_h = 70
    item_w = 500
    start_y = 160
    start_x = (w - item_w)//2

    rects = []
    for i, (label, _) in enumerate(MENU_ITEMS):
        y = start_y + i * (item_h + 12)
        rect = (start_x, y, start_x+item_w, y+item_h)
        rects.append(rect)

        if i == hovered_idx:
            progress = dwell_progress
            color = (0, int(100+155*progress), 0)
            # Progress fill
            fill_w = int(item_w * progress)
            cv2.rectangle(frame,(start_x,y),
                         (start_x+fill_w,y+item_h),
                         color,-1)
            cv2.rectangle(frame,(start_x,y),
                         (start_x+item_w,y+item_h),
                         (0,255,0),3)
        else:
            cv2.rectangle(frame,(start_x,y),
                         (start_x+item_w,y+item_h),
                         (40,40,40),-1)
            cv2.rectangle(frame,(start_x,y),
                         (start_x+item_w,y+item_h),
                         (100,100,100),1)

        cv2.putText(frame,label,
                   (start_x+25,y+45),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.9,(255,255,255),2)

    cv2.rectangle(frame,(0,h-35),(w,h),(20,20,20),-1)
    cv2.putText(frame,
               'Point INDEX finger at item and hold 1.5s to select  |  PINCH = instant select',
               (10,h-12),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.5,(150,150,150),1)

    return frame, rects

# ─── MediaPipe ────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# ─── Webcam ───────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS,          30)

print("=" * 55)
print("   Welcome to AirOS - Main Menu")
print("=" * 55)
print("Point INDEX finger at item, hold 1.5s OR pinch to select")
print("=" * 55)

pos_buffer = deque(maxlen=SMOOTH_BUFFER)
hover_start = None
DWELL_TIME  = 1.5
gesture_cooldown = 0

backend_dir = os.path.dirname(os.path.abspath(__file__))

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    ct = time.time()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mpi = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = detector.detect(mpi)

    gesture_cooldown = max(0, gesture_cooldown - 1)

    # Draw menu first to get rects
    frame_copy = frame.copy()
    frame_copy, rects = draw_menu(frame_copy, -1, 0)

    hovered_idx   = -1
    dwell_progress = 0
    selected_idx  = -1

    if res.hand_landmarks:
        for lm in res.hand_landmarks:
            draw_hand(frame, lm)
            fingers = get_finger_states(lm)

            itip = lm[8]
            rix  = int(itip.x * w)
            riy  = int(itip.y * h)
            ix, iy = smooth_pos(rix, riy, pos_buffer)

            cv2.circle(frame,(ix,iy),10,(0,255,255),-1)
            cv2.circle(frame,(ix,iy),13,(255,255,255),2)

            # Check which item is hovered
            for i, (x1,y1,x2,y2) in enumerate(rects):
                if x1 < ix < x2 and y1 < iy < y2:
                    hovered_idx = i
                    break

            if hovered_idx >= 0:
                if hover_start is None:
                    hover_start = ct
                elapsed = ct - hover_start
                dwell_progress = min(elapsed/DWELL_TIME, 1.0)

                if dwell_progress >= 1.0 and gesture_cooldown == 0:
                    selected_idx = hovered_idx

                # Pinch = instant select
                if is_pinch(lm) and gesture_cooldown == 0:
                    selected_idx = hovered_idx
            else:
                hover_start = None
                dwell_progress = 0

    else:
        pos_buffer.clear()
        hover_start = None

    # Draw menu with hover state
    frame, rects = draw_menu(frame, hovered_idx, dwell_progress)

    cv2.imshow('AirOS Main Menu', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # Handle selection
    if selected_idx >= 0:
        label, script = MENU_ITEMS[selected_idx]
        if script is None:
            print("Goodbye! 👋")
            break
        else:
            print(f"Launching: {label}")
            cap.release()
            cv2.destroyAllWindows()

            script_path = os.path.join(backend_dir, script)
            subprocess.run([sys.executable, script_path])

            # Reopen webcam after returning
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS,          30)
            pos_buffer.clear()
            hover_start = None
            gesture_cooldown = 30

cap.release()
cv2.destroyAllWindows()
print("AirOS Closed.")