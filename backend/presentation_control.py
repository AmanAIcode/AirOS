import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import pyautogui
from collections import deque

pyautogui.FAILSAFE = False

model_path = "models/hand_landmarker.task"

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# ─── State ────────────────────────────────────────────
last_action      = ""
last_action_time = 0
gesture_cooldown = 0
swipe_cooldown   = 0
pointer_mode     = False

SMOOTH_BUFFER = 6
pos_buffer = deque(maxlen=SMOOTH_BUFFER)
swipe_history = deque(maxlen=8)

screen_w, screen_h = pyautogui.size()

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

def is_index_only(fingers):
    return fingers == [0,1,0,0,0]

def is_peace(fingers):
    return fingers == [0,1,1,0,0]

def is_open_palm(fingers):
    return fingers == [1,1,1,1,1]

# ─── Smoothing ────────────────────────────────────────
def smooth_pos(x, y, buf):
    buf.append((x, y))
    ax = int(sum(p[0] for p in buf) / len(buf))
    ay = int(sum(p[1] for p in buf) / len(buf))
    return ax, ay

def draw_hand(frame, lm, show=True):
    h, w, _ = frame.shape
    pts = {}
    for i, p in enumerate(lm):
        x = int(p.x * w)
        y = int(p.y * h)
        pts[i] = (x, y)
        if show:
            cv2.circle(frame,(x,y),4,(0,255,255),-1)
    if show:
        for a, b in HAND_CONNECTIONS:
            if a in pts and b in pts:
                cv2.line(frame,pts[a],pts[b],(0,200,0),1)
    return pts

# ─── Functions ─────────────────────────────────────────
def next_slide():
    pyautogui.press('right')
    return "Next Slide ➡️"

def prev_slide():
    pyautogui.press('left')
    return "Previous Slide ⬅️"

def start_presentation():
    pyautogui.press('f5')
    return "Starting Presentation 🎬"

def end_presentation():
    pyautogui.press('esc')
    return "Ending Presentation ⏹️"

# ─── UI ────────────────────────────────────────────────
def draw_ui(frame, last_action, pointer_mode):
    h, w, _ = frame.shape

    cv2.rectangle(frame,(0,0),(w,50),(20,20,20),-1)
    cv2.putText(frame,'AirOS v8.0 - Presentation Controller',
               (10,32),cv2.FONT_HERSHEY_SIMPLEX,
               0.8,(0,255,255),2)

    pc = (0,255,0) if pointer_mode else (80,80,80)
    cv2.putText(frame,
               f'Pointer:{"ON" if pointer_mode else "OFF"}',
               (w-220,32),cv2.FONT_HERSHEY_SIMPLEX,
               0.6,pc,2)

    if last_action:
        cv2.rectangle(frame,(0,52),(w,90),(20,20,20),-1)
        cv2.putText(frame,f'► {last_action}',
                   (10,78),cv2.FONT_HERSHEY_SIMPLEX,
                   0.8,(0,255,0),2)

    guide = [
        "👉  SWIPE RIGHT  = Next Slide",
        "👈  SWIPE LEFT   = Previous Slide",
        "☝️  INDEX        = Laser Pointer ON/OFF",
        "🖐️  OPEN PALM    = Start Presentation (F5)",
        "✌️  PEACE        = End Presentation (ESC)",
    ]
    cv2.rectangle(frame,(10,100),
                 (380,100+len(guide)*32+10),
                 (20,20,20),-1)
    for i,g in enumerate(guide):
        cv2.putText(frame,g,(15,125+i*32),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.55,(180,180,180),1)

    cv2.rectangle(frame,(0,h-35),(w,h),(20,20,20),-1)
    cv2.putText(frame,
               'Open PowerPoint/Slides first, then use gestures  |  Q = Quit',
               (10,h-12),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.45,(150,150,150),1)
    return frame

# ─── MediaPipe ────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.6
)
detector = vision.HandLandmarker.create_from_options(options)

# ─── Webcam ───────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS,          30)

print("=" * 55)
print("   AirOS v8.0 Presentation Controller Started!")
print("=" * 55)
print("👉  SWIPE RIGHT   = Next Slide")
print("👈  SWIPE LEFT    = Previous Slide")
print("☝️  INDEX         = Laser Pointer ON/OFF")
print("🖐️  OPEN PALM     = Start (F5)")
print("✌️  PEACE         = End (ESC)")
print("Q              = Quit")
print("=" * 55)

# ─── Main Loop ────────────────────────────────────────
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
    swipe_cooldown   = max(0, swipe_cooldown - 1)

    if res.hand_landmarks:
        for lm in res.hand_landmarks:
            fingers = get_finger_states(lm)

            wrist = lm[0]
            wx = int(wrist.x * w)

            itip = lm[8]
            rix  = int(itip.x * w)
            riy  = int(itip.y * h)
            ix, iy = smooth_pos(rix, riy, pos_buffer)

            draw_hand(frame, lm, show=not pointer_mode)

            swipe_history.append(wx)

            # Swipe detection
            if len(swipe_history) == swipe_history.maxlen and swipe_cooldown == 0:
                diff = swipe_history[-1] - swipe_history[0]
                if abs(diff) > 150:
                    if diff > 0:
                        last_action = next_slide()
                    else:
                        last_action = prev_slide()
                    last_action_time = ct
                    swipe_cooldown = 25
                    swipe_history.clear()

            # INDEX ONLY = Toggle pointer mode
            if is_index_only(fingers) and gesture_cooldown == 0:
                pointer_mode = not pointer_mode
                last_action = "Laser Pointer " + ("ON 🔴" if pointer_mode else "OFF")
                last_action_time = ct
                gesture_cooldown = 25

            # If pointer mode — show laser dot at fingertip
            if pointer_mode and is_index_only(fingers):
                # Move actual mouse for laser pointer effect
                mx = int(np.interp(ix,[0,w],[0,screen_w]))
                my = int(np.interp(iy,[0,h],[0,screen_h]))
                pyautogui.moveTo(mx, my, duration=0.02)

                # Draw laser dot
                cv2.circle(frame,(ix,iy),15,(0,0,255),-1)
                cv2.circle(frame,(ix,iy),20,(0,0,255),2)

            # OPEN PALM = Start presentation
            if is_open_palm(fingers) and gesture_cooldown == 0:
                last_action = start_presentation()
                last_action_time = ct
                gesture_cooldown = 35

            # PEACE = End presentation
            elif is_peace(fingers) and gesture_cooldown == 0:
                last_action = end_presentation()
                last_action_time = ct
                gesture_cooldown = 35

    else:
        pos_buffer.clear()
        swipe_history.clear()

    if last_action and ct - last_action_time > 2.5:
        last_action = ""

    frame = draw_ui(frame, last_action, pointer_mode)

    cv2.imshow('AirOS Presentation Controller v8.0', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Presentation Controller Stopped.")