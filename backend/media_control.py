import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import time
import keyboard
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

# ─── State ────────────────────────────────────────────
last_action      = ""
last_action_time = 0
gesture_cooldown = 0
vol_cooldown     = 0
volume_level     = 50

SMOOTH_BUFFER = 6
pos_buffer = deque(maxlen=SMOOTH_BUFFER)

# Quit button position
QUIT_BTN = (1130, 10, 1270, 60)  # x1,y1,x2,y2

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

def is_peace(fingers):
    return fingers == [0, 1, 1, 0, 0]

def is_thumb_up(fingers):
    return fingers == [1, 0, 0, 0, 0]

def is_thumb_down(lm, fingers):
    thumb_tip = lm[4]
    wrist     = lm[0]
    return (fingers[1] == 0 and fingers[2] == 0 and
            fingers[3] == 0 and fingers[4] == 0 and
            thumb_tip.y > wrist.y + 0.05)

def is_index_only(fingers):
    return fingers == [0, 1, 0, 0, 0]

# ─── Smoothing ────────────────────────────────────────
def smooth_pos(x, y, buf):
    buf.append((x, y))
    ax = int(sum(p[0] for p in buf) / len(buf))
    ay = int(sum(p[1] for p in buf) / len(buf))
    return ax, ay

# ─── Hand Drawing ─────────────────────────────────────
def draw_hand(frame, lm, show=True):
    h, w, _ = frame.shape
    pts = {}
    for i, p in enumerate(lm):
        x = int(p.x * w)
        y = int(p.y * h)
        pts[i] = (x, y)
        if show:
            cv2.circle(frame,(x,y),5,(0,255,255),-1)
    if show:
        for a, b in HAND_CONNECTIONS:
            if a in pts and b in pts:
                cv2.line(frame,pts[a],pts[b],
                        (0,200,0),2)
    return pts

# ─── Media Control Functions ──────────────────────────
def play_pause():
    keyboard.send('play/pause media')
    return "Play/Pause ⏯️"

def next_track():
    keyboard.send('next track')
    return "Next Track ⏭️"

def prev_track():
    keyboard.send('previous track')
    return "Previous Track ⏮️"

def vol_up():
    keyboard.send('volume up')
    return "Volume Up 🔊"

def vol_down():
    keyboard.send('volume down')
    return "Volume Down 🔉"

# ─── UI ───────────────────────────────────────────────
def draw_ui(frame, last_action, swipe_direction, volume_level):
    h, w, _ = frame.shape

    cv2.rectangle(frame,(0,0),(w,50),(20,20,20),-1)
    cv2.putText(frame,'AirOS v6.2 - Media Control',
               (10,32),cv2.FONT_HERSHEY_SIMPLEX,
               0.8,(0,255,255),2)

    if last_action:
        cv2.rectangle(frame,(0,52),(w,90),(20,20,20),-1)
        cv2.putText(frame,f'► {last_action}',
                   (10,78),cv2.FONT_HERSHEY_SIMPLEX,
                   0.8,(0,255,0),2)

    if swipe_direction:
        cx, cy = w//2, h//2
        if swipe_direction == "RIGHT":
            cv2.arrowedLine(frame,(cx-100,cy),
                           (cx+100,cy),
                           (0,255,0),8,tipLength=0.3)
        elif swipe_direction == "LEFT":
            cv2.arrowedLine(frame,(cx+100,cy),
                           (cx-100,cy),
                           (0,255,0),8,tipLength=0.3)

    # Volume bar
    bx = w-55
    cv2.rectangle(frame,(bx,95),(bx+30,285),(40,40,40),-1)
    vh = int(190*volume_level/100)
    cv2.rectangle(frame,(bx,285-vh),(bx+30,285),
                 (0,200,255),-1)
    cv2.putText(frame,'VOL',(bx-5,300),
               cv2.FONT_HERSHEY_SIMPLEX,0.4,(200,200,200),1)
    cv2.putText(frame,f'{volume_level}%',(bx-12,316),
               cv2.FONT_HERSHEY_SIMPLEX,0.4,(200,200,200),1)

    # Gesture guide
    guide = [
        "✌️  TWO FINGERS = Play / Pause",
        "👍  THUMB UP    = Volume Up",
        "👎  THUMB DOWN  = Volume Down",
        "👉  SWIPE RIGHT = Next Track",
        "👈  SWIPE LEFT  = Previous Track",
    ]
    cv2.rectangle(frame,(10,100),
                 (370,100+len(guide)*32+10),
                 (20,20,20),-1)
    for i,g in enumerate(guide):
        cv2.putText(frame,g,(15,125+i*32),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.55,(180,180,180),1)

    # QUIT button
    x1,y1,x2,y2 = QUIT_BTN
    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,200),-1)
    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,255),2)
    cv2.putText(frame,'QUIT',(x1+25,y1+35),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.8,(255,255,255),2)

    cv2.rectangle(frame,(0,h-35),(w,h),(20,20,20),-1)
    cv2.putText(frame,
               'Point at QUIT button & hold to exit  |  Keyboard Q also works',
               (10,h-12),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.45,(150,150,150),1)
    return frame

# ─── MediaPipe Setup ──────────────────────────────────
base_options = python.BaseOptions(
    model_asset_path=model_path)
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
print("   AirOS v6.2 Media Control Started!")
print("=" * 55)
print("✌️  TWO FINGERS   = Play / Pause")
print("👍  THUMB UP      = Volume Up")
print("👎  THUMB DOWN    = Volume Down")
print("👉  SWIPE RIGHT   = Next Track")
print("👈  SWIPE LEFT    = Previous Track")
print("Point at QUIT button (hold 2s) or press Q")
print("=" * 55)

# ─── Main Loop ────────────────────────────────────────
gesture_cooldown    = 0
swipe_cooldown      = 0
vol_cooldown        = 0
pos_buffer          = deque(maxlen=SMOOTH_BUFFER)
swipe_history       = deque(maxlen=8)
swipe_direction     = None
swipe_display_time  = 0
volume_level        = 50
quit_hold_start     = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    ct = time.time()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mpi = mp.Image(image_format=mp.ImageFormat.SRGB,
                   data=rgb)
    res = detector.detect(mpi)

    gesture_cooldown = max(0, gesture_cooldown - 1)
    swipe_cooldown   = max(0, swipe_cooldown - 1)
    vol_cooldown     = max(0, vol_cooldown - 1)

    quit_now = False

    if res.hand_landmarks:
        for lm in res.hand_landmarks:
            draw_hand(frame, lm, show=True)
            fingers = get_finger_states(lm)

            wrist = lm[0]
            wx = int(wrist.x * w)

            itip   = lm[8]
            rix    = int(itip.x * w)
            riy    = int(itip.y * h)
            ix, iy = smooth_pos(rix, riy, pos_buffer)

            cv2.circle(frame,(ix,iy),10,(0,255,255),-1)
            cv2.circle(frame,(ix,iy),13,(255,255,255),2)

            # Check QUIT button hover (index finger)
            x1,y1,x2,y2 = QUIT_BTN
            if is_index_only(fingers) and x1<ix<x2 and y1<iy<y2:
                if quit_hold_start == 0:
                    quit_hold_start = ct
                progress = min((ct-quit_hold_start)/2.0, 1.0)
                cv2.rectangle(frame,(x1,y2-int((y2-y1)*progress)),
                             (x2,y2),(0,255,0),-1)
                cv2.putText(frame,'QUIT',(x1+25,y1+35),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.8,(255,255,255),2)
                if progress >= 1.0:
                    quit_now = True
            else:
                quit_hold_start = 0

            swipe_history.append(wx)

            if len(swipe_history) == swipe_history.maxlen and swipe_cooldown == 0:
                diff = swipe_history[-1] - swipe_history[0]
                if abs(diff) > 150:
                    if diff > 0:
                        last_action      = next_track()
                        swipe_direction  = "RIGHT"
                    else:
                        last_action      = prev_track()
                        swipe_direction  = "LEFT"
                    last_action_time    = ct
                    swipe_display_time  = ct
                    swipe_cooldown      = 25
                    swipe_history.clear()

            # TWO FINGERS = Play/Pause
            if is_peace(fingers) and gesture_cooldown == 0:
                last_action      = play_pause()
                last_action_time = ct
                gesture_cooldown = 30

            # THUMB UP = Volume Up
            elif is_thumb_up(fingers) and vol_cooldown == 0:
                volume_level     = min(volume_level+10,100)
                last_action      = vol_up()
                last_action_time = ct
                vol_cooldown     = 15

            # THUMB DOWN = Volume Down
            elif is_thumb_down(lm,fingers) and vol_cooldown == 0:
                volume_level     = max(volume_level-10,0)
                last_action      = vol_down()
                last_action_time = ct
                vol_cooldown     = 15

    else:
        pos_buffer.clear()
        swipe_history.clear()
        quit_hold_start = 0

    if last_action and ct - last_action_time > 2:
        last_action = ""

    if swipe_direction and ct - swipe_display_time > 0.5:
        swipe_direction = None

    frame = draw_ui(frame, last_action, swipe_direction, volume_level)

    cv2.imshow('AirOS Media Control v6.2', frame)

    if quit_now or cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("AirOS Media Control Stopped.")