import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import time
import pyautogui
from collections import deque

pyautogui.FAILSAFE = False

model_path = "models/hand_landmarker.task"

# ─── App Commands ─────────────────────────────────────
APP_COMMANDS = {
    'C': ('Chrome',   'start chrome'),
    'Y': ('YouTube',  'start chrome https://youtube.com'),
    'S': ('Spotify',  'start spotify'),
    'V': ('VS Code',  'code .'),
    'N': ('Notepad',  'notepad'),
    'F': ('Explorer', 'explorer'),
    'T': ('TaskMgr',  'taskmgr'),
    'Z': ('Zoom',     'start zoom'),
    'W': ('WhatsApp', 'start whatsapp'),
    'G': ('Google',   'start chrome https://google.com'),
    'E': ('Edge',     'start msedge'),
    'P': ('Paint',    'mspaint'),
}

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
volume_level     = 50
mouse_mode       = False

# Mouse
prev_mouse_x = 0
prev_mouse_y = 0
hold_start   = 0
hold_progress= 0

# Volume
vol_cooldown = 0

# Drawing mode
draw_mode         = False
letter_points     = []
recognized_letter = ""
draw_mode_start   = 0
prev_draw_x       = 0
prev_draw_y       = 0
LETTER_TIMEOUT    = 8.0

# Peace toggle cooldown
peace_cooldown = 0

# Smoothing
SMOOTH_BUFFER = 6
DRAW_BUFFER   = 10
pos_buffer    = deque(maxlen=SMOOTH_BUFFER)
draw_buffer   = deque(maxlen=DRAW_BUFFER)

# ─── Finger Detection ─────────────────────────────────
def get_finger_states(lm):
    fingers = []

    # Thumb
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

    # 4 fingers
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

def is_index_only(fingers):
    return fingers == [0, 1, 0, 0, 0]

def is_thumb_up(fingers):
    return fingers == [1, 0, 0, 0, 0]

def is_thumb_down(lm, fingers):
    thumb_tip = lm[4]
    wrist     = lm[0]
    return (fingers[1] == 0 and
            fingers[2] == 0 and
            fingers[3] == 0 and
            fingers[4] == 0 and
            thumb_tip.y > wrist.y + 0.05)

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

def is_three_fingers(fingers):
    return fingers == [0, 1, 1, 1, 0]

def is_rock(fingers):
    return fingers == [0, 1, 0, 0, 1]

def is_pinky_only(fingers):
    return fingers == [0, 0, 0, 0, 1]

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

# ─── Letter Recognition ───────────────────────────────
def normalize_pts(pts):
    if len(pts) < 2:
        return pts
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mnx,mxx = min(xs),max(xs)
    mny,mxy = min(ys),max(ys)
    rx = mxx-mnx or 1
    ry = mxy-mny or 1
    return [((p[0]-mnx)/rx,(p[1]-mny)/ry) for p in pts]

def recognize_letter(pts):
    if len(pts) < 15:
        return None

    norm  = normalize_pts(pts)
    xs    = [p[0] for p in norm]
    ys    = [p[1] for p in norm]
    n     = len(norm)
    start = norm[0]
    end   = norm[-1]
    mid   = norm[n//2]

    dx  = end[0] - start[0]
    dy  = end[1] - start[1]
    w_  = max(xs) - min(xs)
    h_  = max(ys) - min(ys)
    asp = w_ / (h_ + 0.001)

    hc = vc = 0
    for i in range(2, n):
        if (norm[i-1][0]-norm[i-2][0])*(norm[i][0]-norm[i-1][0]) < -0.001:
            hc += 1
        if (norm[i-1][1]-norm[i-2][1])*(norm[i][1]-norm[i-1][1]) < -0.001:
            vc += 1

    cx = sum(xs)/len(xs)
    cy = sum(ys)/len(ys)
    ar = sum(((p[0]-cx)**2+(p[1]-cy)**2)**0.5 for p in norm)/n
    rv = sum(abs(((p[0]-cx)**2+(p[1]-cy)**2)**0.5-ar) for p in norm)/n
    is_circ = rv < 0.15 and ar > 0.2

    if (start[0] > 0.4 and end[0] > 0.3 and
        mid[0] < 0.3 and h_ > 0.4 and not is_circ):
        return 'C'
    if (is_circ and abs(start[0]-end[0]) < 0.3 and
        abs(start[1]-end[1]) < 0.3):
        return 'O'
    if h_ > 0.6 and w_ < 0.2 and asp < 0.3:
        return 'I'
    if (dy > 0.5 and dx > 0.25 and
        mid[1] > 0.5 and vc < 3 and hc < 3):
        return 'L'
    if w_ > 0.5 and h_ < 0.3 and asp > 1.5:
        return 'T'
    if (start[1] < 0.35 and end[1] < 0.35 and
        mid[1] > 0.65 and hc >= 1):
        return 'V'
    if (start[1] > 0.5 and end[1] > 0.5 and
        mid[1] < 0.4 and vc > 3):
        return 'N'
    if (start[0] < 0.3 and end[0] > 0.6 and
        hc > 1 and vc > 1 and dx > 0.3):
        return 'Z'
    if (hc > 3 and vc > 3 and
        abs(dx) < 0.35 and not is_circ):
        return 'S'
    if (start[1] < 0.3 and end[0] > 0.45 and
        end[1] > 0.35 and w_ > 0.4 and hc > 2):
        return 'G'
    if w_ > 0.6 and vc > 4 and abs(dy) < 0.35:
        return 'W'
    if (mid[0] > 0.3 and mid[0] < 0.7 and
        end[1] > 0.65 and vc > 2):
        return 'Y'
    if dy > 0.5 and hc > 2 and start[0] < 0.4:
        return 'F'
    if h_ > 0.6 and hc > 3 and dx > 0.2:
        return 'E'
    if (start[1] < 0.2 and dy > 0.5 and hc > 2):
        return 'P'
    return None

# ─── System Functions ─────────────────────────────────
def open_app(letter):
    if letter in APP_COMMANDS:
        name, cmd = APP_COMMANDS[letter]
        os.system(cmd)
        return f"Opening {name} ✅"
    return f"No app for '{letter}' ❌"

def take_screenshot():
    try:
        fn = f'assets/screenshot_{int(time.time())}.png'
        pyautogui.screenshot(fn)
        return "Screenshot saved 📸"
    except:
        return "Screenshot failed"

def switch_app():
    try:
        pyautogui.hotkey('alt','tab')
        return "Switched App 🔄"
    except:
        return ""

def bring_back():
    try:
        pyautogui.hotkey('win','d')
        time.sleep(0.2)
        pyautogui.hotkey('alt','tab')
        return "Back to AirOS 🎥"
    except:
        return ""

def vol_up():
    for _ in range(3):
        pyautogui.press('volumeup')
    return "Volume Up 🔊"

def vol_down():
    for _ in range(3):
        pyautogui.press('volumedown')
    return "Volume Down 🔉"

def vol_mute():
    pyautogui.press('volumemute')
    return "Muted 🔇"

# ─── UI ───────────────────────────────────────────────
def draw_ui(frame, last_action, vol, mouse_mode,
            hold_prog, draw_mode, recognized_letter,
            letter_points):
    h, w, _ = frame.shape

    # Top bar
    cv2.rectangle(frame,(0,0),(w,50),(20,20,20),-1)
    cv2.putText(frame,'AirOS v5.3 - Desktop Control',
               (10,32),cv2.FONT_HERSHEY_SIMPLEX,
               0.8,(0,255,255),2)

    mc = (0,255,0) if mouse_mode else (80,80,80)
    cv2.putText(frame,
               f'Mouse:{"ON" if mouse_mode else "OFF"}',
               (w-160,32),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.6,mc,2)

    # Action
    if last_action:
        cv2.rectangle(frame,(0,52),(w,88),(20,20,20),-1)
        cv2.putText(frame,f'► {last_action}',
                   (10,75),cv2.FONT_HERSHEY_SIMPLEX,
                   0.7,(0,255,0),2)

    # Hold progress
    if hold_prog > 0:
        bw = int(w*hold_prog)
        cv2.rectangle(frame,(0,48),(bw,54),(0,255,0),-1)

    # ── DRAW MODE ────────────────────────────────────
    if draw_mode:
        overlay = frame.copy()
        cv2.rectangle(overlay,(0,0),(w,h),(0,0,60),-1)
        cv2.addWeighted(overlay,0.35,frame,0.65,0,frame)

        # Draw mode banner
        cv2.rectangle(frame,(0,55),(w,210),(15,15,40),-1)
        cv2.putText(frame,'✍️  LETTER DRAW MODE',
                   (w//2-185,100),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   1.1,(0,255,255),3)
        cv2.putText(frame,
                   'Use INDEX FINGER to draw letter slowly',
                   (w//2-230,140),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.65,(200,200,200),2)
        cv2.putText(frame,
                   'Show ✌️ PEACE again to confirm letter',
                   (w//2-210,175),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.6,(180,180,180),1)

        # Draw trail with glow effect
        if len(letter_points) > 1:
            for i in range(1, len(letter_points)):
                # Outer glow
                cv2.line(frame,
                        letter_points[i-1],
                        letter_points[i],
                        (0,100,100), 8,
                        lineType=cv2.LINE_AA)
                # Inner bright line
                cv2.line(frame,
                        letter_points[i-1],
                        letter_points[i],
                        (0,255,255), 3,
                        lineType=cv2.LINE_AA)

        # Recognized letter
        if recognized_letter:
            cv2.putText(frame,
                       f'✅  {recognized_letter}',
                       (w//2-80, h//2+50),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       3.0,(0,255,0),6)

        # Points counter
        cv2.putText(frame,
                   f'Points captured: {len(letter_points)}',
                   (10,h-60),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.5,(120,120,120),1)

        # App hint bottom
        cv2.rectangle(frame,(0,h-45),(w,h),(20,20,20),-1)
        cv2.putText(frame,
                   'C=Chrome  Y=YouTube  S=Spotify  V=VSCode  N=Notepad  F=Explorer  G=Google  E=Edge  P=Paint',
                   (10,h-18),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.38,(0,200,255),1)
        return frame

    # ── NORMAL MODE ──────────────────────────────────

    # Volume bar
    bx = w-55
    cv2.rectangle(frame,(bx,95),(bx+30,285),(40,40,40),-1)
    vh = int(190*vol/100)
    cv2.rectangle(frame,(bx,285-vh),(bx+30,285),
                 (0,200,255),-1)
    cv2.putText(frame,'VOL',(bx-5,300),
               cv2.FONT_HERSHEY_SIMPLEX,0.4,(200,200,200),1)
    cv2.putText(frame,f'{vol}%',(bx-5,316),
               cv2.FONT_HERSHEY_SIMPLEX,0.4,(200,200,200),1)

    # Gesture guide
    guide = [
        "✌️  PEACE     = Draw Letter ON/OFF",
        "☝️  INDEX     = Mouse Control",
        "🤏  PINCH     = Click",
        "   Hold 3s  = Click (backup)",
        "🤟  3FINGER  = Screenshot",
        "🤘  ROCK     = Back to AirOS",
        "👍  THUMB UP = Vol Up",
        "👎  THUMB DN = Vol Down",
        "🤙  PINKY    = Mute",
    ]
    cv2.rectangle(frame,(10,95),
                 (290,95+len(guide)*27+10),
                 (20,20,20),-1)
    for i,g in enumerate(guide):
        cv2.putText(frame,g,(15,112+i*27),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.44,(170,170,170),1)

    # Bottom
    cv2.rectangle(frame,(0,h-35),(w,h),(20,20,20),-1)
    cv2.putText(frame,
               '✌️ PEACE = Toggle Letter Draw Mode  |  Q = Quit',
               (10,h-12),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.45,(150,150,150),1)
    return frame

# ─── MediaPipe ────────────────────────────────────────
base_options = python.BaseOptions(
    model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.80,
    min_hand_presence_confidence=0.80,
    min_tracking_confidence=0.75
)
detector = vision.HandLandmarker.create_from_options(options)

# ─── Webcam ───────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS,          30)

screen_w, screen_h = pyautogui.size()

print("=" * 55)
print("   AirOS v5.3 Desktop Control")
print("=" * 55)
print("✌️  PEACE        = Toggle Letter Draw Mode")
print("☝️  INDEX FINGER = Mouse Control")
print("🤏  PINCH        = Click")
print("🤟  3 FINGERS    = Screenshot")
print("🤘  ROCK SIGN    = Back to AirOS")
print("👍  THUMB UP     = Volume Up")
print("👎  THUMB DOWN   = Volume Down")
print("🤙  PINKY        = Mute")
print("Q              = Quit")
print("=" * 55)

# ─── Main Loop ────────────────────────────────────────
gesture_cooldown = 0
peace_cooldown   = 0
vol_cooldown     = 0
pos_buffer       = deque(maxlen=SMOOTH_BUFFER)
draw_buffer      = deque(maxlen=DRAW_BUFFER)
hold_progress    = 0
draw_mode        = False
letter_points    = []
recognized_letter= ""
draw_mode_start  = 0
prev_draw_x      = 0
prev_draw_y      = 0
volume_level     = 50

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame,1)
    h, w, _ = frame.shape
    ct = time.time()

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    mpi = mp.Image(image_format=mp.ImageFormat.SRGB,
                   data=rgb)
    res = detector.detect(mpi)

    gesture_cooldown = max(0,gesture_cooldown-1)
    peace_cooldown   = max(0,peace_cooldown-1)
    vol_cooldown     = max(0,vol_cooldown-1)
    hold_progress    = 0

    if res.hand_landmarks:
        for lm in res.hand_landmarks:
            fingers = get_finger_states(lm)

            # Smooth index tip position
            itip   = lm[8]
            rix    = int(itip.x * w)
            riy    = int(itip.y * h)
            ix, iy = smooth_pos(rix,riy,pos_buffer)

            # Cursor
            cv2.circle(frame,(ix,iy),10,(0,255,255),-1)
            cv2.circle(frame,(ix,iy),13,(255,255,255),2)

            # ══ PEACE = Toggle Draw Mode ══════════════
            if is_peace(fingers) and peace_cooldown == 0:
                draw_mode = not draw_mode
                if draw_mode:
                    letter_points     = []
                    recognized_letter = ""
                    draw_mode_start   = ct
                    prev_draw_x       = 0
                    prev_draw_y       = 0
                    last_action       = "✍️ Draw Mode ON!"
                    draw_buffer.clear()
                else:
                    # Recognize when turning off
                    if len(letter_points) > 15:
                        recognized_letter = (
                            recognize_letter(letter_points)
                            or "?")
                        if recognized_letter != "?":
                            last_action = open_app(
                                recognized_letter)
                        else:
                            last_action = "Not recognized ❌"
                    else:
                        last_action = "Draw Mode OFF"
                    draw_buffer.clear()
                    pos_buffer.clear()
                last_action_time = ct
                peace_cooldown   = 35
                continue

            # ══ DRAW MODE ═════════════════════════════
            if draw_mode:
                # Only track index finger
                draw_hand(frame,lm,show=False)

                if is_index_only(fingers):
                    # Highly smoothed drawing position
                    dx, dy = smooth_pos(rix,riy,draw_buffer)

                    # Neon cursor
                    cv2.circle(frame,(dx,dy),
                              10,(0,255,255),-1)
                    cv2.circle(frame,(dx,dy),
                              15,(255,255,0),2)

                    if prev_draw_x != 0 and prev_draw_y != 0:
                        dist = ((dx-prev_draw_x)**2 +
                                (dy-prev_draw_y)**2)**0.5
                        # Capture point if moved enough
                        # but filter big jumps
                        if 2 < dist < 60:
                            letter_points.append((dx,dy))
                    prev_draw_x = dx
                    prev_draw_y = dy
                else:
                    prev_draw_x = 0
                    prev_draw_y = 0

                # Auto timeout
                if (ct-draw_mode_start > LETTER_TIMEOUT
                        and len(letter_points) > 15):
                    recognized_letter = (
                        recognize_letter(letter_points)
                        or "?")
                    if recognized_letter != "?":
                        last_action = open_app(
                            recognized_letter)
                    else:
                        last_action = "Not recognized ❌"
                    last_action_time  = ct
                    draw_mode         = False
                    letter_points     = []
                    draw_buffer.clear()

                continue  # Skip all other gestures

            # ══ NORMAL CONTROL MODE ═══════════════════
            draw_hand(frame,lm,show=True)

            # INDEX = Mouse
            if is_index_only(fingers):
                mouse_mode = True
                mx = int(np.interp(ix,[0,w],[0,screen_w]))
                my = int(np.interp(iy,[0,h],[0,screen_h]))
                pyautogui.moveTo(mx,my,duration=0.02)

                # Pinch = instant click
                if is_pinch(lm) and gesture_cooldown == 0:
                    pyautogui.click()
                    last_action      = "Clicked! 🖱️"
                    last_action_time = ct
                    gesture_cooldown = 20

                # Hold 3s = click backup
                if prev_mouse_x == 0:
                    prev_mouse_x = mx
                    prev_mouse_y = my
                    hold_start   = ct
                else:
                    dist = ((mx-prev_mouse_x)**2 +
                            (my-prev_mouse_y)**2)**0.5
                    if dist < 25:
                        hd = ct - hold_start
                        hold_progress = min(hd/3.0,1.0)
                        ang = int(360*hold_progress)
                        cv2.ellipse(frame,(ix,iy),
                                   (22,22),-90,0,
                                   ang,(0,255,0),3)
                        if hd >= 3.0 and gesture_cooldown == 0:
                            pyautogui.click()
                            last_action      = "Clicked! 🖱️"
                            last_action_time = ct
                            gesture_cooldown = 20
                            prev_mouse_x     = 0
                            prev_mouse_y     = 0
                    else:
                        prev_mouse_x = mx
                        prev_mouse_y = my
                        hold_start   = ct
            else:
                mouse_mode   = False
                prev_mouse_x = 0
                prev_mouse_y = 0

            # THUMB UP = Volume Up
            if is_thumb_up(fingers) and vol_cooldown == 0:
                volume_level     = min(volume_level+10,100)
                last_action      = vol_up()
                last_action_time = ct
                vol_cooldown     = 20

            # THUMB DOWN = Volume Down
            elif is_thumb_down(lm,fingers) and vol_cooldown == 0:
                volume_level     = max(volume_level-10,0)
                last_action      = vol_down()
                last_action_time = ct
                vol_cooldown     = 20

            # PINKY = Mute
            elif is_pinky_only(fingers) and gesture_cooldown == 0:
                last_action      = vol_mute()
                last_action_time = ct
                gesture_cooldown = 30

            # 3 FINGERS = Screenshot
            elif is_three_fingers(fingers) and gesture_cooldown == 0:
                last_action      = take_screenshot()
                last_action_time = ct
                gesture_cooldown = 35

            # ROCK = Back to AirOS
            elif is_rock(fingers) and gesture_cooldown == 0:
                last_action      = bring_back()
                last_action_time = ct
                gesture_cooldown = 35

    else:
        pos_buffer.clear()
        draw_buffer.clear()
        mouse_mode   = False
        prev_mouse_x = 0
        prev_mouse_y = 0
        if draw_mode:
            prev_draw_x = 0
            prev_draw_y = 0

    # Clear old action
    if last_action and ct-last_action_time > 3:
        last_action = ""

    # Draw UI
    frame = draw_ui(frame,last_action,volume_level,
                   mouse_mode,hold_progress,
                   draw_mode,recognized_letter,
                   letter_points)

    cv2.imshow('AirOS Desktop Control v5.3',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("AirOS Desktop Control v5.3 Stopped.")