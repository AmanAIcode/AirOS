import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import time
from collections import deque
from groq import Groq
import pyttsx3
import threading
import queue
import speech_recognition as sr

from config import GROQ_API_KEY

model_path = "models/hand_landmarker.task"

# ─── Groq Client ──────────────────────────────────────
client = Groq(api_key=GROQ_API_KEY)

# ─── TTS Queue-Based Worker (fresh engine each time) ──
speech_queue = queue.Queue()

def tts_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 170)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine
        except Exception as e:
            print(f"TTS Error: {e}")

threading.Thread(target=tts_worker, daemon=True).start()

def speak(text):
    speech_queue.put(text)

# ─── Speech Recognition ───────────────────────────────
recognizer = sr.Recognizer()

def listen_microphone():
    global typed_text, mic_status
    mic_status = "Listening... 🎤"
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
        mic_status = "Processing... ⏳"
        text = recognizer.recognize_google(audio)
        typed_text += text
        mic_status = ""
    except sr.WaitTimeoutError:
        mic_status = "No speech detected ❌"
    except sr.UnknownValueError:
        mic_status = "Could not understand ❌"
    except Exception as e:
        mic_status = f"Error: {str(e)[:30]}"

# ─── Keyboard Layout ──────────────────────────────────
KEYS = [
    ['Q','W','E','R','T','Y','U','I','O','P'],
    ['A','S','D','F','G','H','J','K','L','⌫'],
    ['Z','X','C','V','B','N','M','🎤 SPEAK'],
    ['SPACE','ASK AI','CLEAR']
]

KEY_WIDTH  = 65
KEY_HEIGHT = 60
KEY_MARGIN = 8
START_X    = 30
START_Y    = 280

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# ─── State ────────────────────────────────────────────
typed_text     = ""
ai_response    = ""
ai_thinking    = False
mic_status     = ""
hovered_key    = None
dwell_start    = {}
DWELL_TIME     = 1.0
last_press_time= 0
COOLDOWN       = 0.5

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

# ─── Keyboard Layout Helper ───────────────────────────
def get_key_rect(row_idx, col_idx, key):
    if key == 'SPACE':
        x = START_X
        w = KEY_WIDTH * 4 + KEY_MARGIN * 3
    elif key == 'ASK AI':
        x = START_X + KEY_WIDTH * 4 + KEY_MARGIN * 4
        w = KEY_WIDTH * 3 + KEY_MARGIN * 2
    elif key == 'CLEAR':
        x = START_X + KEY_WIDTH * 7 + KEY_MARGIN * 7
        w = KEY_WIDTH * 2 + KEY_MARGIN
    elif key == '🎤 SPEAK':
        x = START_X + col_idx * (KEY_WIDTH + KEY_MARGIN)
        w = KEY_WIDTH * 2 + KEY_MARGIN
    else:
        x = START_X + col_idx * (KEY_WIDTH + KEY_MARGIN)
        w = KEY_WIDTH
    y = START_Y + row_idx * (KEY_HEIGHT + KEY_MARGIN)
    return x, y, w, KEY_HEIGHT

# ─── AI Call ──────────────────────────────────────────
def ask_ai(question):
    global ai_response, ai_thinking
    ai_thinking = True
    ai_response = ""
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system",
                 "content": "You are AirOS AI Assistant. Give short, clear, helpful answers in 2-3 sentences max."},
                {"role": "user", "content": question}
            ],
            max_tokens=150
        )
        ai_response = completion.choices[0].message.content
        speak(ai_response)
    except Exception as e:
        ai_response = f"Error: {str(e)}"
    ai_thinking = False

# ─── UI Drawing ────────────────────────────────────────
def draw_ui(frame, typed_text, ai_response, ai_thinking,
            hovered_key, dwell_start, mic_status):
    h, w, _ = frame.shape

    cv2.rectangle(frame,(0,0),(w,50),(20,20,20),-1)
    cv2.putText(frame,'AirOS v7.2 - AI Assistant',
               (10,32),cv2.FONT_HERSHEY_SIMPLEX,
               0.8,(0,255,255),2)

    if mic_status:
        cv2.putText(frame, mic_status,
                   (w-320,32),cv2.FONT_HERSHEY_SIMPLEX,
                   0.6,(0,255,255),2)

    # Typed text box
    cv2.rectangle(frame,(20,60),(w-20,110),(30,30,30),-1)
    cv2.rectangle(frame,(20,60),(w-20,110),(100,100,100),1)
    display_text = typed_text[-60:] if len(typed_text)>60 else typed_text
    cv2.putText(frame,'You: ' + display_text + '|',
               (30,95),cv2.FONT_HERSHEY_SIMPLEX,
               0.7,(0,255,0),2)

    # AI Response box
    box_color = (40,40,80) if ai_thinking else (30,50,30)
    cv2.rectangle(frame,(20,120),(w-20,210),box_color,-1)
    cv2.rectangle(frame,(20,120),(w-20,210),(100,100,100),1)

    if ai_thinking:
        cv2.putText(frame,'AI: Thinking... 🤔',
                   (30,165),cv2.FONT_HERSHEY_SIMPLEX,
                   0.8,(255,200,0),2)
    elif ai_response:
        words = ai_response.split(' ')
        lines = []
        line = "AI: "
        for word in words:
            if len(line + word) > 70:
                lines.append(line)
                line = ""
            line += word + " "
        lines.append(line)
        for i, l in enumerate(lines[:3]):
            cv2.putText(frame, l,
                       (30, 150 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.6,(0,255,150),2)
    else:
        cv2.putText(frame,'AI: Type or speak your question, then ASK AI',
                   (30,165),cv2.FONT_HERSHEY_SIMPLEX,
                   0.6,(150,150,150),2)

    # Keyboard
    kb_h = len(KEYS) * (KEY_HEIGHT + KEY_MARGIN) + 20
    overlay = frame.copy()
    cv2.rectangle(overlay,(START_X-10,START_Y-10),
                 (w-20,START_Y+kb_h),(15,15,15),-1)
    cv2.addWeighted(overlay,0.75,frame,0.25,0,frame)

    for row_idx, row in enumerate(KEYS):
        for col_idx, key in enumerate(row):
            x, y, kw, kh = get_key_rect(row_idx, col_idx, key)

            if key == 'ASK AI':
                bg_color = (0,100,200)
            elif key == '🎤 SPEAK':
                bg_color = (150,0,150)
            elif key == hovered_key:
                if key in dwell_start:
                    elapsed = time.time() - dwell_start[key]
                    progress = min(elapsed/DWELL_TIME,1.0)
                    g = int(100+155*progress)
                    bg_color = (0,g,0)
                else:
                    bg_color = (0,150,0)
            else:
                bg_color = (50,50,50)

            cv2.rectangle(frame,(x,y),(x+kw,y+kh),
                         bg_color,-1)
            cv2.rectangle(frame,(x,y),(x+kw,y+kh),
                         (100,100,100),1)

            font_scale = 0.42 if len(key)>2 else 0.7
            text_size = cv2.getTextSize(
                key,cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,2)[0]
            tx = x + (kw-text_size[0])//2
            ty = y + (kh+text_size[1])//2
            cv2.putText(frame,key,(tx,ty),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale,(255,255,255),2)

    cv2.rectangle(frame,(0,h-30),(w,h),(20,20,20),-1)
    cv2.putText(frame,
               'Hover to type | PINCH = instant | SPEAK = voice input | ESC = Quit',
               (10,h-10),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.45,(150,150,150),1)
    return frame

# ─── Process Keypress ──────────────────────────────────
def process_key(key, typed_text):
    global ai_response, ai_thinking
    if key == '⌫':
        typed_text = typed_text[:-1]
    elif key == 'SPACE':
        typed_text += ' '
    elif key == 'CLEAR':
        typed_text = ''
        ai_response = ''
    elif key == '🎤 SPEAK':
        threading.Thread(target=listen_microphone,
                        daemon=True).start()
    elif key == 'ASK AI':
        if typed_text.strip():
            threading.Thread(target=ask_ai,
                            args=(typed_text,),
                            daemon=True).start()
    else:
        typed_text += key
    return typed_text

# ─── MediaPipe Setup ──────────────────────────────────
base_options = python.BaseOptions(
    model_asset_path=model_path)
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
print("   AirOS v7.2 AI Assistant Started!")
print("=" * 55)
print("Hover keys to type, or use 🎤 SPEAK for voice")
print("Select 'ASK AI' to get AI response (spoken too)")
print("ESC = Quit")
print("=" * 55)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mpi = mp.Image(image_format=mp.ImageFormat.SRGB,
                   data=rgb)
    res = detector.detect(mpi)

    hovered_key  = None
    current_time = time.time()

    if res.hand_landmarks:
        for lm in res.hand_landmarks:
            draw_hand(frame, lm)

            itip = lm[8]
            rix  = int(itip.x * w)
            riy  = int(itip.y * h)
            ix, iy = smooth_pos(rix, riy, pos_buffer)

            cv2.circle(frame,(ix,iy),10,(0,255,255),-1)
            cv2.circle(frame,(ix,iy),12,(255,255,255),2)

            for row_idx, row in enumerate(KEYS):
                for col_idx, key in enumerate(row):
                    x, y, kw, kh = get_key_rect(
                        row_idx, col_idx, key)
                    if x < ix < x+kw and y < iy < y+kh:
                        hovered_key = key
                        break

            if hovered_key:
                if hovered_key not in dwell_start:
                    dwell_start[hovered_key] = current_time
                else:
                    elapsed = current_time - dwell_start[hovered_key]
                    if elapsed >= DWELL_TIME:
                        if current_time - last_press_time > COOLDOWN:
                            typed_text = process_key(
                                hovered_key, typed_text)
                            last_press_time = current_time
                            dwell_start = {}
            else:
                dwell_start = {}

            if is_pinch(lm) and hovered_key:
                if current_time - last_press_time > COOLDOWN:
                    typed_text = process_key(
                        hovered_key, typed_text)
                    last_press_time = current_time
                    dwell_start = {}

    else:
        pos_buffer.clear()
        dwell_start = {}

    frame = draw_ui(frame, typed_text, ai_response,
                   ai_thinking, hovered_key, dwell_start,
                   mic_status)

    cv2.imshow('AirOS AI Assistant', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("AI Assistant Stopped.")