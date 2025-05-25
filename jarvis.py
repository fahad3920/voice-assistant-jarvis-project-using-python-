import cv2
import mediapipe as mp
import pyttsx3
import speech_recognition as sr
import webbrowser
import datetime
import os
import time
import threading
import queue
import openai

openai.api_key = "YOUR_OPENAI_API_KEY"

def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()

def listen(q):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)

    try:
        query = r.recognize_google(audio)
        print(f"You said: {query}")
        q.put(query.lower())
    except sr.UnknownValueError:
        speak("Sorry, I didn't catch that.")
        q.put("")

def listen_for_wake_word(q, wake_word="jarvis"):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print(f"Say '{wake_word}' to activate Jarvis...")
        audio = r.listen(source)

    try:
        query = r.recognize_google(audio)
        print(f"You said: {query}")
        if wake_word.lower() in query.lower():
            speak("Yes, I am listening.")
            q.put("WAKE_WORD_DETECTED")
        else:
            q.put("")
    except sr.UnknownValueError:
        q.put("")

def ask_chatgpt(prompt):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        answer = response.choices[0].text.strip()
        return answer
    except Exception as e:
        return f"Sorry, I couldn't get a response from ChatGPT. {str(e)}"

def process_command(cmd):
    cmd = cmd.lower()
    if any(kw in cmd for kw in ["time", "what time", "current time"]):
        time_str = datetime.datetime.now().strftime("%I:%M %p")
        speak(f"The time is {time_str}")
    elif any(kw in cmd for kw in ["youtube", "open youtube"]):
        speak("Opening YouTube")
        webbrowser.open("https://youtube.com")
    elif any(kw in cmd for kw in ["notepad", "open notepad"]):
        speak("Opening Notepad")
        os.system("notepad.exe")
    elif any(kw in cmd for kw in ["exit", "sleep", "quit", "stop"]):
        speak("Goodbye!")
        exit()
    else:
        # Use ChatGPT for other queries
        speak("Let me think about that...")
        answer = ask_chatgpt(cmd)
        speak(answer)

def detect_hand_gesture(frame, hands):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    gesture = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Simple gesture detection example: check if thumb is up
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]

            if thumb_tip.y < thumb_ip.y:
                gesture = "Thumbs Up"
            else:
                gesture = "Unknown"

    return gesture, frame

def wake_word_listener(q):
    while True:
        listen_for_wake_word(q)

def command_listener(q):
    while True:
        cmd = q.get()
        if cmd == "WAKE_WORD_DETECTED":
            listen(q)
        elif cmd:
            process_command(cmd)

if __name__ == "__main__":
    speak("Initializing Jarvis")
    cap = cv2.VideoCapture(0)
    mp_face = mp.solutions.face_detection
    face_detector = mp_face.FaceDetection()
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    speak("Hello, I am Jarvis. How can I help you?")

    q = queue.Queue()

    wake_thread = threading.Thread(target=wake_word_listener, args=(q,), daemon=True)
    command_thread = threading.Thread(target=command_listener, args=(q,), daemon=True)

    wake_thread.start()
    command_thread.start()

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame from webcam. Exiting...")
            break

        # Resize frame to smaller size to reduce hanging
        frame = cv2.resize(frame, (480, 360))

        # Face detection
        face_results = face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if face_results.detections:
            cv2.putText(frame, "Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Hand gesture detection
        gesture, frame = detect_hand_gesture(frame, hands)
        if gesture:
            cv2.putText(frame, f"Gesture: {gesture}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            if gesture == "Thumbs Up":
                speak("Thumbs up detected!")

        cv2.imshow("Jarvis Webcam Feed", frame)

        if cv2.getWindowProperty("Jarvis Webcam Feed", cv2.WND_PROP_VISIBLE) < 1:
            print("Webcam window closed by user. Exiting...")
            # Instead of breaking immediately, wait and try to reopen the window
            cv2.destroyAllWindows()
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(0)
            continue

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            speak("Goodbye!")
            break

        # Add small delay to reduce CPU usage and prevent hanging
        time.sleep(0.03)

    cap.release()
    cv2.destroyAllWindows()

    cap.release()
    cv2.destroyAllWindows()
