import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import google.generativeai as genai
import warnings
from time import sleep
import time
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Prompt for Gemini API key
gemini_api_key = input("Enter your Gemini API key: ")
if not gemini_api_key:
    print("Error: Gemini API key is required")
    exit()

# Configure Gemini
try:
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')
    print("Gemini model initialized")
except Exception as e:
    print(f"Error initializing Gemini: {e}")
    exit()

# Load ISL model
try:
    model = tf.keras.models.load_model('isl_model.h5')
    print("ISL model loaded successfully")
except Exception as e:
    print(f"Error loading ISL model: {e}")
    exit()

classes = np.load('label_encoder.npy', allow_pickle=True)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Sentence generation function
def generate_sentence_from_signs(sign_list):
    prompt = f"""
    You are a helpful assistant that converts a list of recognized Indian Sign Language (ISL) signs into grammatically correct and meaningful English sentences.

    Example:
    Input: ['I', 'LOVE', 'YOU']
    Output: I love you.

    Input: ['HELLO', 'MY', 'NAME', 'IS', 'RAVI']
    Output: Hello, my name is Ravi.

    Input: ['HELLO', 'THANKYOU']
    Output: Hello, thank you.

    Now convert this:
    Input: {sign_list}
    Output:"""

    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip() if response.text else "No English sentence generated."
    except Exception as e:
        print(f"Gemini error (English): {e}")
        return f"Error: {e}"

# Translation function
def translate_sentence(sentence, target_language):
    if not sentence or "Error" in sentence or "No English" in sentence:
        return f"No {target_language} translation available."
    prompt = f"Translate the following English sentence into {target_language}: '{sentence}'"
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip() if response.text else f"No {target_language} translation generated."
    except Exception as e:
        print(f"Gemini error ({target_language}): {e}")
        sleep(5)
        try:
            response = gemini_model.generate_content(prompt)
            return response.text.strip() if response.text else f"No {target_language} translation generated."
        except:
            return f"Error in {target_language}: {e}"

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Changed to MP4
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

# Store gestures
gesture_sequence = []
last_gesture = None
sentence_cache = {'English': '', 'Telugu': '', 'Hindi': ''}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks[:2]:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        if len(results.multi_hand_landmarks) == 1:
            landmarks.extend([0] * 63)

        try:
            landmarks = np.array([landmarks], dtype=np.float32)
            pred = model.predict(landmarks, verbose=0)
            gesture = classes[np.argmax(pred)]
            confidence = np.max(pred)

            # Store gesture
            if confidence > 0.8 and gesture != last_gesture:
                gesture_sequence.append(gesture)
                last_gesture = gesture
                print(f"Added gesture: {gesture} (Confidence: {confidence:.2f})")

            # Generate and translate sentences
            if len(gesture_sequence) >= 2:
                # Generate English sentence
                english_sentence = generate_sentence_from_signs(gesture_sequence)
                # Translate
                sentences = {
                    'English': english_sentence,
                    'Telugu': translate_sentence(english_sentence, 'Telugu'),
                    'Hindi': translate_sentence(english_sentence, 'Hindi')
                }
                
                # Log to file
                with open('gesture_log.txt', 'a', encoding='utf-8') as f:
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Sequence: {', '.join(gesture_sequence)}\n")
                    for lang, sentence in sentences.items():
                        f.write(f"{lang}: {sentence}\n")
                    f.write("\n")
                
                sentence_cache = sentences
                gesture_sequence = []  # Reset
                last_gesture = None

            # Clean UI: Draw semi-transparent box
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10, 630, 150), (0, 0, 0), -1)  # Black box
            alpha = 0.6  # Transparency
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Display text
            cv2.putText(frame, f"Gesture: {gesture} ({confidence:.2f})", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Sequence: {', '.join(gesture_sequence)}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Display sentences in a compact layout
            y_offset = 100
            for lang, sentence in sentence_cache.items():
                # Wrap text if too long
                if len(sentence) > 40:
                    words = sentence.split()
                    line = ""
                    lines = []
                    for word in words:
                        if len(line + word) < 40:
                            line += word + " "
                        else:
                            lines.append(line.strip())
                            line = word + " "
                    lines.append(line.strip())
                    for i, line in enumerate(lines):
                        cv2.putText(frame, f"{lang}: {line}", (20, y_offset + i*20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset += len(lines) * 20
                else:
                    cv2.putText(frame, f"{lang}: {sentence}", (20, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset += 25

        except Exception as e:
            print(f"Prediction error: {e}")

    out.write(frame)
    cv2.imshow("ISL Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
hands.close()