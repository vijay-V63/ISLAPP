import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load model and label encoder
model = tf.keras.models.load_model('isl_model.h5')
classes = np.load('label_encoder.npy', allow_pickle=True)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Process frame
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

        # Predict gesture
        landmarks = np.array([landmarks])
        pred = model.predict(landmarks, verbose=0)
        gesture = classes[np.argmax(pred)]
        confidence = np.max(pred)
        cv2.putText(frame, f"{gesture} ({confidence:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ISL Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()