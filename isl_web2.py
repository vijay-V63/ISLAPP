import streamlit as st
import mediapipe as mp
import numpy as np
import tensorflow as tf
import google.generativeai as genai
from time import sleep
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import av

# Streamlit setup
st.title("Indian Sign Language Detection")
st.sidebar.header("Recognized Gestures")
gesture_placeholder = st.sidebar.empty()
sequence_placeholder = st.sidebar.empty()
sentence_placeholder = st.sidebar.empty()

# Prompt for Gemini API key
gemini_api_key = st.text_input("Enter your Gemini API key:", type="password")
if not gemini_api_key:
    st.error("Please enter a Gemini API key.")
    st.stop()

# Configure Gemini
try:
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')
    st.success("Gemini model initialized")
except Exception as e:
    st.error(f"Error initializing Gemini: {e}")
    st.stop()

# Load ISL model
try:
    model = tf.keras.models.load_model('isl_model.h5')
    st.success("ISL model loaded successfully")
except Exception as e:
    st.error(f"Error loading ISL model: {e}")
    st.stop()

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

    Input: ['HELLO', 'MY', 'NAME', 'IS', 'VIJAY']
    Output: Hello, my name is Vijay.

    Input: ['HELLO', 'THANKYOU']
    Output: Hello, thank you.

    Now convert this:
    Input: {sign_list}
    Output:"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip() if response.text else "No English sentence generated."
    except Exception as e:
        return f"Error: {e}"

# Translation function
def translate_sentence(sentence, target_language):
    if not sentence or "Error" in sentence or "No English" in sentence:
        return f"No {target_language} translation available."
    prompt = f"Translate the following English sentence into {target_language}: '{sentence}'"
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip() if response.text else f"No {target_language} translation generated."
    except:
        sleep(5)
        try:
            response = gemini_model.generate_content(prompt)
            return response.text.strip() if response.text else f"No {target_language} translation generated."
        except Exception as e:
            return f"Error in {target_language}: {e}"

# RTC Configuration
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# VideoProcessor class for streamlit-webrtc
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.gesture_sequence = []
        self.last_gesture = None
        self.sentence_cache = {'English': '', 'Telugu': '', 'Hindi': ''}

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks[:2]:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
            if len(results.multi_hand_landmarks) == 1:
                landmarks.extend([0] * 63)

            try:
                landmarks_np = np.array([landmarks], dtype=np.float32)
                pred = model.predict(landmarks_np, verbose=0)
                gesture = classes[np.argmax(pred)]
                confidence = np.max(pred)

                if confidence > 0.8 and gesture != self.last_gesture:
                    self.gesture_sequence.append(gesture)
                    self.last_gesture = gesture

                if len(self.gesture_sequence) >= 2:
                    english_sentence = generate_sentence_from_signs(self.gesture_sequence)
                    sentences = {
                        'English': english_sentence,
                        'Telugu': translate_sentence(english_sentence, 'Telugu'),
                        'Hindi': translate_sentence(english_sentence, 'Hindi')
                    }
                    self.sentence_cache = sentences
                    self.gesture_sequence = []
                    self.last_gesture = None

                gesture_placeholder.write(f"**Current Gesture**: {gesture} ({confidence:.2f})")
                sequence_placeholder.write(f"**Sequence**: {', '.join(self.gesture_sequence)}")
                sentence_text = "\n".join([f"**{lang}**: {sentence}" for lang, sentence in self.sentence_cache.items()])
                sentence_placeholder.markdown(sentence_text)

            except Exception as e:
                st.error(f"Prediction error: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start the webcam stream
webrtc_streamer(
    key="isl-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)
