import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import google.generativeai as genai
import base64
from io import BytesIO
from PIL import Image
import warnings
import time
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

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

# Sentence generation function (unchanged)
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
        return f"Error: {e}"

# Translation function (unchanged)
def translate_sentence(sentence, target_language):
    if not sentence or "Error" in sentence or "No English" in sentence:
        return f"No {target_language} translation available."
    prompt = f"Translate the following English sentence into {target_language}: '{sentence}'"
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip() if response.text else f"No {target_language} translation generated."
    except:
        time.sleep(5)
        try:
            response = gemini_model.generate_content(prompt)
            return response.text.strip() if response.text else f"No {target_language} translation generated."
        except Exception as e:
            return f"Error in {target_language}: {e}"

# JavaScript for webcam capture
js_code = """
<video id="video" width="640" height="480" autoplay style="display:block;"></video>
<canvas id="canvas" width="640" height="480" style="display:none;"></canvas>
<script>
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d');
    let stream = null;

    async function startWebcam() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            sendFrames();
        } catch (err) {
            console.error("Webcam error:", err);
            window.parent.postMessage({ error: err.message }, '*');
        }
    }

    function sendFrames() {
        if (!stream) return;
        context.drawImage(video, 0, 0, 640, 480);
        const data = canvas.toDataURL('image/jpeg', 0.8);
        window.parent.postMessage({ image: data }, '*');
        setTimeout(sendFrames, 100); // Adjust for ~10 FPS
    }

    startWebcam();

    // Cleanup on page unload
    window.addEventListener('unload', () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
    });
</script>
"""

# Embed JavaScript in Streamlit
frame_placeholder = st.empty()
html_component = st.components.v1.html(js_code, height=500)

# Initialize session state
if 'frame' not in st.session_state:
    st.session_state.frame = None
if 'gesture_sequence' not in st.session_state:
    st.session_state.gesture_sequence = []
if 'last_gesture' not in st.session_state:
    st.session_state.last_gesture = None
if 'sentence_cache' not in st.session_state:
    st.session_state.sentence_cache = {'English': '', 'Telugu': '', 'Hindi': ''}

# Streamlit form to receive frames (workaround for postMessage)
frame_form = st.form(key="frame_form")
frame_input = frame_form.text_input("Hidden frame input", value="", key="frame_input", label_visibility="hidden")
form_submitted = frame_form.form_submit_button("Process", disabled=True)

# Main loop to process frames
while True:
    # Simulate receiving frames (in practice, JavaScript updates frame_input via st.experimental_set_query_params or similar)
    # Since Streamlit doesn't natively support postMessage, we rely on manual refresh or session state
    if frame_input and frame_input.startswith("data:image/jpeg;base64,"):
        try:
            # Decode base64 frame
            base64_string = frame_input.split(',')[1]
            img_data = base64.b64decode(base64_string)
            img = Image.open(BytesIO(img_data))
            frame = np.array(img)
            frame_rgb = frame  # Already RGB from PIL
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)  # For display

            # Process with MediaPipe
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                landmarks = []
                for hand_landmarks in results.multi_hand_landmarks[:2]:
                    mp_drawing.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
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
                    if confidence > 0.8 and gesture != st.session_state.last_gesture:
                        st.session_state.gesture_sequence.append(gesture)
                        st.session_state.last_gesture = gesture

                    # Generate and translate sentences
                    if len(st.session_state.gesture_sequence) >= 2:
                        english_sentence = generate_sentence_from_signs(st.session_state.gesture_sequence)
                        sentences = {
                            'English': english_sentence,
                            'Telugu': translate_sentence(english_sentence, 'Telugu'),
                            'Hindi': translate_sentence(english_sentence, 'Hindi')
                        }
                        st.session_state.sentence_cache = sentences
                        st.session_state.gesture_sequence = []
                        st.session_state.last_gesture = None

                    # Update sidebar
                    gesture_placeholder.write(f"**Current Gesture**: {gesture} ({confidence:.2f})")
                    sequence_placeholder.write(f"**Sequence**: {', '.join(st.session_state.gesture_sequence)}")
                    sentence_text = "\n".join([f"**{lang}**: {sentence}" for lang, sentence in st.session_state.sentence_cache.items()])
                    sentence_placeholder.markdown(sentence_text)

                except Exception as e:
                    st.error(f"Prediction error: {e}")

            # Display frame
            frame_placeholder.image(frame_bgr, channels="BGR")

        except Exception as e:
            st.error(f"Frame processing error: {e}")

    # Break loop if Streamlit session stops
    if not st.session_state.get('run', True):
        break

    # Small delay to avoid overloading
    time.sleep(0.1)

# Cleanup (not typically reached in Streamlit)
hands.close()
