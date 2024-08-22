import os
import streamlit as st
import whisper
from gtts import gTTS
from groq import Groq

# Load the Whisper model once
model = whisper.load_model("base")

from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

# Function to process audio input and get response
def chatbot(audio_path):
    if not audio_path:
        st.error("No audio input detected. Please record your voice.")
        return None, None

    # Transcribe the audio input using Whisper
    transcription = model.transcribe(audio_path)
    user_input = transcription.get("text", "")

    if not user_input:
        st.error("Could not understand input.")
        return None, None

    # Generate a response using Llama 8B via Groq API
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": user_input}],
            model="llama3-8b-8192",
        )
        response_text = chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None, None

    # Convert the response text to speech using gTTS
    tts = gTTS(text=response_text, lang='en')
    tts.save("response.mp3")

    return response_text, "response.mp3"

# Streamlit App Layout
st.title("VoiceToVoice Chatbot")

st.markdown("""
    <style>
        .stAudio {
            width: 100%;
            height: auto;
        }
        .stButton {
            background-color: #000;
        }
        body {
            background-color: #f0f2f6;
        }
        .main {
            background-color: #ffffff;
        }
    </style>
    """, unsafe_allow_html=True)

# Upload Audio or Record Voice
audio_input = st.file_uploader("Record Your Voice", type=["wav", "mp3", "ogg"])

# Process Input and Generate Response
if audio_input:
    response_text, response_audio = chatbot(audio_input)
    if response_text and response_audio:
        st.write("Chatbot Response:", response_text)
        st.audio(response_audio)

# Clear Button
if st.button("Clear"):
    st.experimental_rerun()
