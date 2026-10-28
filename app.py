# pip install -r requirements.txt
# streamlit run app.py
import os
import time

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

st.set_page_config(page_title="AI Health Symptoms Assistant", page_icon="🩺", layout="wide")
load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "")

if not API_KEY:
    st.error("GROQ_API_KEY is missing. Add it to the `.env` file in this folder.")
    st.stop()

RED_FLAGS = [
    "chest pain", "chest tightness", "difficulty breathing", "can't breathe",
    "cannot breathe", "shortness of breath", "severe bleeding", "unconscious",
    "fainted", "seizure", "stroke", "face drooping", "slurred speech",
    "sudden weakness", "coughing blood", "vomiting blood", "suicidal",
    "want to die", "overdose", "poison", "severe allergic", "swelling of throat",
    "worst headache", "stiff neck",
]

EMERGENCY_MESSAGE = (
    "🚨 **Your message mentions possible emergency symptoms.**\n\n"
    "Please call **112 / 108** or go to the nearest emergency room right now. "
    "Do not wait for an online answer.\n\n"
    "If you are having thoughts of self-harm, contact a local helpline or a trusted person immediately."
)

SYSTEM_PROMPT = (
    "You are an AI Health Symptoms Assistant for general education only. "
    "Never give a definite diagnosis, prescription, or medicine dosage. "
    "Use cautious wording like 'may be related to'. "
    "Reply in exactly this format:\n"
    "**Possible common causes:**\n"
    "**Self-care tips:**\n"
    "**See a doctor if:**\n"
    "**Emergency signs:**\n"
    "Keep it short, simple, and calm. End with: 'Please consult a doctor for proper advice.'"
)


def has_red_flag(text):
    text = text.lower()
    return any(flag in text for flag in RED_FLAGS)


@st.cache_resource(show_spinner=False)
def get_client():
    return Groq(api_key=API_KEY)


client = get_client()

PREFERRED = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-20b",
             "openai/gpt-oss-120b", "llama3-70b-8192", "llama3-8b-8192"]
SKIP_WORDS = ["whisper", "tts", "guard", "embed", "vision", "playai", "orpheus"]


@st.cache_resource(show_spinner=False)
def pick_model():
    """Use MODEL_NAME from .env if set, else choose a model your key can access."""
    if MODEL_NAME:
        return MODEL_NAME
    try:
        ids = [m.id for m in client.models.list().data]
    except Exception:
        return PREFERRED[0]
    for name in PREFERRED:
        if name in ids:
            return name
    chat = [i for i in ids if not any(w in i.lower() for w in SKIP_WORDS)]
    return chat[0] if chat else PREFERRED[0]


ACTIVE_MODEL = pick_model()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def generate_answer(report, attempts=3):
    for attempt in range(1, attempts + 1):
        try:
            response = client.chat.completions.create(
                model=ACTIVE_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": report},
                ],
                max_tokens=600,
                temperature=0.3,
            )
            answer = response.choices[0].message.content
            if not answer:
                raise ValueError("The model returned an empty response.")
            return answer.strip()
        except Exception as error:
            if attempt < attempts:
                time.sleep(2)
            else:
                return f"The AI service is temporarily unavailable. Please try again.\n\nDetails: {error}"


def ask_question(question, age, gender, duration, severity):
    question = (question or "").strip()
    if not question:
        return

    st.session_state.chat_history.append({"role": "user", "content": question})

    if has_red_flag(question):
        st.session_state.chat_history.append({"role": "assistant", "content": EMERGENCY_MESSAGE})
        return

    report = (
        f"Age: {age}\nGender: {gender}\nDuration: {duration}\n"
        f"Severity (1-10): {severity}\nSymptoms: {question}"
    )
    answer = generate_answer(report)
    if severity >= 9:
        answer += "\n\n⚠️ You rated your symptoms very severe. Please consider seeing a doctor today."
    st.session_state.chat_history.append({"role": "assistant", "content": answer})


st.title("🩺 AI Health Symptoms Assistant")
st.caption("Type your symptoms and get general health information.")

left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("Patient details")
    age = st.number_input("Age", min_value=0, max_value=120, value=25)
    gender = st.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"])
    duration = st.selectbox("Symptom duration", ["Less than 1 day", "1-3 days", "4-7 days", "1-2 weeks", "More than 2 weeks"])
    severity = st.slider("Severity (1 = mild, 10 = severe)", 1, 10, 4)
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

with right_col:
    chat_container = st.container(height=520, border=True)
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    question = st.chat_input("Type your symptoms, e.g. 'fever and sore throat for 2 days'")
    if question:
        with st.spinner("Thinking..."):
            ask_question(question, age, gender, duration, severity)
        st.rerun()
