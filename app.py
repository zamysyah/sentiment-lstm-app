import streamlit as st
import tensorflow as tf
import pickle
import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Unduh stopwords jika belum tersedia
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

# Load model dan tokenizer
model = tf.keras.models.load_model("sentiment_lstm_model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 200
stemmer = SnowballStemmer("english")

def clean_text(text):
    text = text.lower()
    text = re.sub("<.*?>", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = re.sub("\s+", " ", text).strip()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("🧠 Sentiment Analysis with LSTM")

if "log_df" not in st.session_state:
    st.session_state.log_df = pd.DataFrame(columns=["Original Text", "Prediction", "Confidence"])

user_input = st.text_area("✍️ Your Review", "")

if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter a review first.")
    else:
        cleaned = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
        prob = model.predict(padded)[0][0]
        pred = prob > 0.5
        label = "😊 Positive" if pred else "😠 Negative"
        confidence = f"{prob*100:.2f}%" if pred else f"{(1 - prob)*100:.2f}%"

        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {confidence}")

        words = user_input.split()
        word_count = len(words)
        char_count = len(user_input)
        avg_word_len = char_count / word_count if word_count else 0

        st.subheader("📊 Text Statistics")
        st.markdown(f"""
        - **Word count:** {word_count}
        - **Character count:** {char_count}
        - **Avg. word length:** {avg_word_len:.2f}
        """)

        new_row = {
            "Original Text": user_input,
            "Prediction": label,
            "Confidence": confidence
        }
        st.session_state.log_df = pd.concat(
            [st.session_state.log_df, pd.DataFrame([new_row])],
            ignore_index=True
        )

if not st.session_state.log_df.empty:
    st.subheader("📝 Prediction Log")
    st.dataframe(st.session_state.log_df[::-1], use_container_width=True)

    csv = st.session_state.log_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Log as CSV",
        data=csv,
        file_name="sentiment_prediction_log.csv",
        mime="text/csv"
    )

    if st.button("🗑️ Clear Log"):
        st.session_state.log_df = pd.DataFrame(columns=["Original Text", "Prediction", "Confidence"])
        st.success("Log has been cleared.")
