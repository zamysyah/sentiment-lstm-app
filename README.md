# 😎 Sentiment Analysis with LSTM (Streamlit)

Simple and stylish web app for **sentiment analysis** built using an **LSTM neural network** (TensorFlow/Keras) and deployed with **Streamlit Cloud**. It can classify text reviews into **Positive** or **Negative** sentiments and display detailed insights including confidence levels, text statistics, and a downloadable log.

<p align="center">
  <img src="https://github.com/zamysyah/sentiment-lstm-app/blob/main/Aplikasi_Review.png" alt="App Preview" width="600">
</p>

---

## 🚀 Features

- 🧹 Clean text preprocessing (lowercasing, punctuation removal, stopwords, etc.)
- 🤖 Predicts sentiment using trained LSTM model
- 📊 Displays prediction confidence score
- 📝 Shows word count, character count, and average word length
- 📂 Logs each prediction and allows CSV download
- 🌐 Deployed on Streamlit Cloud (simple and free deployment)

---

## 🛠️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/zamysyah/sentiment-lstm-app.git
   cd sentiment-lstm-app
2. Install dependencies Make sure Python ≥3.8 is installed.
pip install -r requirements.txt
3. Run the app locally
streamlit run app.py

🧠 Model Info
Preprocessing uses NLTK and TensorFlow tokenizer

Model trained on IMDB sentiment dataset using LSTM

Stored as: sentiment_lstm_model.keras and tokenizer.pkl

🌍 Deploy to Streamlit Cloud
Push all files to a GitHub repository

Go to streamlit.io/cloud

Click "New app", connect GitHub repo, and deploy 🎉

⚠️ Important Notes:

Make sure sentiment_lstm_model.keras and tokenizer.pkl are included in your repo root

Add this line in app.py to download NLTK stopwords:
import nltk
nltk.download('stopwords')

🧑‍💻 Author
Made with ❤️ by @zamysyah

