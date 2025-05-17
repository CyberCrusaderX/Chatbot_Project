import streamlit as st
import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Download NLTK stopwords (first time only)
nltk.download('stopwords')

# -------------------------------
# 🧠 Load Model and Vectorizer
# -------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("chatbot_intel_rf_model.pkl")
        vectorizer = joblib.load("chatbot_tfidf_vectorizer.pkl")
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {e}")
        return None, None

# -------------------------------
# 📦 Load Dataset for Response Mapping
# -------------------------------
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("bitext-media-llm-chatbot-training-dataset.csv")
        df['cleaned_input'] = df['instruction'].apply(clean_text)
        df.drop_duplicates(subset=['instruction'], keep='first', inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

# -------------------------------
# 🧹 Text Cleaning Function
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# -------------------------------
# 🤖 Chatbot Logic
# -------------------------------
def get_chatbot_response(user_query, model, vectorizer, df):
    cleaned = clean_text(user_query)
    tfidf_input = vectorizer.transform([cleaned])
    predicted_intent = model.predict(tfidf_input)[0]
    try:
        response = df[df['intent'] == predicted_intent].iloc[0]['response']
    except IndexError:
        response = "I'm sorry, I couldn't find a matching intent."
    return predicted_intent, response

# -------------------------------
# 🎨 Streamlit UI
# -------------------------------
def main():
    st.set_page_config(page_title="🤖 Intelligent Customer Support Chatbot", layout="centered")

    st.title("🤖 Intelligent Customer Support Chatbot")
    st.markdown("Enter your query below and the chatbot will classify the intent and provide a response.")

    # Load resources
    model, vectorizer = load_model()
    df = load_dataset()

    if model is None or df.empty:
        st.stop()

    # Chat interface
    user_input = st.text_input("You:", placeholder="Ask something like 'I forgot my password'")

    if user_input:
        with st.spinner("Processing..."):
            intent, response = get_chatbot_response(user_input, model, vectorizer, df)
            st.write(f"🧠 Predicted Intent: `{intent}`")
            st.success(f"💬 Response: {response}")

    st.markdown("---")
    st.markdown("📌 *Try queries like:*")
    st.markdown("- How do I reset my password?")
    st.markdown("- The app keeps crashing.")
    st.markdown("- Where is my invoice?")

if __name__ == "__main__":
    main()
