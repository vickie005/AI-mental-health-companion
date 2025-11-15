from flask import Flask, render_template, request, jsonify
import joblib
import re
import string

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



def ai_chat_response(user_text):
    prompt = f"Respond with empathy, comfort and supportive emotional guidance to:\n\n{user_text}"
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return completion.choices[0].message.content


# Load model and vectorizer
model = joblib.load('../sentiment_model.pkl')
vectorizer = joblib.load('../vectorizer.pkl')

app = Flask(__name__)

# --- Text cleaning---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text.strip()

# --- Emoji + default message ---
def emotion_response(sentiment):
    if sentiment == "positive":
        return "I'm really happy to hear that! Keep shining 💛", "🙂"
    elif sentiment == "negative":
        return "I'm sorry you're going through this. You're not alone 💙", "😞"
    else:
        return "I understand. Feel free to share more.", "😐"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['user_input']
    cleaned = clean_text(text)

    # Sentiment prediction
    vectorized = vectorizer.transform([cleaned])
    sentiment = model.predict(vectorized)[0]

    # Default emotion-based message
    base_response, emoji = emotion_response(sentiment)

    # Use GPT only for negative or neutral
    if sentiment in ["negative", "neutral"]:
        ai_response = ai_chat_response(text)
    else:
        ai_response = base_response

    return jsonify({
        'sentiment': sentiment,
        'response': ai_response,
        'emoji': emoji
    })

if __name__ == '__main__':
    app.run(debug=True)

def mental_health_resources(sentiment):
    if sentiment == "negative":
        return [
            "📞 Kenya Mental Health Hotline: 1199 (Free & Confidential)",
            "💬 Befrienders Kenya: https://befrienderskenya.org/",
            "🧠 WHO Mental Health Resources: https://www.who.int/mental_health/en/",
            "📱 Check-in with someone you trust or a counselor.",
        ]
    elif sentiment == "neutral":
        return [
            "🧘 Try journaling or mindfulness for clarity.",
            "📘 Learn about emotional wellness: https://www.mentalhealth.gov/",
            "💬 Talking to a friend or mentor can help.",
        ]
    else:
        return [
            "🌟 Keep nurturing your well-being!",
            "💪 Continue healthy habits: sleep, exercise, gratitude.",
            "📚 Explore mental wellness tips: https://www.mhanational.org/",
        ]
