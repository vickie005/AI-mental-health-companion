from flask import Flask, render_template, request, jsonify
import joblib
import re
import string
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)

try:
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    print("✅ ML Model and Vectorizer loaded successfully.")
except Exception as e:
    print(f"❌ Error loading ML files: {e}")

def clean_text(text):
    """Cleans user input for the ML model."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text.strip()

def emotion_response(sentiment):
    """Returns a basic message and emoji based on sentiment."""
    if sentiment == "positive":
        return "I'm really happy to hear that! Keep shining 💛", "🙂"
    elif sentiment == "negative":
        return "I'm sorry you're going through this. You're not alone 💙", "😞"
    else:
        return "I understand. Feel free to share more.", "😐"

def mental_health_resources(sentiment):
    """Provides specific resources based on the detected mood."""
    if sentiment == "negative":
        return [
            "📞 Kenya Mental Health Hotline: 1199 (Free & Confidential)",
            "💬 Befrienders Kenya: https://befrienderskenya.org/",
            "🧠 WHO Mental Health Resources: https://www.who.int/mental_health/en/",
            "📱 Check-in with someone you trust or a counselor."
        ]
    elif sentiment == "neutral":
        return [
            "🧘 Try journaling or mindfulness for clarity.",
            "📘 Learn about emotional wellness: https://www.mentalhealth.gov/",
            "💬 Talking to a friend or mentor can help."
        ]
    else:
        return [
            "🌟 Keep nurturing your well-being!",
            "💪 Continue healthy habits: sleep, exercise, gratitude.",
            "📚 Explore mental wellness tips: https://www.mhanational.org/"
        ]

def ai_chat_response(user_text):
    """Generates an empathetic AI response using GPT-4o-mini."""
    try:
        prompt = f"Respond with deep empathy, comfort, and supportive emotional guidance to the following input. Keep it concise but warm:\n\n{user_text}"
        
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"API Error: {e}")
        return "I'm here for you. Please tell me more about how you're feeling."

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_text = request.form.get('user_input', '')
        if not user_text:
            return jsonify({'error': 'No input provided'}), 400

        cleaned = clean_text(user_text)
        vectorized = vectorizer.transform([cleaned])
        sentiment = model.predict(vectorized)[0]

        base_response, emoji = emotion_response(sentiment)
        resources = mental_health_resources(sentiment)

        if sentiment in ["negative", "neutral"]:
            ai_response = ai_chat_response(user_text)
        else:
            ai_response = base_response

        return jsonify({
            'sentiment': sentiment,
            'response': ai_response,
            'emoji': emoji,
            'resources': resources
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)