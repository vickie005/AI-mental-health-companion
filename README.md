# 🧠 AI Mental Health Companion

An empathetic, AI-powered web application designed to provide mental health support, sentiment analysis, and localized resources for users in Kenya. This project aligns with UN Sustainable Development Goal 3: Good Health and Well-being, aiming to make mental health support more accessible through technology.

## 🚀 Overview
The AI Mental Health Companion uses a hybrid approach to provide emotional support:

*Sentiment Analysis*: A custom-trained Scikit-Learn model classifies user input as Positive, Neutral, or Negative.

*Empathetic AI*: For users expressing neutral or negative emotions, the app leverages OpenAI's GPT-4o-mini to provide comforting, non-judgmental guidance.

*Local Resources*: Dynamically provides Kenyan mental health hotlines (e.g., 1199) and professional links based on the detected mood.

## 🛠️ Tech Stack
*Frontend*: HTML5, CSS3, JavaScript

*Backend*: Flask (Python).

*Machine Learning*: Scikit-Learn (Logistic Regression/TF-IDF), Joblib.

*LLM Integration*: OpenAI API (GPT-4o-mini).

*Environment*: Python Dotenv for secure API management.

## 📂 Project Structure

AI-MENTAL-HEALTH-COMPANION/
├── flask_app/
│   ├── static/
│   │   └── style.css         
│   ├── templates/
│   │   └── index.html         
│   └── app.py                
├── app.ipynb      - Model training & EDA
├── chatbot.ipynb       - Local testing environment for the AI
├── sentiment_model.pkl     - Trained ML model
├── vectorizer.pkl         - TF-IDF Vectorizer
├── requirements.txt    
└── .env                       

## ⚙️ Setup & Installation
Clone the Repository:

git clone https://github.com/vickie005/AI-mental-health-companion.git
cd AI-mental-health-companion
Install Dependencies:

pip install -r requirements.txt
Set Up Environment Variables:
Create a .env file in the root directory:

OPENAI_API_KEY=<your_actual_key_here>
Run the Application:

python flask_app/app.py
Visit http://127.0.0.1:5000 in your browser.

## 🌍 SDG 3 Impact: Why This Matters
In Kenya, access to mental health professionals is often limited by cost and stigma. This project contributes to Target 3.4 (Reduce mortality from non-communicable diseases and promote mental health) by:

Providing immediate, 24/7 empathetic responses.

Lowering the barrier to entry for those seeking help.
