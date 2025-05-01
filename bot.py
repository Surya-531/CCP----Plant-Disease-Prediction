from flask import Flask, request, jsonify
import google.generativeai as genai
from flask_cors import CORS
from deep_translator import GoogleTranslator

app = Flask(__name__)
CORS(app)  # Allow all origins

# Configure the API
api_key = "AIzaSyBusmdXcpH2qvFtwH3sSAP6mZq3m1noFu8"  # Replace with your actual API key
genai.configure(api_key=api_key)

# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

chat_session = model.start_chat(history=[])

# Predefined stored questions and answers (English)
stored_qa = {
    "What is your name?": "I am a chatbot powered by Gemini AI.",
    "How does AI work?": "AI works by using algorithms and data to perform tasks that usually require human intelligence.",
    "Who created you?": "I was created by Surya and Sudharsanam.",
    "How do I submit a complaint?": "You can submit a complaint by visiting the complaints portal and filling out the form with the necessary details.",
    "Can I submit an anonymous complaint?": "Yes, anonymous complaints are allowed, but we can't send you updates.",
}

def detect_language(text):
    """Detect the language of the user input."""
    translator = GoogleTranslator(source='auto', target='en')
    detected_text = translator.translate(text)
    return detected_text, translator.source

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"response": "Please enter a message."})

    # Detect and translate to English
    translated_input, detected_lang = detect_language(user_message)

    # Check if the input matches stored Q&A
    response = stored_qa.get(translated_input, None)

    if not response:
        # If not found, generate a response using the AI model
        chat_response = chat_session.send_message(translated_input)
        response = chat_response.text

    # Translate the response back to the detected language
    if detected_lang != "en":
        response = GoogleTranslator(source="en", target=detected_lang).translate(response)

    return jsonify({"response": response, "language": detected_lang})

if __name__ == '__main__':
    app.run(debug=True)
