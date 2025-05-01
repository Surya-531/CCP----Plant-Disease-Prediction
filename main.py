from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai
import keras
import numpy as np
import cv2 as cv

app = Flask(__name__)
CORS(app)

### --- Chatbot Section --- ###

# Configure Gemini API
api_key = "AIzaSyBusmdXcpH2qvFtwH3sSAP6mZq3m1noFu8"  # Replace your valid API key
genai.configure(api_key=api_key)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model_ai = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

chat_session = model_ai.start_chat(history=[])

# Predefined Q&A
stored_qa = {
    "What is your name?": "I am a chatbot powered by AGRO360.",
    "How does AI work?": "AI works by using algorithms and data to perform tasks that usually require human intelligence.",
    "Who created you?": "I was created by Surya and Nedesh Kumar.",
    "How do I submit a complaint?": "You can submit a complaint by visiting the complaints portal and filling out the form with the necessary details.",
    "Can I submit an anonymous complaint?": "Yes, anonymous complaints are allowed, but we can't send you updates.",
}

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"response": "Please enter a message."})

    # Directly check predefined Q&A
    response = stored_qa.get(user_message, None)

    if not response:
        # If not found, generate a response using Gemini AI
        chat_response = chat_session.send_message(user_message)
        response = chat_response.text

    return jsonify({"response": response})

### --- Leaf Disease Prediction Section --- ###

# Load model and labels
label_name = ['Apple scab','Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 'Cherry Powdery mildew',
'Cherry healthy','Corn Cercospora leaf spot Gray leaf spot', 'Corn Common rust', 'Corn Northern Leaf Blight','Corn healthy', 
'Grape Black rot', 'Grape Esca', 'Grape Leaf blight', 'Grape healthy','Peach Bacterial spot','Peach healthy', 'Pepper bell Bacterial spot', 
'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy', 'Strawberry Leaf scorch', 'Strawberry healthy',
'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
'Tomato Spider mites', 'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy']

model = keras.models.load_model(r"Leaf Deases(96,88).h5")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv.imdecode(npimg, cv.IMREAD_COLOR)
    normalized_image = np.expand_dims(cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (150, 150)), axis=0)
    predictions = model.predict(normalized_image)
    confidence = predictions[0][np.argmax(predictions)] * 100

    if confidence >= 80:
        result = label_name[np.argmax(predictions)]
    else:
        result = "Uncertain result, please try another image."

    return jsonify({'result': result})

### --- Serving HTML --- ###

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
