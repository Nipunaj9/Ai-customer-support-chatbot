from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.models import load_model
import random
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the trained model and data
try:
    model = load_model('chatbot_model.h5')
    with open('intents.json', 'r', encoding='utf-8') as file:
        intents = json.load(file)
    with open('words.pkl', 'rb') as file:
        words = pickle.load(file)
    with open('classes.pkl', 'rb') as file:
        classes = pickle.load(file)
    print("Model and data loaded successfully!")
except Exception as e:
    print(f"Error loading model or data: {e}")
    model = None
    intents = []
    words = []
    classes = []

def clean_up_sentence(sentence):
    """Clean and lemmatize the input sentence"""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    """Convert sentence to bag of words array"""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    """Predict the intent class of the input sentence"""
    if model is None:
        return []
    
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    """Get response based on predicted intent"""
    if not intents_list:
        return "I'm sorry, I didn't understand that. Could you please rephrase your question?"
    
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    else:
        result = "I'm sorry, I don't have information about that. Please contact our support team for assistance."
    
    return result

@app.route('/')
def home():
    """Render the main chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Predict intent and get response
        ints = predict_class(message)
        response = get_response(ints, intents)
        
        return jsonify({
            'response': response,
            'intent': ints[0]['intent'] if ints else 'unknown',
            'confidence': ints[0]['probability'] if ints else '0'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    # Download required NLTK data
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except Exception as e:
        print(f"NLTK download warning: {e}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
