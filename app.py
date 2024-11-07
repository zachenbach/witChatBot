from flask import Flask, render_template, request
import tkinter as tk
from tkinter import scrolledtext
from spellchecker import SpellChecker
import random
import json
import pickle
import numpy as np
import nltk
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()
data = json.loads(open('.venv\\trainingData.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('model.h5')

app = Flask(__name__)
@app.route("/")
def index():
        return render_template('Gui.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        user_input = request.json.get('user_input')  # Get data from HTML form
        if(user_input==None):
            return "Womp Womp"
        
        result = chatBot(user_input)  # Call your Python function with the input
        return result

def chatBot(inputD):  
    print(inputD)
    corrected_message = spell_check(inputD)
    ints = predict_class(corrected_message)
    res = get_response(ints, data)
    return res

def spell_check(sentence):
    sentence = sentence.lower()
    spell = SpellChecker()
    for words in sentence:
        words = spell.correction(words)
    return sentence
    
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    print(sentence_words)
    return sentence_words

def bag_of_words (sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class (sentence):
    bow = bag_of_words (sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.01
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'data': classes [r[0]], 'probability': str(r[1])})
    return return_list

def get_response(data_list, data_json):
    PTHRESHOLD = .7
    tag = data_list[0]['data']
    print(tag)
    worth = False
    probability = data_list[0]['probability']
    print(probability)
    probability = float(probability)
    if(probability > PTHRESHOLD):
        worth = True

    if worth:
        list_of_intents = data_json['data']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice (i['responses'])
                break
        return result
    else:
        return  "I am sorry I do not understand! Please try another question or visit..."

if __name__ == '__main__':
        app.run(debug=True, host='0.0.0.0')

