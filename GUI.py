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

# Function to handle sending messages
def send_message():
    message = user_input.get()
    print(message)
    corrected_message = spell_check(message)
    ints = predict_class(corrected_message)
    res = get_response(ints, data)
    if message.strip():
        chat_log.config(state=tk.NORMAL)
        chat_log.insert(tk.END, "You: " + corrected_message + "\n")
        user_input.delete(0, tk.END)

        # Bot response (can be modified with chatbot logic)
        bot_response = "Ruggles: " + res
        chat_log.insert(tk.END, bot_response + "\n")
        chat_log.config(state=tk.DISABLED)
        chat_log.yview(tk.END)  # Auto-scroll to the latest message

# Function to clear the chat log
def clear_chat():
    chat_log.config(state=tk.NORMAL)
    chat_log.delete(1.0, tk.END)
    chat_log.config(state=tk.DISABLED)

# Function to close the window
def close_window():
    window.quit()

def spell_check(sentence):
    spell = SpellChecker()
    for words in sentence:
        words = spell.correction(words)
    return sentence
    
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
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
    PTHRESHOLD = .3
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
        return -1

print("Bot is good")

# Create the main window
lemmatizer = WordNetLemmatizer()
data = json.loads(open('C:\witChatBot\.venv\\trainingData.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('model.h5')
window = tk.Tk()
window.title("WitChat Bot")
window.geometry("400x500")

# Chat log (scrollable text area)
chat_log = scrolledtext.ScrolledText(window, state=tk.DISABLED, wrap=tk.WORD, width=50, height=20)
chat_log.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

# Label above user input
label = tk.Label(window, text="What's your question?")
label.grid(row=1, column=0, columnspan=2)

# User input field
user_input = tk.Entry(window, width=40)
user_input.grid(row=2, column=0, padx=10, pady=10)

# Send button
send_button = tk.Button(window, text="Send", command=send_message)
send_button.grid(row=2, column=1, padx=10, pady=10)

# Clear chat button
clear_button = tk.Button(window, text="Clear Chat", command=clear_chat)
clear_button.grid(row=3, column=0, padx=10, pady=10, sticky="w")

# Exit button
exit_button = tk.Button(window, text="Exit", command=close_window)
exit_button.grid(row=3, column=1, padx=10, pady=10, sticky="e")

# Run the Tkinter event loop
window.mainloop()

