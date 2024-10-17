import random
import json
import pickle
import numpy as np
import nltk

from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()
data = json.loads(open('C:\witChatBot\.venv\\trainingData.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('model.h5')

def spell_check(sentence):
    blob = TextBlob(sentence)
    lowercase = blob.lower()
    corrected_sentence = lowercase.correct()
    return str(corrected_sentence)
    
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

while True:
    message = input("You: ")
    if message.lower() == "exit":
        print("Exiting the chat. Goodbye!")
        break
    corrected_message = spell_check(message)
    print(f"Corrected input: {corrected_message}")
    ints = predict_class(corrected_message)
    res = get_response(ints, data)
    if res == -1:
        print("That Didn't Work, try again.")
    else:
        print(res)

    