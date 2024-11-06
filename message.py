import tkinter as tk
from tkinter import scrolledtext
import random
import json
import pickle
import numpy as np
import nltk

from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from nltk import ConditionalFreqDist, bigrams #new
from nltk.tokenize import word_tokenize #new

sample_data = [
    # General responses
    "How can I help you today?",
    "What can I assist you with?",
    "I'm here to help answer your questions",
    "Please let me know what information you need",
    "What would you like to know?",
    "I can provide information about various topics",
    "How may I assist you?",
    "What brings you here today?",
    
    # Acknowledgments
    "Sure, I understand",
    "Sure, let me help you with that",
    "Of course, I can help",
    "Yes, I can assist you",
    "Thank you for asking",
    "I appreciate your question",
    
    # Follow-ups
    "Would you like to know more?",
    "Is there anything else you need?",
    "Do you have any other questions?",
    "Let me know if you need more information",
    "Would you like me to explain further?",
    
    # Clarifications
    "Could you please provide more details?",
    "Would you mind elaborating on that?",
    "I'm not sure I understand, could you explain more?",
    "Please tell me more about your question",
    "What specific information are you looking for?",
    
    # Common responses
    "That's a good question",
    "I'll help you find the information",
    "Let me check that for you",
    "I'd be happy to help with that",
    "Let's figure this out together"
]

def prepare_training_data(sample_data):
    processed_data = []
    for sentence in sample_data:
        sentence = sentence.strip()
        processed_data.append(sentence)
    
    tokens = []
    for sentence in processed_data:
        words = word_tokenize(sentence.lower())
        tokens.extend(words)
    
    #this is to understand word relationships - 2 consecutive words(pairs)
    bigrams_tokens = list(bigrams(tokens))
    #this is to track word patterns - creating frequency distribution of bigrams 
    cfd = ConditionalFreqDist(bigrams_tokens)
    
    return tokens, bigrams_tokens, cfd

tokens, bigrams_tokens, cfd = prepare_training_data(sample_data)

def generate_sentence(start_word, cfd, length=10):
      word = start_word.lower()
      sentence = []
      current_length = 0
      max_length = length

      #Keeping track of previous used words to avoid repetition
      used_words = set()
    

      while current_length < max_length:
        #checking if the current word is in frequency distrubiton
        if word not in cfd or not cfd[word].items():
            break
            
        #guessing the potential next word and freuqencies
        next_words = list(cfd[word].items())
        #clean up used words
        next_words = [(w, f) for w, f in next_words if w not in used_words]

        if not next_words:
            break

        #calculating the probabilities for the next words
        total = sum(freq for _, freq in next_words)
        probabilities = [freq/total for _, freq in next_words]
        
        #choosing next word based on above probs
        next_word = random.choices(
            [w for w, _ in next_words],
            weights=probabilities,
            k=1
        )[0]
       
        if next_word not in used_words:
            sentence.append(next_word)
            used_words.add(next_word)
        
        word = next_word
        current_length += 1
    
      if not sentence:
        return "Could you please rephrase that?"
        
      result = ' '.join(sentence)
      result = result.capitalize()
     
      return result
# Function to handle sending messages
def send_message():
    message = user_input.get()
    print(f"\nUser input: {message}")
    
    if not message.strip():
        return
        
    try:
        chat_log.config(state=tk.NORMAL)
        chat_log.insert(tk.END, "You: " + message + "\n")
        user_input.delete(0, tk.END)
        
        corrected_message = spell_check(message)
        
        ints = predict_class(corrected_message)
        
        # Generate response
        if ints:  
            try:
                bot_response = get_response(ints, data)
            except Exception as e:
                print(f"Error in get_response: {str(e)}")
                bot_response = generate_nlg_response(None)  #Fallback to NLG
        else:
            #If no intents matched, use NLG
            bot_response = generate_nlg_response(None)
            
        #print(f"Generated response: {bot_response}")
        
        chat_log.insert(tk.END, "Bot: " + bot_response + "\n")
        chat_log.config(state=tk.DISABLED)
        chat_log.yview(tk.END)
        
    except Exception as e:
        chat_log.config(state=tk.NORMAL)
        chat_log.insert(tk.END, "Bot: Let me try to help you with that.\n")
        chat_log.config(state=tk.DISABLED)
        chat_log.yview(tk.END)

# Function to clear the chat log
def clear_chat():
    chat_log.config(state=tk.NORMAL)
    chat_log.delete(1.0, tk.END)
    chat_log.config(state=tk.DISABLED)

# Function to close the window
def close_window():
    window.quit()

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
    is_short_query = len(sentence.split()) <= 1

    bow = bag_of_words (sentence)
    res = model.predict(np.array([bow]))[0]
    #using multiple thresholds based on length
    ERROR_THRESHOLD = 0.60 if is_short_query else 0.25    
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    #print(f"Raw predictions: {res}")
    #print(f"Results above threshold: {results}")

    return_list = []
    for r in results:
        return_list.append({'data': classes [r[0]], 'probability': str(r[1])})
    #print("Prediction results:", return_list) 
    return return_list

def generate_nlg_response(intent_tag):
    default_responses = [
        "I'm here to help. Could you please provide more details?",
        "I'd be happy to assist you. Could you be more specific?",
        "Let me help you with that. What exactly would you like to know?",
        "I'm listening. Please tell me more about what you need.",
        "I want to make sure I understand correctly. Could you elaborate?"
    ]
    #if question is not clear, use one of the above responses
    if not intent_tag or intent_tag.lower() in ['what', 'why', 'how']:
        return random.choice(default_responses)
        
    #generating a more specific response based on intent
    words = word_tokenize(intent_tag.lower())
    #checks if there is any matching word so it can provide answer based on intent
    meaningful_words = [w for w in words 
                       if w.lower() in cfd 
                       and w.lower() not in {'what', 'the', 'a', 'an', 'is', 'are'}]
    
    if meaningful_words:
        response = generate_sentence(random.choice(meaningful_words), cfd, length=15)
        if len(response.split()) > 3:
            return response
    
    return random.choice(default_responses)

def get_response(data_list, data_json):

    try: 
        if not data_list:
            return generate_nlg_response(None)
            
        tag = data_list[0]['data']
        probability = float(data_list[0]['probability'])

        threshold = 0.60 if len(tag) <= 4 else 0.25
        
        if probability > threshold:
            list_of_intents = data_json['data']
            for i in list_of_intents:
                if i['tag'] == tag:
                    return random.choice(i['responses'])


        #falling back to NLG if threshold is low 
        return generate_nlg_response(data_list[0]['data'])
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        return generate_nlg_response(None)  

print("Bot is good")

# Create the main window
lemmatizer = WordNetLemmatizer()
data = json.loads(open('/Users/davudazizov/Desktop/witChatBot/intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')
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
