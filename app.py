from flask import Flask, render_template, request, jsonify
from spellchecker import SpellChecker
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from keras.models import load_model 
from nltk import ConditionalFreqDist, bigrams #new
from nltk.tokenize import word_tokenize #new
nltk.download('universal_tagset')


#Global Resources
SAMPLEDATA = [
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

def prepareTrainingData(sample_data):
    processedData = []
    for sentence in sample_data:
        sentence = sentence.strip()
        processedData.append(sentence)
    
    tokens = []
    for sentence in processedData:
        words = word_tokenize(sentence.lower())
        tokens.extend(words)
    
    #this is to understand word relationships - 2 consecutive words(pairs)
    bigramToken = list(bigrams(tokens))
    #this is to track word patterns - creating frequency distribution of bigrams 
    cfd = ConditionalFreqDist(bigramToken)
    
    return tokens, bigramToken, cfd

TOKENS, BIGRAMS, CFD = prepareTrainingData(SAMPLEDATA)
LEMMATIZER = WordNetLemmatizer()
DATA = json.loads(open('.venv\\nounData.json').read())
DICTIONARY = pickle.load(open('words', 'rb'))
CATEGORIES = pickle.load(open('categories', 'rb'))
MODEL = load_model('trainedModel.h5')
#MCI ( Minimum confidence interval ). This is the confidence cutoff of the bot, if lower than MCI, then category will be discarded.
MCI = 0.5


#Begins rendering ./templates/Gui.html to localhost
app = Flask(__name__)
@app.route("/")
def index():
        return render_template('Gui.html')


#When POST is called under "/submit", checks validity of input, and begins main function
@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        user_input = request.json.get('user_input')
        print("It made it here " + user_input)
        result = chatBot(user_input)
        category, responseStatus, response = result
        return jsonify({"message": response, "category": category}),responseStatus

@app.route('/write', methods=['POST'])
def write():
    message = request.json.get('message')
    category = request.json.get('category')
    print("It made it here " + message + "and here " + category)
    correctedInput = inputCorrector(message)
    for cat in DATA["data"]:
        if cat["category"] == category: 
            if correctedInput not in cat["knownPhrases"]:
                cat["knownPhrases"].append(correctedInput)
            break
    with open('.venv\\nounData.json', 'w') as file:
        json.dump(DATA, file, indent=4)
    return jsonify({"message": "done"})

#Main (chatbot)
#Takes in input, from POST. Sends input to be corrected, then uses the new message to find the intention, then the intention and "trainingData.json" to get a message
def chatBot(inputD):  
    print(inputD)
    correctedInput = inputCorrector(inputD)
    category = predictCategory(correctedInput)
    response = predictBestResponse(category, DATA, correctedInput)
    return response


#inputCorrector. 
# normalizes to lowercase, spell checks, and returns the updated message.
def inputCorrector(sentence):
    sentence = sentence.lower()
    spell = SpellChecker()
    for words in sentence:
        words = spell.correction(words)
    return sentence


#inputNormalizer
#Takes in the input, and tokenizes it ( turns each word into an index in a list ).
#Uses the tokenized input to lemmatize the words ( taking all forms of tyhe same word and grouping them together ). Finally, the functions returns.
def inputNormalizer(input):
    posWords = pos_tag(nltk.word_tokenize(input),tagset='universal')
    nouns = [word for word, pos in posWords if pos in {"NOUN","VERB","PROPN","ADJ","ADV","ADP","PRON"}]
    nouns = [LEMMATIZER.lemmatize(word) for word in nouns]
    print(nouns)
    return nouns


#wordArray
#This function takes in our cleaned up input, tokenizes it, and then creates an array of 0's, with the same length as dictionary
#Dictionary is a library of every word that is used in trainingData. So, when it checks each word in our tokenized input, it is checking if that word is in trainingData.
def wordArray(input):
    inputtedWords = inputNormalizer(input)
    wordArray = [0] * len(DICTIONARY)
    for inputWord in inputtedWords:
        for idx, dictionaryWord in enumerate(DICTIONARY):
            if dictionaryWord == inputWord:
                wordArray[idx] = 1
    return np.array(wordArray)


#predictCategory
#This function takes in the input, makes it into an array based on wordArray. It then uses that array to predict the likelihood it could be in any category and place those into a list.
#This list is checked against MCI, keeping any categories higher than it. And finally, it normalizes the list, so it can be read later ( see line 145-146 ). Returns finalCategories
def predictCategory(input):
    inputArray = wordArray(input)
    response = MODEL.predict(np.array([inputArray]))[0]
    categoryList = [[idx, r] for idx, r in enumerate(response) if r > MCI]
    categoryList.sort(key=lambda x: x[1], reverse=True)
    finalCategories = []
    for r in categoryList:
        finalCategories.append({'category': CATEGORIES[r[0]], 'probability': str(r[1])})
    return finalCategories


#predictBestReponse
#Finds the first category and its probability from the inputted list of suitable Categories. Checks it against PTHRESHOLD. 
#If it is above PTHRESHOLD it will get a reposnse from that category, pulled from trainingData.json or dataJSON. returns reponse.
#If it is below, it runs noun check.
def predictBestResponse(dataList, dataJSON,input):
    worth = False
    try: 
        if not dataList:
            return {0,0,"I'm sorry, could you be more specific?"}
        

        category = dataList[0]['category']
        probability = dataList[0]['probability']
        print("Category: " + category + "\nChance: " + probability)
        probability = float(probability)
        

        pthreshold = 0.65 if len(input) <= 4 else 0.25
        if(probability > pthreshold):
            worth = True

        if worth:
            categoryList = dataJSON['data']
            for idx in categoryList:
                if idx['category'] == category:
                    response = random.choice (idx['response'])
                    break
            if(category in {"thanks","greetings","goodbyes"}):
                return category, 0, response
            else:
                return category, 1, response
        else:
            response = generateNLGResponse(input)
            return category, 0, response
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        response = generateNLGResponse(None)
        return category, 0, response

#generateNLGResponse
#Takes in the category, finds a response to respond with. If it containts a question identifier, generate a response, if not choose from defaultResponses.
def generateNLGResponse(input):

    defaultResponses = [
        "I'm here to help. Could you please provide more details?",
        "I'd be happy to assist you. Could you be more specific?",
        "Let me help you with that. What exactly would you like to know?",
        "I'm listening. Please tell me more about what you need.",
        "I want to make sure I understand correctly. Could you elaborate?"
    ]
    #if question is not clear, use one of the above responses
    if not input in ['what', 'why', 'how']:
        return random.choice(defaultResponses)
        
    #generating a more specific response based on intent
    words = word_tokenize(input)
    #checks if there is any matching word so it can provide answer based on intent
    meaningfulWords = [w for w in words 
                       if w.lower() in CFD 
                       and w.lower() not in {'what', 'the', 'a', 'an', 'is', 'are'}]
    
    if meaningfulWords:
        response = generateSentence(random.choice(meaningfulWords), CFD, length=15)
        if len(response.split()) > 3:
            return response
    
    return random.choice(defaultResponses)

#generateSentences
#Generates sentence by checking against frequency distribution. After this it generates a sentence one word at a time, using the given word. Returns "None" or sentence.
def generateSentence(startWord, cfd, length=10):
      sentence = []
      currentLength = 0
      maxLength = length

      #Keeping track of previous used words to avoid repetition
      usedWords = set()
    

      while currentLength < maxLength:
        #checking if the current word is in frequency distrubiton
        if word not in cfd or not cfd[word].items():
            break
            
        #guessing the potential next word and freuqencies
        nextWords = list(cfd[word].items())
        #clean up used words
        nextWords = [(w, f) for w, f in nextWords if w not in usedWords]

        if not nextWords:
            break

        #calculating the probabilities for the next words
        total = sum(freq for _, freq in nextWords)
        probabilities = [freq/total for _, freq in nextWords]
        
        #choosing next word based on above probs
        nextWord = random.choices(
            [w for w, _ in nextWords],
            weights=probabilities,
            k=1
        )[0]
       
        if nextWord not in usedWords:
            sentence.append(nextWord)
            usedWords.add(nextWord)
        
        word = nextWord
        current_length += 1
    
      if not sentence:
        return "Could you please rephrase that?"
        
      response = ' '.join(sentence)
      response = response.capitalize()
     
      return response


if __name__ == '__main__':
        app.run(debug=True, host='0.0.0.0')

