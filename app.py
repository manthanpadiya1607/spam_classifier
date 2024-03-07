import streamlit as slt
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('Vectorizers.pkl', 'rb'))
model = pickle.load(open('models.pkl', 'rb'))

slt.title("Email/Message Spam Classifier")
input_sms = slt.text_area('Enter the Message')
if slt.button("Predict"):
    # Pre-Processor
    transformed_sms = transform_text(input_sms)
    # Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # Prediction
    result = model.predict(vector_input)[0]
    # Display
    if result == 1:
        slt.header("Spam")
    else:
        slt.header("Not Spam")
