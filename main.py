import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# load the imdb word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# load the h5 file
model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper function
# function to decode reviews
def decode_reviews(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

# function to pre preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

## step 3 prediction function
def predict_sentiment(review):
    preprocesed_input = preprocess_text(review)
    prediction = model.predict(preprocesed_input)
    sentiment = 'Postive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

import streamlit as st
## streamlit app

st.title('IMDB movie review sentiment analysis')
st.write('Enter a review to classify it as positive or negative')

# user input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)

    # make prediction
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    # display the results
    st.write(f"sentiment: {sentiment}")
    st.write(f"Prediction Score: {prediction[0][0]}")
else:
    st.write('Please enter a moview review')
