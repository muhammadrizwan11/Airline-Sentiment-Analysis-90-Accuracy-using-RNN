import streamlit as st
#import tensorflow as tf
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the saved model
model_path = 'my_model.h5'  # Update with your actual model path
model = keras.models.load_model(model_path)

# Initialize the tokenizer (assuming same parameters as during training)
max_fatures = 4000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
# You need to fit the tokenizer on the same data as during training
# If you don't have that data readily available, you might need to 
# load a saved tokenizer or re-fit on a representative dataset. 

# Streamlit app
st.title("Airline Sentiment Analysis")

user_input = st.text_area("Enter your tweet about an airline:")

if st.button("Analyze"):
    if user_input:
        # Preprocess the input text
        sample = tokenizer.texts_to_sequences([user_input])
        sample = pad_sequences(sample, maxlen=31, dtype='int32', value=0) 

        # Predict sentiment
        sentiment = model.predict(sample, batch_size=1, verbose=0)[0]
        predicted_class = np.argmax(sentiment)

        if predicted_class == 0:
            st.write("Sentiment: Negative")
        elif predicted_class == 1:
            st.write("Sentiment: Positive")
    else:
        st.write("Please enter some text.")

