import PyPDF2
import numpy
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import re
import tempfile
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('omw-1.4')

# Function to preprocess the text
def preprocess_text(text):
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    # Join the tokens back into a single string
    processed_text = ' '.join(lemmatized_tokens)
    
    return processed_text

def predict(pth):

    tempo_file = tempfile.NamedTemporaryFile(delete=False)
    tempo_file.write(pth.read())
    tempo_file.close()
    model = tf.keras.models.load_model('models/model_large.h5')
    with open(tempo_file.name, 'rb') as f:
        pdf = PyPDF2.PdfReader(f)
        num_pages = len(pdf.pages)
        text = ''

        for i in range(num_pages):
            page = pdf.pages[i]
            text += page.extract_text()

    preprocessed_text = preprocess_text(text)

    tokenize = Tokenizer()
    tokenize.fit_on_texts(preprocessed_text)
    my_data_tokens = tokenize.texts_to_sequences(preprocessed_text)
    my_data_pad = pad_sequences(my_data_tokens, maxlen=max(len(seq) for seq in my_data_tokens), padding='post')
    
    predictions = model.predict(my_data_pad)
    os.remove(tempo_file.name)
    return predictions[0]

def main():
    
    st.title("Resume Acceptance Prediction")

    # Upload PDF file
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file is not None:
        st.write("Analyzing resume...")
        percentage = predict(uploaded_file)
        percentage_rounded = numpy.round(percentage * 100, 2)
        
        # Display the percentage value with larger font size
        st.markdown(f"<h1 style='text-align: center;'>{percentage_rounded}%</h1>", unsafe_allow_html=True)
        
        # Determine the emoji based on the percentage value
        if percentage_rounded >= 80:
            emoji = "😃"
        elif percentage_rounded >= 60:
            emoji = "🙂"
        else:
            emoji = "😕"
        
        # Display the emoji with larger font size
        st.markdown(f"<h1 style='text-align: center;'>{emoji}</h1>", unsafe_allow_html=True)
        
# Run the app
if __name__ == "__main__":
    main()