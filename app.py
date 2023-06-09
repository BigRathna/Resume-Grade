import PyPDF2
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
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
    
    with open(pth, 'rb') as f:
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

    return predictions[0]

def main():
    model = tf.keras.models.load_model('models/model_large.h5')
    st.title("Resume Acceptance Prediction")

    # Upload PDF file
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Preprocess and predict
        predictions = predict(uploaded_file)
        
        # Display predictions
        st.write("Predicted Probability:", predictions)

# Run the app
if __name__ == "__main__":
    main()