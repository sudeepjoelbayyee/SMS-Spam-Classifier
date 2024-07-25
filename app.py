import streamlit as st
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import nltk
import numpy as np

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    text = [lemmatizer.lemmatize(i) for i in text]
    return " ".join(text)

# Load models and data
tfidf = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))
df = pickle.load(open("df.pkl", "rb"))

# Streamlit App
st.title("Email Spam Classifier")
input_sms = st.text_input("Enter the email content below")

if st.button("Predict"):
    # 1. Preprocess
    transform_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transform_sms]).toarray()
    
    # Calculate number of characters in the input email
    num_characters = len(input_sms)
    
    # Stack the number of characters to the vector input
    vector_input = np.hstack((vector_input, np.array([[num_characters]])))

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
