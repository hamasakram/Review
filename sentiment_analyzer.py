import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load the model and tokenizer
@st.cache(allow_output_mutation=True)
def load_resources():
    model = load_model('my_model.h5')
    # Load tokenizer from a pickle file
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_resources()

# Define text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    words = word_tokenize(text)  # Tokenize text into words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Remove stopwords and lemmatize
    return " ".join(words)

# Define the maximum sequence length (should match the training setup)
max_length = 100

# Setup Streamlit user interface
st.title('Sentiment Analysis App')
user_input = st.text_area("Type your review here:")

if st.button('Predict Sentiment'):
    try:
        processed_input = preprocess_text(user_input)
        seq = tokenizer.texts_to_sequences([processed_input])  # Tokenize
        padded = pad_sequences(seq, maxlen=max_length, padding='post')  # Pad
        prediction = model.predict(padded)  # Predict
        sentiment = 'Positive' if prediction >= 0.5 else 'Negative'
        st.write(f'The predicted sentiment is: **{sentiment}**')
    except Exception as e:
        st.error(f"Error predicting sentiment: {str(e)}")

# Optional: Add a button to clear the input
if st.button('Clear'):
    st.experimental_rerun()
