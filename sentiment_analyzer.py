import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Function to download NLTK data if not already present
# Set the proxy server without authentication
nltk.set_proxy('http://proxy.mycompany.com:8080')  # Replace with your actual proxy URL and port

# Function to download NLTK data if not already present
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

# Call this function to download NLTK data on startup
download_nltk_data()

# Load your trained model
@st.cache(allow_output_mutation=True)
def load_my_model():
    return load_model('my_model.h5')

model = load_my_model()

# Define text preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Convert text to lowercase
    text = text.lower()
    # Tokenize text into words
    words = word_tokenize(text)
    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Define maximum sequence length
max_length = 100  # Adjust according to your model training

# Streamlit user interface
st.title('Sentiment Analysis App')
user_input = st.text_area("Type your review here:")

if st.button('Predict Sentiment'):
    try:
        processed_input = preprocess_text(user_input)
        seq = tokenizer.texts_to_sequences([processed_input])
        padded = pad_sequences(seq, maxlen=max_length, padding='post')
        prediction = model.predict(padded)
        sentiment = 'Positive' if prediction >= 0.5 else 'Negative'
        st.write(f'The predicted sentiment is: **{sentiment}**')
    except Exception as e:
        st.error(f"Error predicting sentiment: {e}")
