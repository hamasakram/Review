# Sentiment Analysis App

## Project Overview
This project is a **Sentiment Analysis App** built using **Streamlit** and a pre-trained **TensorFlow** model. The app predicts whether a given text review has a positive or negative sentiment based on user input.

## Features
- **User-friendly Interface**: Built with Streamlit for an interactive experience.
- **Pre-trained Model**: Uses a TensorFlow model trained on text data for binary sentiment classification.
- **Text Preprocessing**: Includes tokenization, stopword removal, and lemmatization for accurate predictions.

## Requirements
- Python 3.x
- TensorFlow
- Streamlit
- NLTK

# Usage
## Start the app:

streamlit run sentiment_analyzer.py
1. Open the app in your browser and type a review in the text area.
2. Click the Predict Sentiment button to get the prediction.
Text Preprocessing
Steps:
Convert text to lowercase.
Tokenize text into words.
Remove stopwords using NLTK.
Lemmatize words for normalization.
Example Input and Output
Input: "The product is amazing and works perfectly!"

Output: Positive

Input: "This service is terrible and not worth the money."

Output: Negative

# Key Components
## Preprocessing Function:
Handles text normalization and cleaning.
## Prediction Pipeline:
Uses a tokenizer to convert text into sequences.
Pads sequences for consistent input size.
Predicts sentiment using the loaded model.
## Model Summary
The pre-trained model (my_model.h5) is optimized for binary sentiment classification, with predictions:

>= 0.5: Positive Sentiment
< 0.5: Negative Sentiment
## Contributions
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.
## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/hamasakram/Review.git

