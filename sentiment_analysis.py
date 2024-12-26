import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Title
st.title("Sentiment Analysis and Star Prediction")

# Input from user
review = st.text_area("Enter a review:")

if st.button("Analyze"):
    # Calculate sentiment
    polarity_score = sia.polarity_scores(review)['compound']
    if polarity_score > 0:
        sentiment = "Positive"
        stars = 5
    elif polarity_score < 0:
        sentiment = "Negative"
        stars = 1
    else:
        sentiment = "Neutral"
        stars = 3

    # Display results
    st.write(f"Polarity Score: {polarity_score}")
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Predicted Stars: {stars}")
