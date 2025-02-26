# Amazon Customer Reviews Sentiment Analysis

## Overview
This project performs sentiment analysis on Amazon customer reviews. It processes text data by cleaning, tokenizing, removing stopwords, stemming, and applying sentiment analysis using VADER (Valence Aware Dictionary and sEntiment Reasoner). The results are then compared with the original sentiment labels derived from review scores.

## Dataset
The dataset consists of Amazon customer reviews, including:
- **Review Text**: Customer feedback
- **Review Score**: Numeric rating (used to determine original sentiment)

## Preprocessing Steps
1. **Text Cleaning**:
   - Remove URLs, HTML tags, punctuation, and numbers
   - Convert text to lowercase
   - Remove extra whitespace

2. **Tokenization**:
   - Split text into words

3. **Stopword Removal**:
   - Eliminate common words that don’t contribute to sentiment (e.g., "the", "is")

4. **Stemming**:
   - Reduce words to their root form using `PorterStemmer`

## Sentiment Analysis
- **VADER SentimentIntensityAnalyzer** assigns polarity scores to reviews.
- The sentiment labels are determined as:
  - Positive: `compound score > 0.05`
  - Neutral: `-0.05 <= compound score <= 0.05`
  - Negative: `compound score < -0.05`

## Results
- Sentiment classification from VADER is compared to the original sentiment derived from review scores.
- Visualizations are used to analyze discrepancies.

## Installation
To run this project, install the required dependencies:
```bash
pip install nltk pandas matplotlib seaborn
```

## Usage
Run the script to analyze sentiment:
```bash
python sentiment_analysis.py
```

## Future Improvements
- Use a machine learning-based sentiment classifier (e.g., Logistic Regression, Naïve Bayes, or BERT).
- Fine-tune sentiment thresholds for better accuracy.
- Expand preprocessing with lemmatization instead of stemming.

## License
This project is open-source and available under the MIT License.

## Contact
For any questions or contributions, feel free to reach out!

