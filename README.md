# Amazon Fine Food Reviews Sentiment Analysis

## Overview
This project focuses on sentiment analysis using the Amazon Fine Food Reviews dataset. The goal is to analyze customer reviews and determine their sentiment using various machine learning techniques.

## Dataset
The dataset used in this project is the [Amazon Fine Food Reviews dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews). It contains reviews of fine foods from Amazon, including user ratings, text reviews, and metadata.

## Features
- **Id**: Unique identifier for each review
- **ProductId**: Unique identifier for each product
- **UserId**: Unique identifier for each user
- **ProfileName**: Name of the user
- **HelpfulnessNumerator**: Number of users who found the review helpful
- **HelpfulnessDenominator**: Number of users who indicated whether they found the review helpful
- **Score**: Rating given by the user (1 to 5)
- **Time**: Timestamp for the review
- **Summary**: Short summary of the review
- **Text**: Full text of the review

## Objective
The main objectives of this project include:
- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering
- Applying machine learning models for sentiment classification
- Evaluating model performance

## Installation
To run this project, install the required dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
Run the main script to preprocess data and train models:
```bash
python main.py
```

## Results
The performance of different models is evaluated using accuracy, precision, recall, and F1-score.



