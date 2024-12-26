#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.feature_extraction.text import CountVectorizer


# In[2]:


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


# In[3]:


df = pd.read_csv('Amazon Customer Reviews.csv')


# In[4]:


print(df.head())


# In[5]:


df.isnull().any()


# In[6]:


df.dropna(subset=['ProfileName', 'Summary'], inplace=True)


# In[7]:


df.duplicated().sum()


# In[4]:


del df['Id']
del df['ProductId']
del df['UserId']
del df['ProfileName']
del df['HelpfulnessNumerator']
del df['HelpfulnessDenominator']
del df['Time']
del df['Summary']


# In[5]:


df.head()


# In[6]:


# df['text'] = df['Text'] + ' ' + df['Summary']
# del df['Text']
# del df['Summary']


# In[7]:


df['Score'] = df['Score'].astype(int)


# In[8]:


def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation
    #text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = text.strip()
    return text


# In[9]:


df['cleaned_text'] = df['Text'].apply(clean_text)


# In[78]:


def tokenize_text(text):
    # Tokenize using NLTK's word_tokenize
    tokens = word_tokenize(text)
    return tokens


# In[79]:


# Apply tokenization
df['tokens'] = df['cleaned_text'].apply(tokenize_text)


# In[80]:


# Display a sample of tokenized data
print(df[['cleaned_text', 'tokens']].head())


# In[81]:


from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

df['tokens_no_stopwords'] = df['tokens'].apply(remove_stopwords)


# In[82]:


stemmer = PorterStemmer()

def stem_tokens(tokens):
    return [stemmer.stem(word) for word in tokens]

df['stemmed_tokens'] = df['tokens_no_stopwords'].apply(stem_tokens)


# In[83]:


nltk.download('vader_lexicon')


# In[10]:


from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize the VADER Sentiment Analyzer
sia = SentimentIntensityAnalyzer()


# In[11]:


# Apply sentiment analysis and add as a new column
df["Polarity_score"] = df["cleaned_text"].apply(lambda x: sia.polarity_scores(x)['compound'])


# In[12]:


# Optionally, classify the sentiment based on score
df["sentiment_label"] = df["Polarity_score"].apply(
    lambda x: "Positive" if x > 0 else "Negative" if x < 0 else "Neutral"
)


# In[13]:


print(df[['Score', 'Polarity_score', 'sentiment_label']])


# # comparison between the generated sentiment labels and the original ones

# In[14]:


def categorize_rating(star):
    if star >= 4:
        return 'Positive'
    elif star == 3:
        return 'Neutral'
    else:
        return 'Negative'

df['original_sentiment'] = df['Score'].apply(categorize_rating)


# In[15]:


print(df[['Score','original_sentiment', 'sentiment_label']])


# In[16]:


df['match'] = df['original_sentiment'] == df['sentiment_label']


# In[17]:


print(df[['match']])


# In[18]:


comparison = df['match'].value_counts()


# In[19]:


import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))  
comparison.plot(kind='bar')  

plt.title('Comparison of original and generated labels', fontsize=16)
plt.xlabel('Comparison', fontsize=14)
plt.ylabel('Count', fontsize=14)

plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# In[20]:


# Count the number of False values in 'Sentiment_Error' column
false_count = (df['match'] == False).sum()

print(f'Number of False values: {false_count}')


# In[21]:


# Count the number of False values in 'Sentiment_Error' column
true_count = (df['match'] == True).sum()

print(f'Number of true values: {true_count}')


# In[22]:


# Calculate accuracy based on the number of False values (correct sentiment labels)
accuracy = (df['match'] == True).mean()

# Print the accuracy
print(f'Accuracy: {accuracy * 100:.2f}%')


# # comparison between the generated stars and the original ones

# In[23]:


def polarity_to_label(polarity):
    if polarity > 0:
        return 5 # Positive
    elif polarity < 0:
        return 1  # Negative
    else:
        return 3  # Neutral


df['generated_Score'] = df['Polarity_score'].apply(polarity_to_label)


# In[24]:


def score_to_label(score):
    if score <= 2:
        return 1  # Negative
    elif score == 3:
        return 3 # Neutral
    else:
        return 5 # Positive

df['original_Score'] = df['Score'].apply(score_to_label)


# In[25]:


df.columns


# In[26]:


print(df[['original_Score', 'original_sentiment', 'sentiment_label', 'generated_Score']])


# In[27]:


df['original_Score'].value_counts()


# In[28]:


original_score = df['original_Score'].value_counts()


# In[29]:


plt.figure(figsize=(6, 4))  
original_score.plot(kind='bar')  

plt.title('original Scores counts', fontsize=16)
plt.xlabel('Comparison', fontsize=14)
plt.ylabel('Count', fontsize=14)

plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# In[30]:


df['generated_Score'].value_counts()


# In[31]:


generated_score = df['generated_Score'].value_counts()


# In[32]:


plt.figure(figsize=(6, 4))  
original_score.plot(kind='bar')  

plt.title('generated Scores counts', fontsize=16)
plt.xlabel('Comparison', fontsize=14)
plt.ylabel('Count', fontsize=14)

plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# In[33]:


# Calculate the accuracy of the generated stars compared to the original stars
correct_predictions = (df['original_Score'] == df['generated_Score']).sum()
total_predictions = len(df)

accuracy = correct_predictions / total_predictions * 100

# Display the results
print(f"Accuracy: {accuracy:.2f}%")


# In[ ]:





# In[46]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[47]:


X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['generated_Score'], test_size=0.2, random_state=42)


# In[109]:


tfidf = TfidfVectorizer(max_features=5000)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# In[110]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)


# In[111]:


y_pred = model.predict(X_test_tfidf)


# In[112]:


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

acc=accuracy_score(y_test,y_pred)
print(acc)


# In[113]:


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

y_pred = model.predict(X_test_tfidf)

print("Classification Report:\n", classification_report(y_test, y_pred))


# In[36]:


# Select a sentence from the dataset based on user input
user_input_index = int(input("Enter the sentence index you want to analyze from the dataset: "))

# Validate the input
if user_input_index < 0 or user_input_index >= len(df):
    print("Index out of range. Please enter a valid index.")
else:
    # Extract the original text
    original_text = df.iloc[user_input_index]['cleaned_text']

    # Extract the original score
    original_score = df.iloc[user_input_index]['Score']

    # Extract the original sentiment
    original_sentiment = df.iloc[user_input_index]['original_sentiment']

    # Predict stars based on calculated sentiment
    generated_Score = df.iloc[user_input_index]['generated_Score']

    # Predict sentiment based on the sentence
    sentiment_label = df.iloc[user_input_index]['sentiment_label']

    # Display the results
    print("\nAnalysis Results:")
    print(f"Text: {original_text}")
    print(f"Original Stars: {original_score}")
    print(f"Original Sentiment: {original_sentiment}")
    print(f"Predicted Stars: {generated_Score}")
    print(f"Predicted Sentiment: {sentiment_label}")

    # Compare stars and sentiment
    stars_match = "Match" if original_score == generated_Score else "Mismatch"
    sentiment_match = "Match" if original_sentiment == sentiment_label else "Mismatch"

    print("\nComparison Results:")
    print(f"Stars Match: {stars_match}")
    print(f"Sentiment Match: {sentiment_match}")

    # Check overall match
    if stars_match == "Match" and sentiment_match == "Match":
        print("\nThe prediction matches perfectly with the original data.")
    else:
        print("\nThere is a discrepancy between the predictions and the original data.")


# In[36]:


df.head()


# In[35]:


# Select a sentence from the dataset based on user input
user_input_index = int(input("Enter the sentence index you want to analyze from the dataset: "))

# Validate the input
if user_input_index < 0 or user_input_index >= len(df):
    print("Index out of range. Please enter a valid index.")
else:
    # Extract the original text
    original_text = df.iloc[user_input_index]['cleaned_text']

    # Extract the original score from original_Score column
    original_score = df.iloc[user_input_index]['original_Score']

    # Extract the original sentiment
    original_sentiment = df.iloc[user_input_index]['original_sentiment']

  # Predict stars based on calculated sentiment
    generated_Score = df.iloc[user_input_index]['generated_Score']

    # Predict sentiment based on the sentence
    sentiment_label = df.iloc[user_input_index]['sentiment_label']

    # Display the results
    print("\nAnalysis Results:")
    print(f"Text: {original_text}")
    print(f"Original Stars: {original_score}")
    print(f"Original Sentiment: {original_sentiment}")
    print(f"Predicted Stars: {generated_Score}")
    print(f"Predicted Sentiment: {sentiment_label}")

    # Compare stars and sentiment
    stars_match = "Match" if original_score == generated_Score else "Mismatch"
    sentiment_match = "Match" if original_sentiment == sentiment_label else "Mismatch"

    print("\nComparison Results:")
    print(f"Stars Match: {stars_match}")
    print(f"Sentiment Match: {sentiment_match}")

    # Check overall match
    if stars_match == "Match" and sentiment_match == "Match":
        print("\nThe prediction matches perfectly with the original data.")
    else:
        print("\nThere is a discrepancy between the predictions and the original data.")


# In[ ]:




