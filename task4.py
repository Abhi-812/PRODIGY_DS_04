import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

url = 'https://raw.githubusercontent.com/guyz/twitter-sentiment-dataset/master/corpus.csv'
df = pd.read_csv(url, encoding='ISO-8859-1')

df = df[['positive', '126415614616154112']]

df.columns = ['Sentiment', 'Tweet']

df['Tweet'] = df['Tweet'].astype(str)

df['Sentiment'] = df['Sentiment'].map({'positive': 'Positive'})  # Modify as needed if there are more sentiment labels

sid = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    sentiment_scores = sid.polarity_scores(text)
    if sentiment_scores['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['VADER_Sentiment'] = df['Tweet'].apply(analyze_sentiment)

plt.figure(figsize=(8, 6))
sns.countplot(x='Sentiment', data=df, palette='coolwarm')
plt.title('Sentiment Distribution (Original Data)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='VADER_Sentiment', data=df, palette='coolwarm')
plt.title('Sentiment Distribution (VADER Analysis)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

sentiment_counts = df['Sentiment'].value_counts(normalize=True) * 100
print("\nSentiment Breakdown (Original Data) in %:")
print(sentiment_counts)

vader_sentiment_counts = df['VADER_Sentiment'].value_counts(normalize=True) * 100
print("\nSentiment Breakdown (VADER Analysis) in %:")
print(vader_sentiment_counts)

print("\nFinal Dataframe with Sentiments:")
print(df.head())
