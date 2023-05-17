from flask import Flask, render_template, request, after_this_request
import pandas as pd
import threading
import nltk
import string
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import seaborn as sns
import demoji

from time import sleep
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as op
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from wordcloud import WordCloud
from nltk import pos_tag
from selenium.common.exceptions import NoSuchElementException, WebDriverException
from datapy import analyze_sentiments, clean_text, remove_emojis


stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def generate_wordcloud(df):
    # Join the words in the cleaned comments to a single string
    text = ' '.join(df['cleaned_comments'])

    # Create a word cloud object and generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text)

    # Plot the word cloud
    plt.figure(figsize=(12,10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('static/wordcloud.png')
    plt.close()

def plot_sentiment_counts(df):
    # Set the style and color palette
    sns.set_style('darkgrid')
    sns.set_palette('husl')

    # Get the count of comments in each sentiment category
    sentiment_counts = df['sentiment_category'].value_counts().sort_index()

    # Plot the bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax)
    ax.set_xlabel('Sentiment Category')
    ax.set_ylabel('Number of Comments')
    ax.set_title('Sentiment Analysis Results')

    # Add labels to the bars
    for i, v in enumerate(sentiment_counts.values):
        ax.text(i, v+10, str(v), ha='center', fontweight='bold', fontsize=12)

    # Label the bars
    for i, label in enumerate(ax.get_xticklabels()):
        sentiment = label.get_text()
        if sentiment == 'positive':
            label.set_text(f'{sentiment.capitalize()} comments')
            label.set_color('green')
        elif sentiment == 'neutral':
            label.set_text(f'{sentiment.capitalize()} comments')
            label.set_color('gray')
        else:
            label.set_text(f'{sentiment.capitalize()} comments')
            label.set_color('red')
            # Update the label for negative sentiment category
            label.set_text(f'{sentiment.capitalize()} comments ({sentiment_counts[sentiment]})')
        label.set_fontweight('bold')
        label.set_fontsize(12)

    plt.savefig('static/count.png')
    plt.close()

def plot_word_frequencies(df):
    # Define the stop words
    stop_words = set(stopwords.words('english'))

    # Define the minimum word frequency threshold
    min_freq = 10

    # Create separate dataframes for each sentiment category
    positive_df = df[df['sentiment_category'] == 'positive']
    neutral_df = df[df['sentiment_category'] == 'neutral']
    negative_df = df[df['sentiment_category'] == 'negative']

    # Tokenize the comments and remove stop words for each dataframe
    positive_words = tokenize_comments(positive_df['cleaned_comments'])
    neutral_words = tokenize_comments(neutral_df['cleaned_comments'])
    negative_words = tokenize_comments(negative_df['cleaned_comments'])

    # Count the frequency of each word in each dataframe
    positive_freq = nltk.FreqDist(positive_words)
    neutral_freq = nltk.FreqDist(neutral_words)
    negative_freq = nltk.FreqDist(negative_words)

    # Remove words that appear less frequently than the minimum frequency threshold
    positive_freq = {k: v for k, v in positive_freq.items() if v >= min_freq}
    neutral_freq = {k: v for k, v in neutral_freq.items() if v >= min_freq}
    negative_freq = {k: v for k, v in negative_freq.items() if v >= min_freq}

    # Sort the dictionaries by word frequency in descending order
    positive_freq = dict(sorted(positive_freq.items(), key=lambda item: item[1], reverse=True))
    neutral_freq = dict(sorted(neutral_freq.items(), key=lambda item: item[1], reverse=True))
    negative_freq = dict(sorted(negative_freq.items(), key=lambda item: item[1], reverse=True))

    # Plot the top 20 most frequent words in each sentiment category
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.bar(list(positive_freq.keys())[:20], list(positive_freq.values())[:20])
    plt.title('Most Frequent Words in Positive Comments')
    plt.xlabel('Word')
    plt.ylabel('Frequency')

    plt.subplot(3, 1, 2)
    plt.bar(list(neutral_freq.keys())[:20], list(neutral_freq.values())[:20])
    plt.title('Most Frequent Words in Neutral Comments')
    plt.xlabel('Word')
    plt.ylabel('Frequency')

    plt.subplot(3, 1, 3)
    plt.bar(list(negative_freq.keys())[:20], list(negative_freq.values())[:20])
    plt.title('Most Frequent Words in Negative Comments')
    plt.xlabel('Word')
    plt.ylabel('Frequency')

    plt.tight_layout()

    plt.savefig('static/frequency.png')
    plt.close()

def tokenize_comments(comment_series):
    # Combine the comments into a single string
    all_comments = ' '.join(comment_series.tolist())

    # Tokenize the string into words
    words = word_tokenize(all_comments)

    # Remove stop words
    filtered_words = [word for word in words if word.lower() not in stop_words]

    return filtered_words

@app.route('/results', methods=['POST'])
def results():
    device_name = request.form['device_name']
    query = device_name.lower()

    df = pd.read_csv(f'{query}_comments.csv')

    df = df.astype(str)

    # Analyze sentiments using imported function
    df = analyze_sentiments(df)

    # Generate word cloud using imported function
    generate_wordcloud(df)

    # # Plot sentiment counts using imported function
    plot_sentiment_counts(df)

    # Plot word frequencies using imported function
    plot_word_frequencies(df)

    @after_this_request
    def cleanup(response):
        plt.close('all')  # Close all existing Matplotlib figures
        return response

    return render_template('results.html')

if __name__ == '__main__':
    app.run(debug=True)

