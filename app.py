from flask import Flask, render_template, request
import pandas as pd
import nltk
import string
import re
import matplotlib.pyplot as plt
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
    # plot_sentiment_counts(df)

    # # Plot word frequencies using imported function
    # plot_word_frequencies(df)

    return render_template('results.html')

if __name__ == '__main__':
    app.run(debug=True)
