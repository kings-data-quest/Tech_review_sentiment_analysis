
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datapy import search, getComments, clean_text, plot_sentiment_counts, plot_word_frequencies, df


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    # Get the device name from the form
    device_name = request.form['device_name']

    # Search for reviews on YouTube
    urls = search(device_name)

    # Extract comments from the top 15 videos
    comments = []
    for url in urls:
        com = getComments(url)
        comments.extend(com)

    # Save comments to a CSV file
    df = pd.DataFrame({'comments': comments})
    df.to_csv(f'{device_name}_comments.csv', index=False)

    # Clean the comments
    df['cleaned_comments'] = df['comments'].apply(clean_text, device_name=device_name)

    # Generate sentiment analysis graphs
    plot_sentiment(df['cleaned_comments'], device_name)

    # Redirect to the result page
    return redirect(url_for('show_graphs', device_name=device_name))

@app.route('/result/<device_name>')
def show_graphs(device_name):
    # Display the sentiment analysis graphs
    fig, ax = plt.subplots(2, figsize=(10, 8))
    fig.suptitle(f'Sentiment Analysis for {device_name}', fontsize=16)
    sns.histplot(df['polarity'], ax=ax[0])
    ax[0].set_title('Polarity Distribution')
    sns.histplot(df['subjectivity'], ax=ax[1])
    ax[1].set_title('Subjectivity Distribution')
    plt.tight_layout()
    return fig
if __name__ == '__main__':
    app.run(debug=True)
