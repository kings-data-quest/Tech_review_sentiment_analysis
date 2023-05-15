from flask import Flask, render_template, request
from datapy import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

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

    # Plot sentiment counts using imported function
    plot_sentiment_counts(df)

    # Plot word frequencies using imported function
    plot_word_frequencies(df)

    return render_template('results.html')

if __name__ == '__main__':
    app.run(debug=True)
