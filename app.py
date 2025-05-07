from flask import Flask, request, render_template
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from difflib import SequenceMatcher
from googlesearch import search
from transformers import pipeline
from flask_caching import Cache
import wikipedia

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)
app.config['CACHE_TYPE'] = 'simple'
cache = Cache(app)

summarizer = pipeline("summarization")

def scrape_articles(urls):
    all_texts = []
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('article')
        texts = [article.get_text() for article in articles]
        all_texts.extend(texts)
    return all_texts

def google_search(query):
    urls = []
    for url in search(query, num_results=10):
        urls.append(url)
    return urls

def scrape_google(query):
    urls = google_search(query)
    all_texts = []
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        texts = soup.get_text()
        all_texts.append(texts)
    return all_texts

def scrape_wikipedia(query):
    summary = wikipedia.summary(query)
    return summary

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    tokens = [stemmer.stem(word) for word in tokens]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def compare_texts(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/check', methods=['POST'])
@cache.cached(timeout=300, key_prefix='plagiarism_check')
def check_plagiarism():
    text = request.form['text']
    processed_text = preprocess_text(text)
    urls = [
        'https://arxiv.org/list/cs.AI/recent',
        'https://pubmed.ncbi.nlm.nih.gov/',
        'https://dl.acm.org/',
        'https://www.frontiersin.org/journals/computer-science/articles'
    ]
    articles = scrape_articles(urls)
    google_results = scrape_google(text)
    wikipedia_results = scrape_wikipedia(text)
    all_texts = articles + google_results + [wikipedia_results]
    processed_articles = [preprocess_text(article) for article in all_texts]
    results = []
    for i, article in enumerate(processed_articles):
        similarity = compare_texts(processed_text, article)
        results.append((i, similarity))
    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
