from flask import Flask, request, render_template, jsonify
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
import aiohttp
import asyncio
import time
import ssl
import matplotlib
import matplotlib.pyplot as plt
import io
import base64

# Use the Agg backend for Matplotlib
matplotlib.use('Agg')

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)
app.config['CACHE_TYPE'] = 'simple'
cache = Cache(app)
summarizer = pipeline("summarization")

def handle_long_query(query, max_length=300):
    if len(query) > max_length:
        return [query[i:i+max_length] for i in range(0, len(query), max_length)]
    return [query]

def scrape_articles(urls, query):
    queries = handle_long_query(query)
    all_texts = []
    for q in queries:
        for url in urls:
            try:
                response = requests.get(url, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                articles = soup.find_all('article')
                texts = [article.get_text() for article in articles]
                all_texts.extend(texts)
            except requests.exceptions.Timeout:
                print(f"Timeout occurred while trying to access {url}")
            except requests.exceptions.RequestException as e:
                print(f"An error occurred: {e}")
    return all_texts

def google_search(query):
    queries = handle_long_query(query)
    urls = []
    for q in queries:
        urls.extend(search(q, num_results=10))
    return urls

async def fetch(session, url, retries=3):
    ssl_context = ssl.create_default_context()
    ssl_context.verify_flags &= ~ssl.VERIFY_X509_STRICT  # Disable strict verification

    for attempt in range(retries):
        try:
            async with session.get(url, ssl=ssl_context, timeout=10) as response:
                text = await response.read()
                return text.decode('utf-8', errors='ignore')
        except asyncio.TimeoutError:
            print(f"Timeout occurred while trying to access {url}")
        except Exception as e:
            print(f"An error occurred: {e}")
        time.sleep(2)  # Wait before retrying
    return ""

async def scrape_google_async(query):
    queries = handle_long_query(query)
    all_texts = []
    async with aiohttp.ClientSession() as session:
        for q in queries:
            urls = google_search(q)[:5]  # Limit to the first 5 URLs
            tasks = [fetch(session, url) for url in urls]
            texts = await asyncio.gather(*tasks)
            all_texts.extend(texts)
    return all_texts, urls

def scrape_wikipedia(query):
    queries = handle_long_query(query)
    summaries = []
    for q in queries:
        try:
            summary = wikipedia.summary(q)
            summaries.append(summary)
        except wikipedia.exceptions.PageError:
            summaries.append("No matching Wikipedia page found.")
        except wikipedia.exceptions.WikipediaException as e:
            summaries.append(f"An error occurred: {e}")
    return " ".join(summaries)

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
    articles = scrape_articles(urls, text)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    google_results, google_urls = loop.run_until_complete(scrape_google_async(text))
    wikipedia_results = scrape_wikipedia(text)
    all_texts = articles + google_results + [wikipedia_results]
    processed_articles = [preprocess_text(article) for article in all_texts if article]
    results = []
    sources = urls + google_urls + ["Wikipedia"]
    for i, article in enumerate(processed_articles):
        similarity = compare_texts(processed_text, article)
        results.append((i, similarity))

    # Generate bar chart
    fig, ax = plt.subplots()
    indices = [result[0] for result in results]
    scores = [result[1] for result in results]
    ax.bar(indices, scores)
    ax.set_xlabel('Document Index')
    ax.set_ylabel('Similarity Score')
    ax.set_title('Plagiarism Check Results')

    # Save the plot to a PNG image in memory
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('results.html', results=results, sources=sources, plot_url=plot_url, zip=zip)

if __name__ == '__main__':
    app.jinja_env.globals.update(zip=zip)
    app.run(debug=True)
