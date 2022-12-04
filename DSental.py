# Import library for ReGex, SQLite, Pandas, Numpy, and joblib
import re
import sqlite3
import pandas as pd
import numpy as np
import joblib

# Import library for Tokenize, Stemming, and Stopwords
import nltk
from nltk import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords as stopwords_scratch

# Import library for SKLearn Model Sentiment Analysis
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import train_test_split

# Import library for Tensorflow Model Sentiment Analysis
from keras import optimizers
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM,Dense
from keras.layers import Embedding
from keras.callbacks import EarlyStopping
from keras.models import load_model

# Import library for Flask
from flask import Flask, jsonify
from flask import request, make_response
from flask_swagger_ui import get_swaggerui_blueprint

# Swagger UI Definition
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False # Mengubah order JSON menjadi urutan yang benar
SWAGGER_URL = '/swagger'
API_URL = '/static/restapi_sentiment.yml'
SWAGGERUI_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "DSental (Data Sentiment Analysis)"
    }
)
app.register_blueprint(SWAGGERUI_BLUEPRINT, url_prefix=SWAGGER_URL)

# Connect to db & csv
conn = sqlite3.connect('data/output.db', check_same_thread=False)
df_alay = pd.read_csv('data/new_kamusalay.csv', names=['alay','cleaned'], encoding ='latin-1')
data_raw = pd.read_csv(r'data/train_preprocess.tsv', sep='\t',names=['text','label'])
data_raw.drop_duplicates()

# Define and Execute query for unexistence data tables
# Tables will contain fields with dirty text (text & file) and cleaned text (text & file)
conn.execute('''CREATE TABLE IF NOT EXISTS data_text_sk (text_id INTEGER PRIMARY KEY AUTOINCREMENT, text varchar(255), sentiment varchar(255));''')
conn.execute('''CREATE TABLE IF NOT EXISTS data_file_sk (text_id INTEGER PRIMARY KEY AUTOINCREMENT, text varchar(255), sentiment varchar(255));''')
conn.execute('''CREATE TABLE IF NOT EXISTS data_text_tf (text_id INTEGER PRIMARY KEY AUTOINCREMENT, text varchar(255), sentiment varchar(255));''')
conn.execute('''CREATE TABLE IF NOT EXISTS data_file_tf (text_id INTEGER PRIMARY KEY AUTOINCREMENT, text varchar(255), sentiment varchar(255));''')

# Create Stopwords
list_stopwords = stopwords_scratch.words('indonesian')
list_stopwords_en = stopwords_scratch.words('english')
list_stopwords.extend(list_stopwords_en)
list_stopwords.extend(['ya','yg','ga','yuk','dah','baiknya', 'berkali', 'kali', 'kurangnya', 'mata', 'olah', 'sekurang', 'setidak', 'tama', 'tidaknya'])

# Add External Stopwords
f = open("stopwords/tala-stopwords-indonesia.txt", "r")
stopword_external = []
for line in f:
    stripped_line = line.strip()
    line_list = stripped_line.split()
    stopword_external.append(line_list[0])
f.close()
list_stopwords.extend(stopword_external)
stopwords = list_stopwords

# Creating Function for Cleansing Process
def lowercase(text): # Change uppercase characters to lowercase
    return text.lower()

def special(text): # Remove all the special characters
    text = re.sub(r'\W', ' ',str(text), flags=re.MULTILINE)
    return text

def single(text): # remove all single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text, flags=re.MULTILINE)
    return text

def singlestart(text): # Remove single characters from the start
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text, flags=re.MULTILINE)
    return text

def mulspace(text): # Substituting multiple spaces with single space
    text = re.sub(r'\s+', ' ', text, flags=re.MULTILINE)
    return text

def rt(text): # Remove RT
    text = re.sub(r'rt @\w+: ', ' ', text, flags=re.MULTILINE)
    return text

def prefixedb(text): # Removing prefixed 'b'
    text = re.sub(r'^b\s+', '', text, flags=re.MULTILINE)
    return text

def misc(text): # Remove URL, Mention, Hashtag, user, Line, and Tab
    text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))|([#@]\S+)|user|\n|\t', ' ', text, flags=re.MULTILINE)
    return text

alay_mapping = dict(zip(df_alay['alay'], df_alay['cleaned'])) # Mapping for kamusalay
def alay(text): # Remove by replacing 'alay' words
    wordlist = text.split()
    text_alay = [alay_mapping.get(x,x) for x in wordlist]
    clean_alay = ' '.join(text_alay)
    return clean_alay

def stopwrds(text): # Stopwords fuction
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords]
    output_sw = ' '.join(tokens_without_sw)
    return output_sw

# Function for text cleansing
def cleaning(text):
    text = lowercase(text)
    text = special(text)
    text = single(text)
    text = singlestart(text)
    text = mulspace(text)
    text = rt(text)
    text = prefixedb(text)
    text = misc(text)
    text = alay(text)
    text = stopwrds(text)
    return text

### SKLEARN NEURAL NETWORK MODEL ANALYSIS SENTIMENT ###
# Load the SKLearn Model
f1 = joblib.load('sklearn/score.pkl')
clf = joblib.load('sklearn/model.pkl')
vectorizer = joblib.load('sklearn/vectorizer.pkl')

# Function for CSV SKLearn Analysis
def sentiment_csv_nn(input_file):
    column = input_file.iloc[:, 0]
    print(column)

    for data_file in column: # Define and execute query for insert cleaned text and sentiment to sqlite database
        data_clean = cleaning(data_file)
        sent = clf.predict(vectorizer.transform([data_clean]).toarray())
        query = "insert into data_file_sk (text,sentiment) values (?, ?)"
        val = (data_clean,str(sent))
        conn.execute(query, val)
        conn.commit()
        print(data_file)

# Create Homepage
@app.route('/', methods=['GET'])
def get():
    return "Welcome to DSental!"

# Endpoint for Text Analysis SKLearn
# Input text to analyze
@app.route('/text_sklearn', methods=['POST'])
def text_sentiment_sk():

    # Get text from user
    input_text = str(request.form['text'])

    # Cleaning text
    output_text = cleaning(input_text)

    # Model Prediction for Sentiment Analysis
    sent = clf.predict(vectorizer.transform([output_text]).toarray())

    # Define and execute query for insert cleaned text and sentiment to sqlite database
    query = "insert into data_text_sk (text,sentiment) values (?, ?)"
    val = (output_text,str(sent))
    conn.execute(query, val)
    conn.commit()

    # Define API response
    json_response = {
        'description': "Analysis Sentiment Success!",
        'F1 on test set': f1,
        'text' : output_text,
        'sentiment' : str(sent)
    }
    response_data = jsonify(json_response)
    return response_data

# Endpoint for File Analysis SKLearn
@app.route('/file_sklearn', methods=['POST'])
def file_sentiment_sk():

    # Get file
    file = request.files['file']
    try:
            datacsv = pd.read_csv(file, encoding='iso-8859-1')
    except:
            datacsv = pd.read_csv(file, encoding='utf-8')
    
    # Cleaning file
    sentiment_csv_nn(datacsv)

    # Define API response
    select_data = conn.execute("SELECT * FROM data_file_sk")
    conn.commit
    data = [
        dict(text_id=row[0], text=row[1], sentiment=row[2])
    for row in select_data.fetchall()
    ]
    
    return jsonify(data)

### TENSORFLOW LSTM MODEL ANALYSIS SENTIMENT ###
# Load the Tensorflow Model
model = load_model('tensorflow/model.h5')
tokenizer = joblib.load('tensorflow/tokenizer.pkl')

# Model Prediction
# Create Function for Sentiment Prediction
def predict_sentiment(text):
    sentiment_tf = ['negative', 'neutral', 'positive']
    text = cleaning(text)
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw, maxlen=200)
    prediction = model.predict(tw)
    polarity = np.argmax(prediction[0])
    return sentiment_tf[polarity]

# Function for CSV Tensorflow Analysis
def sentiment_csv_tf(input_file):
    column = input_file.iloc[:, 0]
    print(column)

    for data_file in column: # Define and execute query for insert cleaned text and sentiment to sqlite database
        data_clean = cleaning(data_file)
        sent = predict_sentiment(data_clean)
        query = "insert into data_file_tf (text,sentiment) values (?, ?)"
        val = (data_clean,sent)
        conn.execute(query, val)
        conn.commit()
        print(data_file)

# Endpoint for Text Analysis TensorFlow
# Input text to analyze
@app.route('/text_tensorflow', methods=['POST'])
def text_sentiment_tf():

    # Get text from user
    input_text = str(request.form['text'])

    # Cleaning text
    output_text = cleaning(input_text)

    # Model Prediction for Sentiment Analysis
    output_sent = predict_sentiment(output_text)

    # Define and execute query for insert cleaned text and sentiment to sqlite database
    query = "insert into data_text_tf (text,sentiment) values (?, ?)"
    val = (output_text,output_sent)
    conn.execute(query, val)
    conn.commit()

    # Define API response
    json_response = {
        'description': "Analysis Sentiment Success!",
        'text' : output_text,
        'sentiment' : output_sent
    }
    response_data = jsonify(json_response)
    return response_data

# Endpoint for File Analysis TensorFlow
@app.route('/file_tensorflow', methods=['POST'])
def file_sentiment_tf():

    # Get file
    file = request.files['file']
    try:
            datacsv = pd.read_csv(file, encoding='iso-8859-1')
    except:
            datacsv = pd.read_csv(file, encoding='utf-8')

    # Cleaning file
    sentiment_csv_tf(datacsv)

    # Define API response
    select_data = conn.execute("SELECT * FROM data_file_tf")
    conn.commit
    data = [
        dict(text_id=row[0], text=row[1], sentiment=row[2])
    for row in select_data.fetchall()
    ]
    
    return jsonify(data)

# Error Handling
@app.errorhandler(400)
def handle_400_error(_error):
    "Return a http 400 error to client"
    return make_response(jsonify({'error': 'Misunderstood'}), 400)


@app.errorhandler(401)
def handle_401_error(_error):
    "Return a http 401 error to client"
    return make_response(jsonify({'error': 'Unauthorised'}), 401)


@app.errorhandler(404)
def handle_404_error(_error):
    "Return a http 404 error to client"
    return make_response(jsonify({'error': 'Not found'}), 404)


@app.errorhandler(500)
def handle_500_error(_error):
    "Return a http 500 error to client"
    return make_response(jsonify({'error': 'Server error'}), 500)

# Run Server
if __name__ == '__main__':
   app.run(debug=True)

