
import streamlit as st
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import joblib
from sklearn.metrics import  roc_auc_score
import re
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
import string
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve,auc
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import word2vec
from gensim.models import KeyedVectors
import pickle

from tqdm import tqdm
from nltk.corpus import stopwords
import numpy as np
import tensorflow as tf
import tensorflow 
import nltk
nltk.download('stopwords')
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])
#loading the trained model
loaded_model = tf.keras.models.load_model('lstm.h5')

#defining the function which will make the prediction using the data which the user inputs 
def prediction(sentance):
  from tqdm import tqdm
  preprocessed_query = []
  lemmatizer = WordNetLemmatizer()
  sentance = re.sub('[^A-Za-z0-9]+', ' ', sentance)
  sentance = re.sub(r',', ' ', sentance)
  tokenization = nltk.word_tokenize(sentance)
  sentance = ' '.join([lemmatizer.lemmatize(w) for w in tokenization])
  sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)
  preprocessed_query.append(sentance.strip())
  print(preprocessed_query)
  max_length = max([len(s.split()) for s in preprocessed_query])
  print("ans:",max_length)
  tokenizer_obj = Tokenizer()
  tokenizer_obj.fit_on_texts(preprocessed_query)
  sequences = tokenizer_obj.texts_to_sequences(preprocessed_query)

  word_index = tokenizer_obj.word_index
  print('found %s unique tokens.'%len(word_index))

  query_pad = pad_sequences(sequences,maxlen=522)
  print("shape of query_pad",query_pad.shape)
  

  predictions = loaded_model.predict([[query_pad]])
  if predictions>=0.99:

    pred = 'Sql Injection Attack is there'

  else:
    pred = "No Sql injection is there"
  return pred

def main():       
    st.title("SQL Injection Attack Detection")  
    query = st.text_input('Enter the query')

    result = ""

    if st.button("Predict"):
      result = prediction(query)
      st.success(result)
if __name__=='__main__': 
  main()