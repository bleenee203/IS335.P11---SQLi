import streamlit as st
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load the pre-trained model and vectorizer
loaded_model = joblib.load('xgb_model.pkl')
vectorizer = joblib.load('bigram_vectorizer.pkl')

# Define stopwords and lemmatizer
stopwords_set= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
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
lemmatizer = WordNetLemmatizer()

# Prediction function
def prediction(query):
    # Preprocess input query
    query = re.sub('[^A-Za-z0-9]+', ' ', query)
    tokens = word_tokenize(query.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords_set]
    processed_query = ' '.join(tokens)

    # Vectorize the input using the saved bigram vectorizer
    query_vectorized = vectorizer.transform([processed_query])

    # Predict using the XGBoost model
    prediction = loaded_model.predict(query_vectorized)

    # Return the result
    if prediction[0] == 1:
        return "SQL Injection Attack is detected"
    else:
        return "No SQL Injection detected"

# Streamlit app
def main():
    st.title("SQL Injection Attack Detection")
    query = st.text_input("Enter a SQL query to check")
    result = ""
    if st.button("Predict"):
        result = prediction(query)
        st.success(result)

if __name__ == "__main__":
    main()
