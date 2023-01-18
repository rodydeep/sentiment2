import pandas as pd
import streamlit as st
import cleantext
import pickle
import nltk
import string
import re
nltk.download("punkt")
nltk.download("stopwords")
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")



#pembuatan header

st.header('Sentiment Analysis')

st.sidebar.header('Data Input')

# input data
uploaded_file = st.sidebar.file_uploader("Upload data", type=["csv","xlsx"])
if uploaded_file is not None:
    inputan = pd.read_excel(uploaded_file)
else:
    def input_user():
        tweet = st.sidebar('tweet')
        data = {'tweet': tweet,}

        features = pd.DataFrame(data, index=[0])
        return features
   


# Displays the user input features
st.subheader('Data Input')

if uploaded_file is not None:
    st.write(inputan[['Author','Content']])
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(inputan)

    
#preprocessing awal

def casefolding(text):
  text = text.lower()         #merubah kalimat menjadi huruf kecil
  return text

def cleaning(text):
    text = text.replace('\\t',"").replace('\\n',"").replace('\\u',"").replace('\\',"")

    text = text.encode('ascii', 'replace').decode('ascii')

    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\s+)"," ",text).split())

    text = re.sub(r'http\S+', '',text)
    text = re.sub(r"[.,:;+!\-_<^/=?\"'\(\)\d\*]", " ",text)
    text = re.sub(r'http\S+', '',text)
    text = text.translate(str.maketrans(" ", " ", string.punctuation))
    text = text.strip()
    text = re.sub(r"\b[a-z A-Z]\b", " ", text)
    text = re.sub("/s+", " ", text)
    text = re.sub(r"\b[a-z A-Z]\b", " ", text)
    
    return text

def tokenize(text):
  return word_tokenize(text)

def freq(text):
  return FreqDist(text)

def preprocess_data(text):
    text = casefolding (text)
    text = cleaning (text)
    text = tokenize (text)

   

    return text


if st.button("Preprocessing"):
  st.subheader('Hasil Preprocessing Data')
  inputan['Data Bersih'] = inputan['Content'].apply(preprocess_data)
  st.write(inputan[['Author','Data Bersih',]])

#preprocessing akhir    



inputan['vector'] = inputan['Content'].astype(str)

vec = CountVectorizer().fit(inputan['vector'])
vec_transform = vec.transform(inputan['vector'])
print(vec_transform)
x_test = vec_transform.toarray()


# membaca model 
load_model = pickle.load(open('model_bayes_.pkl', 'rb'))

inputan['sentimen'] = load_model.predict(x_test)
# Apply model to make predictions
if st.button("Klasifikasi"):
  st.subheader('Hasil klasifikasi')
  inputan['sentimen'] = load_model.predict(x_test)
  st.write(inputan[['Author','Content','sentimen']])
  st.write(inputan['sentimen'].value_counts())
  

def convert_inputan(inputan):
# IMPORTANT: Cache the conversion to prevent computation on every rerun
    return inputan.to_csv().encode('utf-8')

csv = convert_inputan(inputan)
st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='sentiment.csv',
    mime='text/csv',
        )


st.set_option('deprecation.showPyplotGlobalUse', False)
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

if st.button("visualisasi"):
  st.subheader('Hasil Visualisasi')
  data = inputan['sentimen'].value_counts()

  plt.figure(figsize=(10, 6))
  plt.bar(['Positive', 'Netral', 'Negative'], data, color=['royalblue','green', 'orange'])
  plt.xlabel('Jenis Sentimen', size=14)
  plt.ylabel('Frekuensi', size=14)

  st.pyplot()
