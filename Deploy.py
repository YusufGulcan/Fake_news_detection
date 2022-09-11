import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import re
header = st.container()
features = st.container()

imodel = "model_a.sav"
ivec = "vector.sav"

model= pickle.load(open(imodel, 'rb'))
vec= pickle.load(open(ivec, 'rb'))

def clean_m(x):
    x = x.strip().lower()
    x = re.sub('\(.+\)','',x)
    x = re.sub('\\n', '', x)
    x = re.sub('[^1-9a-zA-Z ]', '', x)
    return x


with header:
    st.title('Fake News Detection')
    st.text('I can help you decide if the news you just read is fake or not')

with features:
    txt = st.text_area('Enter your content')
    txt= clean_m(txt)
    data = vec.transform([txt])
    result = model.predict(data).tolist()[0]
    if result ==1:
        result = 'Real'
    else:
        result='Fake'
    st.write('Result:', result)

