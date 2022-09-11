import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

header = st.container()
features = st.container()

imodel = "model_a.sav"
ivec = "vector.sav"

model= pickle.load(open(imodel, 'rb'))
vec= pickle.load(open(ivec, 'rb'))



with header:
    st.title('Fake News Detection')
    st.text('I can help you decide if the news you just read is fake or not')

with features:
    txt = st.text_area('Enter your content')
    data = vec.transform([txt])
    result = model.predict(data).tolist()[0]
    if result ==1:
        result = 'Real'
    else:
        result='Fake'
    st.write('Result:', result)

