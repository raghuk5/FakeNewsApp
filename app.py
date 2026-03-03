import streamlit as st
import pickle
import re


model = pickle.load(open("model/fake_news_model(2).pkl","rb"))
vectorizer = pickle.load(open("model/tfidf_vectorizer(2).pkl","rb"))

# Text Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+','',text)
    text = re.sub(r'\W','',text)
    text = re.sub(r'\s+','',text)
    return text

# App title
st.title("Fake News Detection App")
st.write("Enter news title and content to check whether it is Fake or Real.")

# User input
title = st.text_input("Enter News Title")
content = st.text_area("Enter News Content")

if st.button("Predict"):
    if title and content:
        combined = title + " " + content
        cleaned = clean_text(combined)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)

        if prediction[0] == 1:
            st.success("This is Real News")
        else:
            st.error("This is Fake News")
    else:
        st.warning("Please enter both title and content.")            