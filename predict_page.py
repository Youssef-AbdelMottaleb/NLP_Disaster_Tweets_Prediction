
import streamlit as st
import pickle  
import numpy as np
import sklearn

def load_vectorizer():
    with open('vecotrizer.pkl', 'rb') as file:
        vec = pickle.load(file)
    return vec

def load_model():
    with open('Predict_model.pkl', 'rb') as file:
        LR = pickle.load(file)  
    return LR

vecotrizer = load_vectorizer()
Model=load_model()


def show_predict_page():
    st.title("""Tweets Disaster Prediction""")
    st.write("""We need some information to recognize the activity""")
    st.write("""So please fill this form""")

    tweet=st.text_input("tweet")
    st.write('The tweet is :', tweet)


    ok=st.button("Predict")
    if ok:
        X = vecotrizer.transform([tweet])

        # mlp_loaded = data["model"]
        Prediction=Model.predict(X)
        if Prediction[0]==0:
            st.subheader(f"Normal Tweet")
        elif Prediction[0]==1:
            st.subheader(f"Disaster Tweet")



        st.subheader(f"Prediction of the Tweet is {Prediction[0]}")