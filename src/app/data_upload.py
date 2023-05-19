import streamlit as st
import numpy as np
import easyocr
from PIL import Image
import pandas as pd
import pickle
import torch
import sys
sys.path.append('..')
from preprocessing.preprocessor import Preprocessor
from modeling.model import Model


def parse_image(file):
    img = Image.open(file)
    np_img = np.array(img)
    return np_img


def apply_ocr(parsed_file):
    reader = easyocr.Reader(['en','pl'])
    text = reader.readtext(parsed_file, detail = 0)
    print(text)
    return text

def preprocess_text(text: str) -> pd.DataFrame:
    df = pd.DataFrame({'Text': [text]})
    preprocessor = Preprocessor(df)
    preprocessor.preprocess_text('Text', lemmatize=True)
    df_preprocessed = preprocessor.df
    return df_preprocessed


def get_inference(obs: pd.Series, model: Model):
    model.predict_obs(0.6)

def run():
    st.markdown("## Data Upload")
    with open('pretrained/tfidf_vectorizer.pkl', "rb") as file:
        tfidf_vectorizer = pickle.load(file)
    with open('pretrained/svm_model.pkl', "rb") as file:
        text_model = pickle.load(file)
    
    # vision_model = torch.load('pretrained/cnn_model.pth')    

    uploaded_image = st.file_uploader("Choose an image", type=["tiff", "jpg"])

    if uploaded_image:
        st.markdown("Uploaded image:")
        st.image(uploaded_image, width=700)

        st.markdown("Detected text:")
        ocr_string = apply_ocr((parse_image(uploaded_image)))
        st.write(ocr_string)

        st.markdown("Preprocessed text:")
        preprocessed_df = preprocess_text(ocr_string)
        st.write(preprocessed_df['Text'])

        st.markdown("Predictions:")
        model = Model(None, text_model, tfidf_vectorizer, preprocessed_df, uploaded_image)
        predict, proba = model.predict_obs(0.6) 
        st.write(predict)

