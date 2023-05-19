import streamlit as st
import numpy as np
import easyocr
from PIL import Image
import pandas as pd
import pickle
import torch
from src.preprocessing.preprocessor import Preprocessor
from src.modeling.model import Model


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

    tfidf_vectorizer = pickle.load('pretrained/tfidf_vectorizer.pkl')
    text_model = pickle.load('pretrained/svm_model.pkl')
    vision_model = torch.load('pretrained/cnn_model.pth')    

    uploaded_image = st.file_uploader("Choose an image", type=["tiff", "jpg"])

    if uploaded_image:
        st.markdown("Uploaded image:")
        st.image(uploaded_image, width=700)

        st.markdown("Detected text:")
        ocr_string = apply_ocr((parse_image(uploaded_image)))
        st.write(ocr_string)

        st.markdown("Preprocessed text:")
        preprocessed_df = preprocess_text(ocr_string)

        st.markdown("Predictions:")
        model = Model(vision_model, text_model, tfidf_vectorizer, preprocessed_df, uploaded_image)
        st.write(model.predict_obs(0.6))

