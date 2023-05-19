import streamlit as st
import numpy as np
import easyocr


def parse(file):
    #TODO
    #parse image to np array 
    # return 
    pass

def apply_ocr(file):
    reader = easyocr.Reader(['en','pl'])
    text = reader.readtext(file, detail = 0)
    print(text)
    return text


def run():
    st.markdown("## Data Upload")

    uploaded_image = st.file_uploader("Choose an image", type=["tiff", "jpg"])

    if uploaded_image:
        st.markdown("Uploaded image:")
        st.image(uploaded_image, width=700)

        st.markdown("Ocr")
        st.write(str(apply_ocr(uploaded_image)))

        st.markdown("Predictions:")
    return parse(uploaded_image)
