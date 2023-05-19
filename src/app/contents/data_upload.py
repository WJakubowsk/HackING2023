import streamlit as st
import numpy as np
<<<<<<< HEAD
import easyocr
=======
from PIL import Image

>>>>>>> 8d13cb4efb241f39e801818b6467b39573a3e789


def parse(file):
    
    #parse image to np array  
    img = Image.open(file)
    np_img = np.array(img)

    return np_img


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
