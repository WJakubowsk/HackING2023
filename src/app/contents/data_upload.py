import streamlit as st
import numpy as np
import easyocr
from PIL import Image

def parse_image(file):
    img = Image.open(file)
    np_img = np.array(img)
    return np_img


def apply_ocr(parsed_file):
    reader = easyocr.Reader(['en','pl'])
    text = reader.readtext(parsed_file, detail = 0)
    print(text)
    return text


def run():
    st.markdown("## Data Upload")

    uploaded_image = st.file_uploader("Choose an image", type=["tiff", "jpg"])

    if uploaded_image:
        st.markdown("Uploaded image:")
        st.image(uploaded_image, width=700)

        st.markdown("Detected text:")
        st.write(str(apply_ocr((parse_image(uploaded_image)))))

        st.markdown("Predictions:")
    return parse_image(uploaded_image)
