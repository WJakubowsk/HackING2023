import streamlit as st
import numpy as np
from PIL import Image



def parse(file):
    
    #parse image to np array  
    img = Image.open(file)
    np_img = np.array(img)

    return np_img



def run():
    st.markdown("## Data Upload")

    uploaded_image = st.file_uploader("Choose an image", type=["tiff", "jpg"])

    if uploaded_image:
        st.markdown("Uploaded image:")
        st.image(uploaded_image, width=700)

        st.markdown("Predictions:")
    return parse(uploaded_image)
