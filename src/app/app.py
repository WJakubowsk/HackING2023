import streamlit as st
from contents import data_upload


def main():
    st.title("Document classification demo")

    data_upload.run()




if __name__ == "__main__":
    main()