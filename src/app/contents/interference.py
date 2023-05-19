import pandas as pd

from src.preprocessing.preprocessor import Preprocessor


def preprocess_text(text):
    df = pd.DataFrame({'Text': [text]})
    preprocessor = Preprocessor(df)
    preprocessor.preprocess_text('Text', lemmatize=True)
    df_preprocessed = preprocessor.df
    return df_preprocessed
