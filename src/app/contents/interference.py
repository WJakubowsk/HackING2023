import pandas as pd
import pickle
from src.preprocessing.preprocessor import Preprocessor
from src.modeling.model import Model


def preprocess_text(text):
    df = pd.DataFrame({'Text': [text]})
    preprocessor = Preprocessor(df)
    preprocessor.preprocess_text('Text', lemmatize=True)
    df_preprocessed = preprocessor.df
    return df_preprocessed


def get_inference(obs: pd.Series):
    model = Model()