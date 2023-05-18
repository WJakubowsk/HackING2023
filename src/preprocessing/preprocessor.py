import numpy as np
import pandas as pd
from autocorrect import Speller
class Preprocessor:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
    
    def preprocess_text(self, text_col: str):
        self._convert_text_to_lowercase(text_col)
        pass

    def preprocess_image(self, image_col):
        #TODO
        pass
    
    def _convert_text_to_lowercase(self, text_col: str):
        self.df[text_col] = self.df[text_col].str.lower()
    
    def _autocorrect_words(self, text_col: str, language: str = 'en'):
        spell = Speller(language=language, only_replacements=True) # for OCR detected text it should be better
        self.df[text_col] = self.df[text_col].apply(lambda x: spell(str(x)))
    


