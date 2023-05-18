import numpy as np
import pandas as pd
from autocorrect import Speller
from langdetect import detect
class Preprocessor:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
    
    def preprocess_text(self, text_col: str):
        self.df[text_col] = self.df[text_col].apply(lambda x: str(x))
        self.convert_text_to_lowercase(text_col)
        self.detect_language(text_col)
        self.autocorrect_words(text_col)
        pass

    def preprocess_image(self, image_col):
        #TODO
        pass
    

    def detect_language(self, text_col: str):
        def _detect_with_ignore(text: str):
            try:
                lang = detect(text)
            except:
                lang='en'
            return lang
        self.df['language'] = self.df[text_col].apply(lambda x: _detect_with_ignore(x))

    def convert_text_to_lowercase(self, text_col: str):
        self.df[text_col] = self.df[text_col].str.lower()
    
    def autocorrect_words(self, text_col: str):
        languages = set(self.df['language'])
        for lang in languages:
            try:
                spell = Speller(lang = lang, only_replacements=True)
            except:
                spell = Speller(lang = 'en', only_replacements=True)
            self.df[text_col] = np.where(self.df['language'] == lang, self.df[text_col].apply(lambda x: spell(x)), self.df[text_col])

        
    


