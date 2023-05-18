import numpy as np
import pandas as pd
from autocorrect import Speller
from langdetect import detect
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class Preprocessor:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
    
    def preprocess_text(self, text_col, language='eng'):
        self.df[text_col] = self.df[text_col].apply(lambda x: str(x))
        self.detect_language(text_col)
        self.convert_text_to_lowercase(text_col)
        self.remove_stopwords_column(text_col, language)
        self.lemmatize_column(text_col, language)
        self.autocorrect_words(text_col)

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

    def remove_stopwords_column(self, text_col, language='eng'):
        self.df[text_col] = self.df[text_col].apply(
            lambda text: self.remove_stopwords(text, language=language)
        )

    def remove_stopwords(self, text, language='eng'):
        if language == 'eng':
            stop_words = set(stopwords.words('english'))
        elif language == 'pl':
            with open('../../data/stop_words_polish.txt', 'r', encoding='utf-8') as file:
                content = file.readlines()
                content = [line.strip() for line in content]
                stop_words = set(content)

        word_tokens = word_tokenize(text)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        return ' '.join(filtered_sentence)

    def lemmatize_column(self, text_col, language='eng'):
        self.df[text_col] = self.df[text_col].apply(
            lambda text: self.lemmatize_text(text, language=language)
        )

    def lemmatize_text(self, text, language='eng'):
        if language == "pl":
            nlp = spacy.load("pl_core_news_sm")
        elif language == "eng":
            nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        lemmas = [token.lemma_ for token in doc]
        return " ".join(lemmas)
