import numpy as np
import pandas as pd
from sklearn import model_selection, svm
from nltk.tokenize import word_tokenize
from sklearn.metrics import f1_score

class Model:
    """
    Ensemble method model class, which takes predictions of three independent models into consideration
    The models are already trained, so this class is for the inference purpose
    """
    def __init__(self, vision_model, text_model, tfidf_vectorizer, statistical_model, dataset):
        self.vision_model = vision_model
        self.text_model = text_model
        self.statistical_model = statistical_model
        self.tfidf_vectorizer = tfidf_vectorizer
        X = dataset.drop(axis=1)
    
    def text_model_predict(self):
        X_vect = self.tfidf_vectorizer.transform(self.X)
        predictions = self.text_model.predict(X_vect)
        predictions_proba = self.text_model.predict_proba(X_vect)
        return predictions, predictions_proba

    def predict_obs(self, obs: pd.Series, prob_threshold: float):
        """
        We base our predictions mainly on textual data as it is a stanrd approach in document classification.
        If the textual model is not certain about its predictions (could be because of lack of the words, lack of 
        relevant textual infromation), the visual model is utilized.
        """
        text_predictions, text_pred_proba = self.text_model_predict()

        if np.max(self.text_model.predict_proba(obs)) > prob_threshold:
            return self.text_model.predict(obs)
        else:
            return self.vision_model.predict(obs)
