import numpy as np
import pandas as pd
from sklearn import model_selection, svm
from nltk.tokenize import word_tokenize
from sklearn.metrics import f1_score
from PIL import Image
import torch
from torchvision import transforms

class Model:
    """
    Ensemble method model class, which takes predictions of three independent models into consideration
    The models are already trained, so this class is for the inference purpose
    """
    def __init__(self, vision_model, text_model, tfidf_vectorizer, dataset, file):
        self.vision_model = vision_model
        self.text_model = text_model
        self.tfidf_vectorizer = tfidf_vectorizer
        self.file = file
        X = dataset
    
    def text_model_predict(self):
        X_vect = self.tfidf_vectorizer.transform(self.X)
        predictions = self.text_model.predict(X_vect)
        predictions_proba = self.text_model.predict_proba(X_vect)
        return predictions, predictions_proba
    
    def vision_model_predict(self):
        def to_device(data, device):
            """Move tensor(s) to chosen device"""
            if isinstance(data, (list,tuple)):
                return [to_device(x, device) for x in data]
            return data.to(device, non_blocking=True)
        transform = transforms.Compose([transforms.Resize((32, 32)),
                                transforms.Grayscale(),
                                transforms.ToTensor()])
        img = Image.open(self.file)
        img = transform(img)
        xb = to_device(img.unsqueeze(0), 'cpu')
        yb = self.vision_model(xb)
        _, preds  = torch.max(yb, dim=1)
        classes = ['advertisement', 'budget', 'email', 'file_folder', 'form', 'handwritten', 'invoice',
                    'letter', 'memo', 'news_article', 'pit37_v1', 'pozwolenie_uzytkowanie_obiektu_budowlanego',
                    'presentation', 'questionnaire', 'resume', 'scientific_publication', 'scientific_report',
                    'specification', 'umowa_na_odleglosc_odstapienie', 'umowa_o_dzielo', 'umowa_sprzedazy_samochodu']
        return classes[preds[0].item()]

    def predict_obs(self, prob_threshold: float):
        """
        We base our predictions mainly on textual data as it is a stanrd approach in document classification.
        If the textual model is not certain about its predictions (could be because of lack of the words, lack of 
        relevant textual infromation), the visual model is utilized.
        """
        text_prediction, text_pred_proba = self.text_model_predict()

        if np.max(self.text_model.predict_proba(obs)) > prob_threshold:
            return text_prediction
        else:
            visual_prediction = self.vision_model_predict()
            return visual_prediction

