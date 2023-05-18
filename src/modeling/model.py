import pandas as pd
class Model:
    """
    Ensemble method model class, which takes predictions of three independent models into consideration
    The models are already trained, so this class is for the inference purpose
    """
    def __init__(self, vision_model, text_model):
        self.vision_model = vision_model
        self.text_model = text_model
    
    def predict(self, obs: pd.Series):
        """
        We base our predictions mainly on textual data as it is a stanrd approach in document classification.
        If the textual model is not certain about its predictions (could be because of lack of the words, lack of 
        relevant textual infromation), the visual model is utilized.
        """
        if max(self.text_model.predict_proba(obs)) > 50:
            return self.text_model.predict(obs)
        else:
            return self.vision_model.predict(obs)
