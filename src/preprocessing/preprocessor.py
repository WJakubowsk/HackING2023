class Preprocessor:
    def __init__(self, df) -> None:
        self.df = df
    
    def preprocess_text(self, text_col):
        #TODO
        pass

    def preprocess_image(self, image_col):
        #TODO
        pass
    
    def _convert_text_to_lowercase(self, text_col):
        self.df[text_col] = self.df[text_col].str.lower()

