import torch
from transformers import pipeline

class ThemeClassifier():
    def __init__(self, theme_list):
        self.model_name = "facebook/bart-large-mnli"
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.theme_list = theme_list
        self.theme_classifier = self.load_model(self.device)
    
    def load_model(self, device):
        theme_classifier = pipeline(
            "zero-shot-classification",
            model= self.model_name,
            device = device,
            framework="pt"
        )
        return theme_classifier