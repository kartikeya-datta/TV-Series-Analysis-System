import spacy
from nltk.tokenize import sent_tokenize
import os 
import sys
import pathlib
import pandas as pd
from ast import literal_eval as eval
import numpy as np
folder_path = pathlib.Path().parent.resolve()
sys.path.append(os.path.join(folder_path, '../'))
from utils import load_subtitles_dataset


class named_entity_recognizer:
    def __init__(self):
        self.nlp_model = self.load_model()
        pass
        
    def load_model(self):
        nlp = spacy.load("en_core_web_trf")
        return nlp
    
    def get_ners_inference(self,script):
        script_sentences = sent_tokenize(script)
        
        ner_output = []
        for sentence in script_sentences:
            doc = self.nlp_model(sentence)
            ners = set()
            for entity in doc.ents:
                if entity.label_ == 'PERSON':
                    full_name = entity.text
                    first_name = full_name.split(" ")[0]
                    first_name = first_name.strip()
                    ners.add(first_name)
            ner_output.append(ners)
            
        return ner_output
    
    def get_ners(self,data_set_path,save_path=None):
        if save_path is not None and os.path.exists(save_path):
            df = pd.read_csv(save_path)
            df['ners'] = df['ners'].apply(lambda x: eval(x) if isinstance(x, str) else x)
            return df
            
        
        #Loading the dataset
        df = load_subtitles_dataset(data_set_path)
        
        #Run the inference
        df['ners'] = df['script'].apply(lambda x: self.get_ners_inference(x))
        
        if save_path is not None:
            df.to_csv(save_path, index=False)
            
        return df