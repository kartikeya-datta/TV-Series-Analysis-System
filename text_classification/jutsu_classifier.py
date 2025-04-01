import torch
import huggingface_hub
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          DataCollatorWithPadding,
                          TrainingArguments,
                          pipeline
)
import pandas as pd
from .cleaner import cleaner
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset
import gc
from .training_utils import get_class_weights, compute_metrics
from .custom_trainer import CustomTrainer

class jutsu_classifier:
    def __init__(self,
                 model_path,
                 data_path,
                 text_clolumn_name = "text",
                 label_column_name = "jutsu",
                 model_name = "distilbert/distilbert-base-uncased",
                 test_size = 0.2,
                 num_labels = 3,
                 huggingface_token = None,):
        self.model_path = model_path
        self.data_path = data_path
        self.text_column_name = text_clolumn_name
        self.label_column_name = label_column_name
        self.model_name = model_name
        self.test_size = test_size
        self.num_labels = num_labels
        self.huggingface_token = huggingface_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.huggingface_token is not None:
            # Login to Hugging Face Hub
            from huggingface_hub import login
            login(token=self.huggingface_token)
            
        self.tokenizer = self.load_tokenizer()
        
        if not huggingface_hub.repo_exists(self.model_path):
            #check if the datapath is provided
            if data_path is None:
                raise ValueError("data path is required to train the model, since the model path does not exist in huggingface hub.")
            
            train_data, test_data = self.load_data(self.data_path)
            train_data_df = train_data.to_pandas()
            test_data_df = test_data.to_pandas()
            
            all_data_df = pd.concat([train_data_df, test_data_df]).reset_index(drop=True)
            class_weights = get_class_weights(all_data_df)
            
            self.train_model(train_data, test_data, class_weights)
            
        self.model = self.load_model(self.model_path)
      
    def load_model(self, model_path):
        model = pipeline('text-classification',model = model_path, return_all_scores=True)     
        return model
    
            
    def train_model(self, train_data, test_data, class_weights):
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name,
                                                                   num_labels=self.num_labels,
                                                                   id2label=self.label_dict,
                                                                   )
        Data_Collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True)
        training_args = TrainingArguments(
            output_dir = self.model_path,
            learning_rate= 2e-4,
            per_device_train_batch_size= 8,
            per_device_test_batch_size= 8,
            num_train_epochs= 5,
            weight_decay= 0.01,
            evaluation_strategy="epoch",
            logginng_strategy="epoch",
            push_to_hub=True,
            )
        
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            tokenizer=self.tokenizer,
            data_collator=Data_Collator,
            compute_metrics=compute_metrics,
            
        )
        trainer.set_device(self.device)
        trainer.set_class_weights(class_weights)
        trainer.train()
            
        # Flush the cache
        del trainer
        del model
        gc.collect()
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
                     
    def preprocess_funtion(self,tokenizer, example):
        return tokenizer(example['text_cleaned'], truncation=True, padding='max_length', max_length=512, return_tensors="pt")

    def simplify_jutsus(self, jutsu):
        if "Genjutsu" in jutsu:
            return "Genjutsu"
        if "Ninjutsu" in jutsu:
            return "Ninjutsu"
        if "Taijutsu" in jutsu:
            return "Taijutsu"
    
    def load_data(self, data_path):
        df = pd.read_csv(data_path)
        df['jutsu_type_simplified'] = df['jutsu_type'].apply(self.simplify_jutsus)
        df['text'] = df['jutsu_name'] + ". " + df['jutsu_description']
        df['jutsu'] = df['jutsu_type_simplified']
        df = df[['text', 'jutsu']]
        df = df.dropna()
        
        #cleaning the text
        cleaner = cleaner()
        df['text_cleaned'] = df[self.text_column_name].apply(cleaner.clean_text)

        #Encode Labels
        le = preprocessing.LabelEncoder()
        le.fit(df[self.label_columdn_name].tolist())
        
        label_dict = {index:label_name for index, label_name in enumerate(le.__dict__['classes_'].tolist())}
        self.label_dict = label_dict
        df['label'] = le.transform(df[self.label_columdn_name].tolist())
        
        #Split the data into train and test
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        
        #Convert the data into huggingface datasets
        train_dataset = Dataset.from_pandas(df_train)
        test_dataset = Dataset.from_pandas(df_test)

        #Tokenize the datatset
        tokenizer_train = train_dataset.map(lambda examples: self.preprocess_funtion(self.tokenizer, examples), batched=True)
        tokenizer_test = test_dataset.map(lambda examples: self.preprocess_funtion(self.tokenizer, examples), batched=True)
    
        return tokenizer_train, tokenizer_test 
    
    def load_tokenizer(self):
        if huggingface_hub.repo_exists(self.model_path):
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return tokenizer
    
    def postprocess(self, model_output):
        output = []
        for pred in model_output:
            label = max(pred, key=lambda x: x['score'])['label']
            output.append(label)
        return output
    
    def classify_jutsu(self, text):
        model_output = self.model(text)
        predictions = self.postprocess(model_output)
        return predictions