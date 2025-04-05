import torch
from torch import nn
from transformers import Trainer 

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")

        # Ensure inputs are on the correct device
        device = self.device if hasattr(self, "device") else torch.device("cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward Pass
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to device
        outputs = model(**inputs)
        logits = outputs.get("logits").float().to(device)

        # Ensure class_weights tensor is on the correct device
        class_weights = torch.tensor(self.class_weights, dtype=torch.float32).to(device)

        # Compute Custom Loss
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1).to(device))
        
        return (loss, outputs) if return_outputs else loss

    def set_class_weights(self, class_weights):
        self.class_weights = class_weights
    
    def set_device(self, device):
        self.device = device