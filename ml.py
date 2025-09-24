# app/ml.py
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

class ModelWrapper:
    def __init__(self, model, tokenizer, device='cpu', label_map=None):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.label_map = label_map or {0: "non-toxic", 1: "toxic"}
        self.model.eval()

    @classmethod
    def load(cls, model_path, tokenizer_dir=None, device='cpu', num_labels=2, base_model="bert-base-uncased"):
        """
        Load:
         - If model_path is a state_dict, create a AutoModelForSequenceClassification and load_state_dict.
         - If model_path is a pickled model object, try to load that directly (less portable).
        """
        # tokenizer
        if tokenizer_dir and os.path.isdir(tokenizer_dir):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        else:
            tokenizer = AutoTokenizer.from_pretrained(base_model)

        # load checkpoint
        try:
            ckpt = torch.load(model_path, map_location=device)
        except Exception as e:
            raise RuntimeError(f"Failed to read model file {model_path}: {e}")

        # Detect state_dict vs full model
        model = None
        if isinstance(ckpt, dict):
            # some save format: {'model_state_dict':...} or raw state_dict
            sd = ckpt.get("model_state_dict", ckpt)
            # create model with same architecture
            config = AutoConfig.from_pretrained(base_model, num_labels=num_labels)
            model = AutoModelForSequenceClassification.from_config(config)
            model.load_state_dict(sd)
        else:
            # assume saved full model (torch.save(model)) — less common but handle it
            model = ckpt

        return cls(model, tokenizer, device=device)

    def predict_batch(self, texts, batch_size=32, max_length=256):
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            enc = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc)
                logits = out.logits
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                preds = probs.argmax(axis=1)
                for pred, prob in zip(preds, probs):
                    results.append({
                        "label": self.label_map.get(int(pred), str(int(pred))),
                        "score": float(prob[int(pred)])
                    })
        return results
