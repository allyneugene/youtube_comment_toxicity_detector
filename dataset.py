import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer

class CommentDataset(Dataset):
    def __init__(self, csv_path, tokenizer_name="distilbert-base-uncased", max_length=128):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]["comment_text"])
        label = int(self.df.iloc[idx]["label"])
        return {"text": text, "label": label}

def collate_fn(batch, tokenizer, max_length=128):
    texts = [item["text"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    encodings["labels"] = labels
    return encodings

# Example usage:
# dataset = CommentDataset("data/cleaned_comments.csv")
# dataloader = DataLoader(
#     dataset,
#     batch_size=32,
#     collate_fn=lambda batch: collate_fn(batch, dataset.tokenizer, dataset.max_length)
# )