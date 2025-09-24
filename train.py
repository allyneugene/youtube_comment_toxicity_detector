import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from dataset import CommentDataset, collate_fn

def train(
    csv_path="data/cleaned_comments.csv",
    model_name="distilbert-base-uncased",
    num_labels=2,
    epochs=3,
    batch_size=32,
    lr=2e-5,
    max_length=128,
    device=None,
    save_path="model_state.pth"
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CommentDataset(csv_path, tokenizer_name=model_name, max_length=max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, dataset.tokenizer, max_length)
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=epochs * len(dataloader)
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}")

    # Save model state and metadata
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_name": model_name,
        "num_labels": num_labels,
        "max_length": max_length
    }, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()