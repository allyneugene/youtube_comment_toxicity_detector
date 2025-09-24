import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd  # Fix: use pandas, not _curses_panel
from fastapi import FastAPI, Request
from predict import load_model, predict

app = FastAPI()

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location="cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint["model_name"],
        num_labels=checkpoint["num_labels"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    tokenizer = AutoTokenizer.from_pretrained(checkpoint["model_name"])
    max_length = checkpoint.get("max_length", 128)
    return model, tokenizer, max_length

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, max_length
    model_path = "model_state.pth"
    model, tokenizer, max_length = load_model(model_path)

@app.post("/predict")
async def predict_comment(request: Request):
    data = await request.json()
    comments = data.get("comments", [])
    preds = predict(comments, model, tokenizer, max_length)
    return {"predictions": preds.tolist()}

def predict(comments, model, tokenizer, max_length=128, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    inputs = tokenizer(
        comments,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
    return preds

if __name__ == "__main__":
    # Example usage
    model_path = "model_state.pth"
    model, tokenizer, max_length = load_model(model_path)

    # Predict on new comments
    new_comments = [
        "I love this video!",
        "You are so dumb.",
        "Great job!",
        "Nobody likes you."
    ]
    predictions = predict(new_comments, model, tokenizer, max_length)
    for comment, label in zip(new_comments, predictions):
        print(f"Comment: {comment}\nPredicted label: {label}\n")