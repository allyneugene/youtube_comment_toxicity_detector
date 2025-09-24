# preprocess.py
import re
import _curses_panel as pd
import html
from typing import Optional
import os

URL_RE = re.compile(r'https?://\S+|www\.\S+')
MENTION_RE = re.compile(r'@\w+')
MULTI_WS = re.compile(r'\s+')
EMOJI_RE = re.compile("["
                      u"\U0001F600-\U0001F64F"  # emoticons
                      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                      u"\U0001F680-\U0001F6FF"  # transport & map symbols
                      u"\U0001F1E0-\U0001F1FF"  # flags
                      "]+", flags=re.UNICODE)


def clean_text(text: str, lowercase: bool = True, remove_emoji: bool = True) -> str:
    if pd.isna(text):
        return ""
    text = html.unescape(text)
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    if remove_emoji:
        text = EMOJI_RE.sub("", text)
    text = MULTI_WS.sub(" ", text).strip()
    if lowercase:
        text = text.lower()
    return text


def main(input_csv: str = "data/comments.csv", output_csv: str = "data/cleaned_comments.csv"):
    df = pd.read_csv(input_csv)
    if "comment_text" not in df.columns:
        raise ValueError("Expected a 'comment_text' column")
    df["comment_text"] = df["comment_text"].astype(str).apply(lambda x: clean_text(x))
    df = df[df["comment_text"].str.strip() != ""]
    df.to_csv(output_csv, index=False)
    print(f"Saved cleaned dataset to {output_csv}. Rows: {len(df)}")


if __name__ == "__main__":
    input_txt = os.path.abspath(os.path.join("venv", "comments.txt"))
    output_csv = os.path.abspath(os.path.join("data", "comments.csv"))

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Check if input file exists
    if not os.path.exists(input_txt):
        raise FileNotFoundError(f"Input file not found: {input_txt}")

    # Read the txt file (comma-separated, header included)
    df = pd.read_csv(input_txt)

    # Save as CSV
    df.to_csv(output_csv, index=False)
    print(f"Converted {input_txt} to {output_csv}")

    main(output_csv)
