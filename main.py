# app/main.py
import os
import sqlite3
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from youtube_api import fetch_comments, API_KEY, extract_video_id
from ml import ModelWrapper
from youtube_api import fetch_comments
from fastapi import Form


app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET", "dev-secret"))

BASE_DIR = os.path.dirname(__file__)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# config via env
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "models", "toxicity_model.pth"))
TOKENIZER_DIR = os.getenv("TOKENIZER_DIR", os.path.join(BASE_DIR, "models", "tokenizer"))
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

# global wrapper (loaded at startup)
model_wrapper: ModelWrapper = None


from urllib.parse import urlparse, parse_qs

def extract_video_id(url: str):
    """Extracts video ID from a YouTube URL."""
    parsed = urlparse(url)
    if parsed.hostname in ["www.youtube.com", "youtube.com"]:
        return parse_qs(parsed.query).get("v", [None])[0]
    elif parsed.hostname in ["youtu.be"]:
        return parsed.path[1:]
    return None

@app.on_event("startup")
def startup():
    global model_wrapper
    if not os.path.isfile(MODEL_PATH):
        print(f"Warning: model file not found at {MODEL_PATH}. Start without model or place it there.")
        model_wrapper = None
    else:
        model_wrapper = ModelWrapper.load(MODEL_PATH, tokenizer_dir=TOKENIZER_DIR, device="cpu")
    _init_db()

def _init_db():
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    conn = sqlite3.connect(os.path.join(BASE_DIR, "data", "db.sqlite3"))
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT,
            text TEXT,
            label TEXT,
            score REAL,
            moderated INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()

@app.post("/predict")
async def predict(request: Request):
    if model_wrapper is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    payload = await request.json()
    comments = payload.get("comments")
    if not comments or not isinstance(comments, list):
        raise HTTPException(status_code=400, detail="Provide JSON with 'comments': [..]")
    preds = model_wrapper.predict_batch(comments)
    # return list of {label, score} aligned with comments
    return JSONResponse([{"text": c, **p} for c, p in zip(comments, preds)])

@app.post("/fetch_comments")
def fetch_and_predict(video_link: str = Form(...), max_results: int = Form(100)):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Set YOUTUBE_API_KEY env var")

    video_id = extract_video_id(video_link)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube link")

    comments = fetch_comments(video_id, API_KEY, max_results=max_results)

    if model_wrapper is None:
        return JSONResponse([{"text": c} for c in comments])

    preds = model_wrapper.predict_batch(comments)

    # Save to DB
    conn = sqlite3.connect(os.path.join(BASE_DIR, "data", "db.sqlite3"))
    cur = conn.cursor()
    merged = []
    for c, p in zip(comments, preds):
        cur.execute(
            "INSERT INTO comments (video_id, text, label, score) VALUES (?,?,?,?)",
            (video_id, c, p["label"], p["score"])
        )
        merged.append({"text": c, **p})
    conn.commit()
    conn.close()

    return JSONResponse(merged)

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, video_id: str = None):
    comments = []
    if video_id:
        conn = sqlite3.connect(os.path.join(BASE_DIR, "data", "db.sqlite3"))
        cur = conn.cursor()
        cur.execute("SELECT id, text, label, score, moderated FROM comments WHERE video_id=? ORDER BY id DESC", (video_id,))
        rows = cur.fetchall()
        conn.close()
        for r in rows:
            comments.append({"id": r[0], "text": r[1], "label": r[2], "score": r[3], "moderated": bool(r[4])})
    return templates.TemplateResponse("dashboard.html", {"request": request, "video_id": video_id, "comments": comments})

@app.post("/moderate")
async def moderate(id: int = Form(...), action: str = Form(...)):
    conn = sqlite3.connect(os.path.join(BASE_DIR, "data", "db.sqlite3"))
    cur = conn.cursor()
    if action == "delete":
        cur.execute("DELETE FROM comments WHERE id=?", (id,))
    elif action == "approve":
        cur.execute("UPDATE comments SET moderated=1 WHERE id=?", (id,))
    conn.commit()
    conn.close()
    return RedirectResponse("/", status_code=303)
