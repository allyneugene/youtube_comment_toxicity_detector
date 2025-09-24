# app/youtube_api.py
import requests
import os
from dotenv import load_dotenv
import requests
from urllib.parse import urlparse, parse_qs

load_dotenv()  # loads variables from .env

API_KEY = os.getenv("YOUTUBE_API_KEY")  # fetch the API key


YOUTUBE_COMMENTTHREADS = "https://www.googleapis.com/youtube/v3/commentThreads"

def extract_video_id(url: str) -> str | None:
    """
    Extracts the video ID from a full YouTube URL.
    Returns None if invalid.
    """
    parsed_url = urlparse(url)
    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        return parse_qs(parsed_url.query).get("v", [None])[0]
    elif parsed_url.hostname == "youtu.be":  # short link format
        return parsed_url.path[1:]
    return None

def fetch_comments(video_id: str, API_KEY: str, max_results: int = 200):
    """
    Fetch top-level comments for a video. Returns a list of comment strings.
    """
    comments = []
    params = {
        "part": "snippet",
        "videoId": video_id,
        "key": API_KEY,
        "maxResults": 100,
        "textFormat": "plainText"
    }
    nextPageToken = None
    while True:
        if nextPageToken:
            params["pageToken"] = nextPageToken
        r = requests.get(YOUTUBE_COMMENTTHREADS, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        for item in data.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            comments.append(top.get("textDisplay", ""))

            if len(comments) >= max_results:
                return comments[:max_results]
        nextPageToken = data.get("nextPageToken")
        if not nextPageToken:
            break
    return comments
