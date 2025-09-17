from flask import Flask, render_template, request, jsonify, session
import os
import requests
from google.oauth2 import service_account
from google.auth.transport.requests import Request

# ----------------------------
# Configuration
# ----------------------------
PROJECT_ID = os.getenv("PROJECT_ID", "vocal-marking-471109-k9")
LOCATION = os.getenv("LOCATION", "us-central1")
# Use the model you prefer; the user of this project likes gemini-2.5-pro
MODEL = os.getenv("MODEL", "gemini-2.5-flash")
# Path to your service account key file. DO NOT store it under static/ or serve it.
SERVICE_ACCOUNT_FILE = "service_account.json"


# Flask app
app = Flask(__name__)
# Set a secure secret key in production
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")

# System prompt / instructions for the chatbot
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful, concise assistant. Answer clearly and avoid unnecessary jargon.")

def _get_credentials():
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        raise FileNotFoundError(f"Service account file not found: {SERVICE_ACCOUNT_FILE}")
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    creds.refresh(Request())
    return creds

def _vertex_url():
    # Vertex AI Generative endpoint (REST) for text
    return f"https://{LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL}:generateContent"

def _generate_reply(contents):
    creds = _get_credentials()
    token = creds.token

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # You can tune generation here
    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.6,
            "maxOutputTokens": 1024,
        },
        # Add a safetySettings block if you want stricter defaults
    }

    resp = requests.post(_vertex_url(), headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # Extract the first candidate's first text part (Vertex AI response shape)
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        # Fallback: return raw JSON so user can see issues
        return str(data)

def _history_to_contents(history, user_msg):
    """
    Convert Flask session history into Vertex 'contents' format.
    history: list of dicts {"role": "user"|"model", "text": "..."}
    user_msg: the latest user message (string)
    """
    contents = []
    # Start with a system message as a user role "system" isn't standard; so we embed as a user meta prefix
    contents.append({
        "role": "user",
        "parts": [{"text": f"[SYSTEM INSTRUCTIONS]\n{SYSTEM_PROMPT}"}]
    })
    for turn in history:
        role = "user" if turn["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": turn["text"]}]})
    # Latest user message
    contents.append({"role": "user", "parts": [{"text": user_msg}]})
    return contents

@app.route("/", methods=["GET"])
def home():
    # Ensure history exists
    if "history" not in session:
        session["history"] = []
    return render_template("chat.html", history=session["history"])

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_msg = (data.get("message") or "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    # Initialize history
    history = session.get("history", [])

    # Build contents and call Vertex AI
    contents = _history_to_contents(history, user_msg)
    reply = _generate_reply(contents)

    # Update history (limit to last N turns to keep payload small)
    history.append({"role": "user", "text": user_msg})
    history.append({"role": "model", "text": reply})
    session["history"] = history[-20:]

    return jsonify({"reply": reply})

@app.route("/reset", methods=["POST"])
def reset():
    session["history"] = []
    return jsonify({"ok": True})

if __name__ == "__main__":
    # For local testing only. In prod, use gunicorn/uvicorn, etc.
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)