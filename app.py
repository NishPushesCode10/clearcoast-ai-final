"""
ClearCoast AI — Flask Application
===================================
AI-Powered Cloud Removal & Hallucination for Coastal Monitoring.
"""

import os, io, logging, json, base64
from flask import Flask, render_template, request, jsonify
from model import process_image




# Optional: Google Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB upload limit

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_IMAGE_PATH = os.path.join(app.static_folder or "static",
                                 "sample_cloudy.tif")

# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------
_gemini_model = None


def _get_gemini():
    global _gemini_model
    if _gemini_model is None and GEMINI_AVAILABLE:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if api_key:
            genai.configure(api_key=api_key)
            _gemini_model = genai.GenerativeModel("gemini-2.0-flash")
            logger.info("Gemini model initialised.")
        else:
            logger.warning("GEMINI_API_KEY not set — GenAI features disabled.")
    return _gemini_model


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html",
                           gemini_available=GEMINI_AVAILABLE and bool(
                               os.environ.get("GEMINI_API_KEY")))


@app.route("/process", methods=["POST"])
def process():
    """Accept an image upload (or 'use_sample') and return processed results."""
    try:
        use_sample = request.form.get("use_sample", "false") == "true"

        if use_sample:
            if not os.path.isfile(SAMPLE_IMAGE_PATH):
                return jsonify({"error": "Sample image not found on server."}), 404
            with open(SAMPLE_IMAGE_PATH, "rb") as f:
                image_bytes = f.read()
        else:
            file = request.files.get("image")
            if file is None or file.filename == "":
                return jsonify({"error": "No image file provided."}), 400
            image_bytes = file.read()

        results = process_image(image_bytes)
        return jsonify(results)

    except Exception as exc:
        logger.exception("Processing failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/gemini/report", methods=["POST"])
def gemini_report():
    """Generate a Gemini-powered coastal monitoring report."""
    model = _get_gemini()
    if model is None:
        return jsonify({"error": "Gemini API is not configured."}), 503

    data = request.get_json(silent=True) or {}
    cloud_pct = data.get("cloud_pct", "N/A")
    alerts = data.get("alerts", [])

    prompt = (
        "You are ClearCoast AI, an expert coastal monitoring assistant.\n"
        "An AI cloud-removal pipeline just processed a satellite image.\n\n"
        f"Cloud cover detected: {cloud_pct}%\n"
        f"Alerts: {json.dumps(alerts)}\n\n"
        "Write a concise professional coastal-monitoring report (≈200 words) "
        "covering: image quality assessment, potential environmental observations, "
        "and recommended next steps. Use markdown formatting."
    )
    try:
        resp = model.generate_content(prompt)
        return jsonify({"report": resp.text})
    except Exception as exc:
        logger.exception("Gemini report failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/gemini/chat", methods=["POST"])
def gemini_chat():
    """Answer a free-form user question using Gemini."""
    model = _get_gemini()
    if model is None:
        return jsonify({"error": "Gemini API is not configured."}), 503

    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    cloud_pct = data.get("cloud_pct", "N/A")

    if not question:
        return jsonify({"error": "No question provided."}), 400

    prompt = (
        "You are ClearCoast AI, a helpful coastal monitoring assistant.\n"
        f"Context — cloud cover in the current image: {cloud_pct}%.\n\n"
        f"User question: {question}\n\n"
        "Provide a clear, concise answer using markdown."
    )
    try:
        resp = model.generate_content(prompt)
        return jsonify({"answer": resp.text})
    except Exception as exc:
        logger.exception("Gemini chat failed")
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
