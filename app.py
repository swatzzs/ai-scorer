# app.py

from flask import Flask, request, jsonify, render_template
from scoring import score_transcript

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/score", methods=["POST"])
def api_score():
    data = request.get_json() or {}
    transcript = data.get("transcript", "") or ""
    duration_seconds = data.get("duration_seconds")

    try:
        if duration_seconds is not None:
            duration_seconds = float(duration_seconds)
    except ValueError:
        duration_seconds = None

    result = score_transcript(transcript, duration_seconds=duration_seconds)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
