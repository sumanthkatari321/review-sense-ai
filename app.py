from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# A compact labelled dataset keeps the project runnable out of the box.
TRAINING_EXAMPLES = [
    ("Amazing quality and very fast delivery", "positive"),
    ("I love this product, it works perfectly", "positive"),
    ("Excellent service, helpful team and great value", "positive"),
    ("The item exceeded my expectations", "positive"),
    ("Beautiful design, I would definitely buy again", "positive"),
    ("Everything arrived on time and works well", "positive"),
    ("Terrible quality, it broke after one day", "negative"),
    ("Very disappointed with the purchase", "negative"),
    ("The delivery was late and customer service was rude", "negative"),
    ("Waste of money, I want a refund", "negative"),
    ("The product does not match the description", "negative"),
    ("Poor experience and frustrating support", "negative"),
]


TOKEN_PATTERN = re.compile(r"[a-z']+")
PROJECT_DIR = Path(__file__).parent


class NaiveBayesSentiment:
    """A tiny multinomial Naive Bayes classifier for demonstration purposes."""

    def __init__(self, examples: list[tuple[str, str]]):
        self.document_counts: Counter[str] = Counter()
        self.word_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.total_words: Counter[str] = Counter()
        self.vocabulary: set[str] = set()

        for text, label in examples:
            self.document_counts[label] += 1
            words = self.tokenize(text)
            self.word_counts[label].update(words)
            self.total_words[label] += len(words)
            self.vocabulary.update(words)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.lower())

    def predict(self, text: str) -> dict[str, float | str]:
        words = self.tokenize(text)
        total_documents = sum(self.document_counts.values())
        scores: dict[str, float] = {}
        vocabulary_size = len(self.vocabulary)

        for label, document_count in self.document_counts.items():
            score = math.log(document_count / total_documents)
            denominator = self.total_words[label] + vocabulary_size
            for word in words:
                score += math.log((self.word_counts[label][word] + 1) / denominator)
            scores[label] = score

        max_score = max(scores.values())
        normalizer = sum(math.exp(score - max_score) for score in scores.values())
        probabilities = {label: math.exp(score - max_score) / normalizer for label, score in scores.items()}
        sentiment = max(probabilities, key=probabilities.get)
        return {"sentiment": sentiment, "confidence": round(probabilities[sentiment], 2)}


model = NaiveBayesSentiment(TRAINING_EXAMPLES)


class ReviewSenseHandler(BaseHTTPRequestHandler):
    def send_content(self, content: bytes, content_type: str, status: int = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802 - inherited method name
        if self.path == "/":
            self.send_content((PROJECT_DIR / "templates" / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/static/style.css":
            self.send_content((PROJECT_DIR / "static" / "style.css").read_bytes(), "text/css; charset=utf-8")
        else:
            self.send_content(b"Not found", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - inherited method name
        if self.path != "/api/predict":
            self.send_content(b'{"error":"Not found"}', "application/json", HTTPStatus.NOT_FOUND)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length))
            text = str(payload.get("text", "")).strip()
        except (json.JSONDecodeError, ValueError):
            text = ""

        if not text:
            body = json.dumps({"error": "Please provide a non-empty 'text' field."}).encode()
            self.send_content(body, "application/json", HTTPStatus.BAD_REQUEST)
            return

        self.send_content(json.dumps(model.predict(text)).encode(), "application/json")

    def log_message(self, format: str, *args: object) -> None:
        """Keep the terminal output focused on the startup message."""


def run_server(port: int = 5000) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), ReviewSenseHandler)
    print(f"ReviewSense AI is running at http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
