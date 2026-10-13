# ReviewSense AI

ReviewSense AI is a compact, beginner-friendly web app that classifies customer reviews as **positive** or **negative**. It uses a small Naive Bayes classifier trained on a built-in sample dataset, so it works immediately without API keys, downloads, or third-party packages.

## Features

- Instant sentiment prediction with a confidence score
- Lightweight Naive Bayes machine-learning classifier implemented with Python's standard library
- Clean browser interface
- JSON API for integrations
- Unit tests for the prediction endpoint

## Quick start

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## API

Send a `POST` request to `/api/predict`:

```json
{ "text": "The delivery was fast and the product is excellent." }
```

Example response:

```json
{
  "sentiment": "positive",
  "confidence": 0.91
}
```

## Test

```bash
python -m unittest discover -s tests
```

## Project structure

```text
app.py              Web server and ML model
templates/index.html Browser interface
static/style.css     Interface styling
tests/test_app.py    Endpoint tests
```

> The included training data is deliberately small for a fast demo. Replace `TRAINING_EXAMPLES` in `app.py` with a real labelled dataset before using the model for production decisions.
