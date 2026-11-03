# LAB-2 Report: Sentiment Analysis API with Hugging Face and FastAPI

## Table of Contents

1. [Student Information](#1-student-information)
2. [Introduction to the Selected Hugging Face Model](#2-introduction-to-the-selected-hugging-face-model)
3. [System Description](#3-system-description)
4. [Source Code Structure](#4-source-code-structure)
5. [How to Run the Application](#5-how-to-run-the-application)
6. [Calling the API (Requests and Responses)](#6-calling-the-api-requests-and-responses)
7. [Explanation of the Results](#7-explanation-of-the-results)
8. [Testing with `requests`](#8-testing-with-requests)
9. [Demo Video](#9-demo-video)

---

## 1. Student Information

| Field | Value |
|---|---|
| Full name | Tô Phúc Thành |
| Student ID | 25127499 |
| Class | 25C09 |
| Course | Computational Thinking, University of Science, VNU-HCM |
| GitHub repository | `<FILL IN PUBLIC GITHUB LINK>` |

---

## 2. Introduction to the Selected Hugging Face Model

| Item | Details |
|---|---|
| Model name | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` |
| Hugging Face page | <https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english> |
| Task | Sentiment analysis (text classification) |
| Language | English |
| Transformers pipeline | `sentiment-analysis` |
| Output labels | `POSITIVE`, `NEGATIVE` |

### What is the model?

**DistilBERT** is a smaller, faster version of **BERT**, a well-known language model from Google. It is created by *knowledge distillation*: a small "student" model is trained to imitate the behavior of the larger BERT "teacher" model. As a result, DistilBERT is much lighter (roughly 40% fewer parameters) and noticeably faster, while keeping most of BERT's language-understanding quality.

The model used in this project is `distilbert-base-uncased` **fine-tuned on SST-2** (the Stanford Sentiment Treebank, a dataset of movie-review sentences labeled as positive or negative). According to the model card, it reaches about 91% accuracy on the SST-2 development set.

### Why this model?

- It solves a clear and easy-to-demonstrate problem: deciding whether a sentence is positive or negative.
- It is small enough to run on an ordinary laptop CPU, with no GPU required.
- It works directly with the Hugging Face `transformers` `pipeline`, so the integration with FastAPI is simple.
- Its input (a text string) and output (a label and a score) map naturally to a JSON API.

### Limitations

- **Uncased**: it does not distinguish uppercase from lowercase letters.
- **English only**: text in other languages (including Vietnamese) will give unreliable results.
- **Binary output**: there is no "neutral" class, so every sentence is forced into either `POSITIVE` or `NEGATIVE`.
- Sarcasm, irony, and very short or ambiguous sentences can be misclassified.
- Input longer than the model's limit (512 tokens) is truncated.

---

## 3. System Description

The system is a **Web API built with FastAPI** that exposes the sentiment-analysis model over HTTP. A client sends English text as JSON; the API validates it, runs the model, and returns a structured JSON result.

```text
Client ──POST /predict {"text": "..."}──▶ FastAPI ──▶ Input validation
                                                        │
Client ◀── JSON {label, score, ...} ◀── Hugging Face pipeline (DistilBERT)
```

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Brief introduction to the API, the model, and example usage |
| `/health` | GET | Checks that the system is running and whether the model has been loaded |
| `/predict` | POST | Receives `{"text": "..."}`, calls the model, and returns the result in JSON |
| `/docs` | GET | Automatic Swagger UI for trying the API in the browser |

### Input validation and error handling

| Situation | HTTP status | Meaning |
|---|---|---|
| `text` is missing | `422` | Required field not provided |
| `text` is empty or only whitespace | `422` | Empty input rejected |
| `text` is not a string (e.g. a number) | `422` | Wrong data type |
| `text` is longer than 2000 characters | `422` | Input too long |
| Model cannot be loaded (e.g. no Internet on first run) | `503` | Service unavailable |
| Error while the model is running | `500` | Inference failure |

All error responses are JSON with a clear message.

### Model loading

The model is loaded **once**, on the first `/predict` request, and cached in memory. Later requests reuse the loaded model, so they are much faster. This is why the first request is slow while the model is downloaded and loaded.

---

## 4. Source Code Structure

```text
lab2-huggingface-api/
├── CLAUDE.md          # Working instructions for the coding assistant
├── main.py            # FastAPI application (endpoints, validation, model loading)
├── test_api.py        # API tests written with the requests library
├── requirements.txt   # Python dependencies
├── README.md          # This report
└── .gitignore         # Files excluded from Git (.venv/, __pycache__/, .env, ...)
```

### `main.py`

| Part | Purpose |
|---|---|
| Constants (`MODEL_NAME`, `MAX_TEXT_LENGTH`) | Model identifier and input length limit |
| `PredictRequest` (Pydantic) | Defines and validates the input: `text` must be a non-empty string of at most 2000 characters |
| `PredictResponse`, `RootResponse`, `HealthResponse` | Define a stable JSON structure for each response |
| `_build_pipeline()` and `get_classifier()` | Load the Hugging Face pipeline once (cached) and handle loading errors |
| Exception handlers | Turn validation errors and HTTP errors into clear JSON messages |
| `GET /`, `GET /health`, `POST /predict` | The three required endpoints |

### `test_api.py`

Uses `requests` to call the running API. It tests `GET /`, `GET /health`, two different inputs (one positive, one negative), and three error cases (empty text, missing field, wrong type), then prints `[OK]` / `[FAIL]` for each check.

### `requirements.txt`

`fastapi`, `uvicorn[standard]`, `transformers`, `torch`, `requests`.

---

## 5. How to Run the Application

**Requirements:** Python 3.10+ (Python 3.12 is recommended), an Internet connection for the first run (the model is downloaded from Hugging Face, a few hundred MB), and a few GB of free disk space for `torch`.

The commands below are for **Windows PowerShell**.

### Step 1: Create and activate a virtual environment

```powershell
cd path\to\lab2-huggingface-api
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> If PowerShell refuses to run the activation script, run this once: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate `.venv` again.
>
> After activation, the prompt starts with `(.venv)`.

### Step 2: Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Start the API

```powershell
uvicorn main:app --reload
```

When you see `Uvicorn running on http://127.0.0.1:8000`, the API is ready. Keep this window open.

### Step 4: Try the API

Open <http://127.0.0.1:8000/docs> in a browser, click **POST /predict**, then **Try it out**, enter a JSON body such as `{"text": "I love this movie!"}`, and click **Execute**.

> The very first `/predict` request is slow (the model is downloaded and loaded). Later requests are fast.

---

## 6. Calling the API (Requests and Responses)

### `GET /`

```powershell
curl.exe http://127.0.0.1:8000/
```

Example response:

```json
{
  "name": "LAB-2 Sentiment Analysis API",
  "description": "Sentiment analysis API for English text. Send a piece of text and receive a POSITIVE/NEGATIVE label with a confidence score.",
  "model": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
  "model_url": "https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english",
  "endpoints": {
    "GET /": "API introduction",
    "GET /health": "System health check",
    "POST /predict": "Sentiment analysis (JSON body: {\"text\": \"...\"})",
    "GET /docs": "Swagger UI"
  },
  "example_request": { "text": "I love this movie, it was fantastic!" }
}
```

### `GET /health`

```powershell
curl.exe http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": false,
  "model": "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
}
```

`model_loaded` becomes `true` after the first `/predict` request.

### `POST /predict`: PowerShell

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post `
  -ContentType "application/json" `
  -Body '{"text": "I love this movie, it was fantastic!"}'
```

### `POST /predict`: Python (`requests`)

```python
import requests

r = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"text": "I love this movie, it was fantastic!"},
)
print(r.status_code, r.json())
```

### Example 1: positive input

Request:

```json
{ "text": "I love this movie, it was fantastic!" }
```

Response (the score may differ slightly on your machine):

```json
{
  "text": "I love this movie, it was fantastic!",
  "label": "POSITIVE",
  "score": 0.9999,
  "model": "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
}
```

### Example 2: negative input

Request:

```json
{ "text": "This was a terrible experience, I want my money back." }
```

Response:

```json
{
  "text": "This was a terrible experience, I want my money back.",
  "label": "NEGATIVE",
  "score": 0.9998,
  "model": "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
}
```

### Example 3: invalid input (empty text)

Request:

```json
{ "text": "   " }
```

Response (HTTP 422):

```json
{
  "error": "Invalid input data.",
  "details": [
    { "field": "text", "message": "The 'text' field must not be empty." }
  ]
}
```

> The `score` values above are illustrative. Replace them with the real values you get when you run the API (for example, while recording the demo video).

---

## 7. Explanation of the Results

A successful `POST /predict` response contains four fields:

| Field | Type | Meaning |
|---|---|---|
| `text` | string | The input text after leading and trailing whitespace has been removed |
| `label` | string | The predicted sentiment: `POSITIVE` or `NEGATIVE` |
| `score` | number (0 to 1) | The model's confidence in the predicted label, rounded to 4 decimals |
| `model` | string | The Hugging Face model that produced the result |

### How to read `label` and `score`

- `label` is the class the model considers more likely.
- `score` is the probability the model assigns to that class. Because there are only two classes, a score close to **1.0** means the model is very confident, while a score close to **0.5** means the model is unsure (for example, with mixed or neutral sentences).
- The score is a measure of the model's confidence, not a guarantee that the prediction is correct.

### Interpreting the examples

- *"I love this movie, it was fantastic!"* contains strongly positive words ("love", "fantastic"), so the model returns `POSITIVE` with a score close to 1.
- *"This was a terrible experience, I want my money back."* contains strongly negative words and a complaint, so the model returns `NEGATIVE` with a score close to 1.
- A neutral or mixed sentence (for example, *"The movie was okay."*) will still receive one of the two labels, but usually with a lower score, because the model has no neutral class.

### Interpreting the errors

- **`422`** means the problem is in the request (missing, empty, wrongly typed, or too long `text`). The `details` list says which field is wrong and why.
- **`503`** means the model could not be loaded, usually because of a network problem on the first run.
- **`500`** means an unexpected error occurred while the model was processing the input.

---

## 8. Testing with `requests`

**The API must be running** (Section 5, Step 3) before running the tests. Open a second PowerShell window, go to the project folder, activate `.venv`, and run:

```powershell
python test_api.py
```

`test_api.py` checks `GET /`, `GET /health`, two different inputs (one positive, one negative), and three error cases. If everything works, every line shows `[OK]` and the script ends with `Done: 0 failure(s).`

---

## 9. Demo Video

- Video link: `https://youtu.be/_b6dkr2Q5cc`

