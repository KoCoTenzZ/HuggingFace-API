"""LAB-2: Hugging Face API with FastAPI.

Model: distilbert/distilbert-base-uncased-finetuned-sst-2-english
Task: sentiment analysis of English text (POSITIVE / NEGATIVE).
"""

import logging
from functools import lru_cache
from threading import Lock
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StrictStr, field_validator

MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
MODEL_URL = f"https://huggingface.co/{MODEL_NAME}"
MAX_TEXT_LENGTH = 2000

logger = logging.getLogger("lab2")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="LAB-2: Sentiment Analysis API",
    description="API for sentiment analysis of English text using a Hugging Face model.",
    version="1.0.0",
)

_load_lock = Lock()


# ----------------------------- Schemas -----------------------------
class PredictRequest(BaseModel):
    text: StrictStr = Field(
        ...,
        description="English text to analyze.",
        examples=["I love this movie, it was fantastic!"],
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("The 'text' field must not be empty.")
        if len(value) > MAX_TEXT_LENGTH:
            raise ValueError(
                f"The 'text' field is too long (maximum {MAX_TEXT_LENGTH} characters)."
            )
        return value


class PredictResponse(BaseModel):
    text: str
    label: str
    score: float
    model: str


class RootResponse(BaseModel):
    name: str
    description: str
    model: str
    model_url: str
    endpoints: dict
    example_request: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model: str


# ----------------------------- Model -----------------------------
@lru_cache(maxsize=1)
def _build_pipeline():
    # Imported lazily so the API starts fast; the model is loaded only once.
    from transformers import pipeline

    logger.info("Loading model %s ...", MODEL_NAME)
    clf = pipeline("sentiment-analysis", model=MODEL_NAME)
    logger.info("Model loaded.")
    return clf


_model_loaded = False


def get_classifier():
    global _model_loaded
    with _load_lock:
        try:
            clf = _build_pipeline()
        except Exception as exc:  # network error, missing library, out of memory...
            logger.exception("Failed to load the model")
            raise HTTPException(
                status_code=503,
                detail=f"Failed to load the Hugging Face model: {exc}",
            )
        _model_loaded = True
        return clf


# ----------------------------- Error handling -----------------------------
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    details: List[dict] = []
    for err in exc.errors():
        loc = [str(p) for p in err.get("loc", []) if p != "body"]
        field = ".".join(loc) if loc else "body"
        err_type = err.get("type", "")
        if err_type == "missing":
            message = f"Missing required field '{field}'."
        elif err_type in ("string_type", "model_attributes_type"):
            message = f"Field '{field}' has an invalid data type (a text string is required)."
        elif err_type in ("json_invalid", "model_type"):
            message = 'The request body must be valid JSON like {"text": "..."}.'
        else:
            message = str(err.get("msg", "Invalid input")).replace("Value error, ", "")
        details.append({"field": field, "message": message})
    return JSONResponse(
        status_code=422,
        content={"error": "Invalid input data.", "details": details},
    )


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


# ----------------------------- Endpoints -----------------------------
@app.get("/", response_model=RootResponse)
def root():
    return RootResponse(
        name="LAB-2 Sentiment Analysis API",
        description=(
            "Sentiment analysis API for English text. Send a piece of text and "
            "receive a POSITIVE/NEGATIVE label with a confidence score."
        ),
        model=MODEL_NAME,
        model_url=MODEL_URL,
        endpoints={
            "GET /": "API introduction",
            "GET /health": "System health check",
            "POST /predict": 'Sentiment analysis (JSON body: {"text": "..."})',
            "GET /docs": "Swagger UI",
        },
        example_request={"text": "I love this movie, it was fantastic!"},
    )


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=_model_loaded, model=MODEL_NAME)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    classifier = get_classifier()
    try:
        result = classifier(request.text, truncation=True)[0]
    except Exception as exc:
        logger.exception("Error while running the model")
        raise HTTPException(
            status_code=500, detail=f"Error while running the model (inference): {exc}"
        )
    return PredictResponse(
        text=request.text,
        label=str(result["label"]),
        score=round(float(result["score"]), 4),
        model=MODEL_NAME,
    )
