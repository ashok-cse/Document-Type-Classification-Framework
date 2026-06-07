"""FastAPI service for the DTCF German document classifier.

Endpoints:
    GET  /          — HTML upload page
    GET  /health    — liveness + model status
    GET  /classes   — list of class labels
    POST /predict   — multipart form upload, returns JSON prediction
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import inference

APP_DIR = Path(__file__).parent

app = FastAPI(
    title="DTCF — German Document Type Classification",
    description="Upload a German document page; receive its predicted type.",
    version="0.1.0",
)
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")


@app.on_event("startup")
async def _warm_model() -> None:
    """Pre-load the model so the first user request isn't cold."""
    if os.getenv("EAGER_LOAD", "1") == "1":
        try:
            inference.load_model()
        except Exception as exc:
            # Don't crash the worker — surface via /health instead.
            print(f"[startup] model load failed: {exc}")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (APP_DIR / "templates" / "index.html").read_text()


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": inference.model_loaded(),
        "classes": inference.CLASS_NAMES,
    }


@app.get("/classes")
async def classes() -> dict:
    return {"classes": inference.CLASS_NAMES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected an image upload, got content-type {file.content_type!r}",
        )
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        result = inference.predict(data)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failure: {exc}")

    return JSONResponse(result)
