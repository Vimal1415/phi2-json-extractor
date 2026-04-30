# app/main.py
# ─────────────────────────────────────────────────────────────
# The web server — routes, startup, error handling.
# No ML code here. model.py handles all of that.
#
# FastAPI automatically generates docs at /docs
# Open http://localhost:8000/docs after starting the server
# ─────────────────────────────────────────────────────────────

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.schemas import ExtractRequest, ExtractResponse, ErrorResponse
from app.model import load_model, extract

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── LIFESPAN ─────────────────────────────────────────────────
# lifespan controls what happens at server startup and shutdown.
#
# @asynccontextmanager means this function has two phases:
# - Everything before yield  → runs at STARTUP
# - Everything after yield   → runs at SHUTDOWN
#
# We load the model at startup so it's ready before
# the first request arrives. Never load it per-request.

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ──
    logger.info("Server starting — loading model...")
    load_model()                    # load once, keep in memory
    logger.info("Model loaded — server ready")
    yield
    # ── SHUTDOWN ──
    logger.info("Server shutting down")


# ── APP INSTANCE ─────────────────────────────────────────────
app = FastAPI(
    title="Phi-2 Nested JSON Extractor",
    description="Extract structured nested JSON from natural language using a fine-tuned Phi-2 model",
    version="1.0.0",
    lifespan=lifespan,
)


# ── ROUTES ───────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """
    Health check endpoint.
    
    Used by:
    - Docker to verify container is running
    - Render to verify deployment succeeded
    - GitHub Actions to verify server started correctly
    
    Always returns 200 if server is up.
    """
    return {"status": "ok", "model": "phi2-nested-json-lora"}


@app.post(
    "/extract",
    response_model=ExtractResponse,     # FastAPI validates response matches this
    responses={
        200: {"description": "Successfully extracted JSON"},
        422: {"description": "Invalid input — text too short/long"},
        500: {"description": "Model failed to produce valid JSON"},
    }
)
def extract_endpoint(request: ExtractRequest):
    """
    Extract structured nested JSON from a natural language sentence.
    
    Send a sentence describing a person and receive a nested JSON
    with their name, age, job, city, experience, and seniority.
    
    Example input:
        {"text": "Aarav is 28, a senior software engineer in Bangalore with 6 years experience"}
    
    Example output:
        {
            "person": {"name": "Aarav", "age": 28},
            "work": {"job": "software engineer", "city": "Bangalore", "years_experience": 6},
            "details": {"is_senior": true, "has_certification": false},
            "raw_output": "...",
            "success": true
        }
    """
    start_time = time.time()
    logger.info(f"Received request: {request.text[:80]}")

    try:
        # Call model.py — one line, clean separation
        parsed, raw_output = extract(request.text)

        latency = time.time() - start_time
        logger.info(f"Extraction successful in {latency:.2f}s")

        # Build response using our schema
        json_start = raw_output.find('{')
        json_end   = raw_output.rfind('}') + 1
        clean_output = raw_output[json_start:json_end] if json_start != -1 else raw_output

        return ExtractResponse(
        person=parsed["person"],
        work=parsed["work"],
        details=parsed["details"],
        raw_output=clean_output,
        success=True
    )

    except ValueError as e:
        # Model produced output but it wasn't valid JSON
        # Return 500 with details so client knows what went wrong
        latency = time.time() - start_time
        logger.error(f"Extraction failed in {latency:.2f}s: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e),
                "message": "Model failed to produce valid JSON. Try rephrasing the input."
            }
        )

    except Exception as e:
        # Unexpected error — log it and return generic 500
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail={"success": False, "error": "Internal server error"}
        )