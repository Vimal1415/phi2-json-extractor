# app/schemas.py
# ─────────────────────────────────────────────────────────────
# Pydantic schemas define the exact shape of every request
# and response in your API.
#
# FastAPI uses these automatically:
# - If a request comes in missing "text", FastAPI returns 422
#   before your code even runs
# - Response schemas document your API automatically
# ─────────────────────────────────────────────────────────────

from pydantic import BaseModel, Field
from typing import Optional


# ── REQUEST SCHEMA ───────────────────────────────────────────
class ExtractRequest(BaseModel):
    """
    What the client sends to POST /extract
    
    Example:
        {
            "text": "Aarav is 28, software engineer in Bangalore"
        }
    """
    text: str = Field(
        ...,                          # ... means required — cannot be None
        min_length=10,                # reject suspiciously short inputs
        max_length=500,               # reject inputs too long for the model
        description="The natural language sentence to extract from",
        example="Aarav is 28 years old and works as a software engineer in Bangalore."
    )


# ── RESPONSE SCHEMAS ─────────────────────────────────────────
# These mirror the nested JSON structure the model outputs.
# We use nested Pydantic models to match nested JSON.

class PersonDetails(BaseModel):
    """The 'person' key in the response"""
    name: str
    age: int


class WorkDetails(BaseModel):
    """The 'work' key in the response"""
    job: str
    city: str
    years_experience: int


class SeniorityDetails(BaseModel):
    """The 'details' key in the response"""
    is_senior: bool
    has_certification: bool


class ExtractResponse(BaseModel):
    """
    What the server returns on success
    
    Example:
        {
            "person": {"name": "Aarav", "age": 28},
            "work": {"job": "software engineer", "city": "Bangalore", "years_experience": 4},
            "details": {"is_senior": false, "has_certification": false},
            "raw_output": "...",
            "success": true
        }
    """
    person:     PersonDetails
    work:       WorkDetails
    details:    SeniorityDetails
    raw_output: str   = Field(description="Raw text output from model before parsing")
    success:    bool  = True


class ErrorResponse(BaseModel):
    """
    What the server returns when something goes wrong
    
    Example:
        {
            "success": false,
            "error": "Model output was not valid JSON",
            "raw_output": "Here is the person: Aarav..."
        }
    """
    success:    bool = False
    error:      str
    raw_output: Optional[str] = None