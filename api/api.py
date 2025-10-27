"""
💊 Drug Interaction Checker API - FastAPI REST API
API for serving precomputed drug interaction analysis results.

Author: Ridwan Oladipo, MD | Medical AI Specialist
"""

from datetime import datetime
import logging
import time
import traceback
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from .drug_api_helpers import initialize_data_service, data_service

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d-%m-%Y | %I:%M%p",
    handlers=[logging.FileHandler("drug_api.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# FastAPI app initialization
app = FastAPI(
    title="Drug Interaction Checker API",
    description="AI-Powered Clinical Decision Support API for drug interactions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Log request processing time."""
    start = time.time()
    response = await call_next(request)
    process_time = (time.time() - start) * 1000
    logger.info(f"{request.method} {request.url.path} in {process_time:.2f} ms")
    return response


# ============================================================================
# Pydantic Models
# ============================================================================

class ScenarioInfo(BaseModel):
    id: int
    title: str


class ScenarioList(BaseModel):
    scenarios: List[ScenarioInfo]
    total: int


class CheckInteractionRequest(BaseModel):
    scenario_id: int


class LiveInteractionRequest(BaseModel):
    drug_query: str = Field(
        ...,
        description=(
            "STRICT FORMAT REQUIRED:\n"
            "• 2 drugs: 'Drug1 and Drug2' (example: 'Warfarin and Aspirin')\n"
            "• 3+ drugs: 'Drug1, Drug2 and Drug3' (example: 'Warfarin, Aspirin and Ibuprofen')\n"
            "DO NOT mix formats (e.g., 'Drug1 and Drug2, Drug3' is INVALID)"
        ),
        example="Warfarin and Aspirin",
        min_length=5,
        max_length=200
    )

    @validator('drug_query')
    def validate_query_format(cls, v):
        """Validate drug query format to prevent common errors."""
        v = v.strip()
        and_count = v.lower().count(' and ')
        comma_count = v.count(',')

        if and_count == 0 and comma_count == 0:
            raise ValueError(
                "INVALID: Must contain 'and'. Valid formats: 'Drug1 and Drug2' OR 'Drug1, Drug2 and Drug3'"
            )

        if comma_count > 0 and and_count == 0:
            raise ValueError(
                "INVALID: When using commas, must end with 'and DrugN'. Example: 'Warfarin, Aspirin and Ibuprofen'"
            )

        if comma_count > 0 and and_count > 0:
            and_pos = v.lower().find(' and ')
            comma_pos = v.find(',')
            if comma_pos > and_pos:
                raise ValueError(
                    "INVALID: Cannot mix formats like 'Drug1 and Drug2, Drug3'. Use 'Drug1 and Drug2' OR 'Drug1, Drug2 and Drug3'"
                )

        if and_count > 1:
            raise ValueError(
                "INVALID: Multiple 'and' keywords detected. Use 'Drug1 and Drug2' OR 'Drug1, Drug2 and Drug3' (only ONE 'and' at the end)"
            )

        return v


class HealthResponse(BaseModel):
    status: str
    scenarios_loaded: int
    live_lookup_enabled: bool
    version: str
    startup_time: str
    timestamp: str
