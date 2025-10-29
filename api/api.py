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


# Global state
data_loaded = False
startup_time = None


def current_time_iso():
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()


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


# ============================================================================
# Lifecycle Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize data service on application startup."""
    global data_loaded, startup_time
    startup_time = current_time_iso()
    logger.info("Starting Drug Interaction API...")

    try:
        data_loaded = initialize_data_service()
        status_msg = "Data service ready" if data_loaded else "Data service failed to load"
        logger.info(status_msg)
    except Exception as e:
        logger.error(f"Startup error: {e}")
        logger.error(traceback.format_exc())


@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown logging."""
    logger.info("Shutting down Drug Interaction API...")


# ============================================================================
# API Routes
# ============================================================================

@app.get("/", summary="Drug Interaction Checker API Overview", tags=["App Info"])
async def root():
    """API root endpoint with system overview."""
    return {
        "app": "Drug Interaction Checker API",
        "purpose": "AI-Powered Clinical Decision Support for polypharmacy safety.",
        "model": {
            "llm": "OpenAI GPT-5",
            "embeddings": "OpenAI text-embedding-3-large (3072-dim) — selected after benchmarking vs AWS Bedrock Titan V2 (512-dim)",
            "knowledge_base": "170,782 DrugBank interactions + 77,518 RxNorm mappings"
        },
        "author": "Ridwan Oladipo, MD | Medical AI Specialist",
        "version": "1.0.0",
        "documentation": "/docs",
    }


@app.get("/health", response_model=HealthResponse, summary="Service health check", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    scenarios_count = len(data_service.scenarios) if data_service.scenarios else 0
    live_enabled = data_service.kb_df is not None and data_service.lookups is not None

    return HealthResponse(
        status="healthy" if data_loaded else "unhealthy",
        scenarios_loaded=scenarios_count,
        live_lookup_enabled=live_enabled,
        version="1.0.0",
        startup_time=startup_time,
        timestamp=current_time_iso(),
    )


@app.get("/api/v1/scenarios", response_model=ScenarioList, summary="Get all available scenarios", tags=["Scenarios"])
@limiter.limit("10/minute")
async def get_scenarios(request: Request):
    """Retrieve list of all precomputed clinical scenarios."""
    if not data_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Data service not loaded")

    try:
        logger.info("Scenarios list requested")
        scenarios = data_service.get_all_scenarios()
        return ScenarioList(scenarios=scenarios, total=len(scenarios))
    except Exception as e:
        logger.error(f"Error getting scenarios: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve scenarios")


@app.post("/api/v1/check-interaction", summary="Check drug interaction (precomputed)", tags=["Interaction"])
@limiter.limit("20/minute")
async def check_interaction(request: Request, body: CheckInteractionRequest):
    """Get precomputed interaction analysis for a specific scenario."""
    if not data_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Data service not loaded")

    try:
        scenario_id = body.scenario_id
        logger.info(f"Interaction check requested for scenario: {scenario_id}")

        analysis = data_service.get_analysis_only(scenario_id)
        if analysis is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Scenario {scenario_id} not found")

        return analysis
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking interaction: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to check interaction")


@app.post("/api/v1/check-interaction-live", summary="Live drug interaction lookup (real-time KB query)",
          tags=["Interaction"])
@limiter.limit("10/minute")
async def check_interaction_live(request: Request, body: LiveInteractionRequest):
    """
    Perform real-time drug interaction lookup using direct KB matching.
    Returns Tier 1 (direct match) or Tier 3 (no interaction found).
    No LLM or embedding costs - pure KB lookup.
    """
    if not data_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Data service not loaded")

    try:
        drug_query = body.drug_query
        logger.info(f"Live interaction check requested: {drug_query}")

        result = data_service.check_interaction_live(drug_query)

        # Check for parsing errors
        if 'error' in result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['error']
            )

        return result

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error(f"Knowledge base error: {e}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Knowledge base not available")
    except ValueError as e:
        error_msg = str(e)
        logger.error(f"Validation error: {error_msg}")

        if "DataFrame is ambiguous" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal data structure error"
            )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )
    except Exception as e:
        logger.error(f"Error checking live interaction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check interaction: {str(e)}"
        )
