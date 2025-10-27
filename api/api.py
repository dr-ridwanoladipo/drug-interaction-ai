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
