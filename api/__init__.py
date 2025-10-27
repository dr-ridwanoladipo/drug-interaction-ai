"""
💊 Drug Interaction Checker API Package
FastAPI backend for serving drug interaction analysis.

Author: Ridwan Oladipo, MD | Medical AI Specialist
"""

from .api import app
from .drug_api_helpers import data_service, initialize_data_service, get_data_service

__all__ = ["app", "data_service", "initialize_data_service", "get_data_service"]
