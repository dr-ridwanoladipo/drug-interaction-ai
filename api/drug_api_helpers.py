"""
💊 Drug Interaction Checker - Data Service Module
Data serving functions for FastAPI backend.

Author: Ridwan Oladipo, MD | Medical AI Specialist
"""

import json
import logging
import sys
from typing import Dict, List, Optional, Any
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))
from rag_pipeline import load_data, check_polypharmacy_light as check_polypharmacy_light_orig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DrugDataService:
    """Drug Interaction Analysis Data Service - Serves precomputed results."""

    def __init__(self):
        """Initialize empty placeholders; populate via load_data()."""
        self.scenarios = None
        self.kb_df = None
        self.lookups = None
        self.data_path = Path("data")
