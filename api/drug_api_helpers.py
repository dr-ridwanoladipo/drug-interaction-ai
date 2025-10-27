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

    def load_data(self) -> bool:
        """Load precomputed analysis results and KB for live lookup."""
        try:
            logger.info("Loading drug interaction analysis data...")

            # Load precomputed scenarios
            with open(self.data_path / 'precomputed_samples.json', 'r') as f:
                self.scenarios = json.load(f)
            logger.info(f"Loaded {len(self.scenarios)} scenarios successfully")

            # Load KB and lookups for live interaction checking
            logger.info("Loading knowledge base for live lookups...")
            self.kb_df, self.lookups = load_data()
            logger.info("Knowledge base loaded successfully")

            return True

        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            return False

    def get_all_scenarios(self) -> List[Dict[str, Any]]:
        """Return list of all scenarios with titles only."""
        if self.scenarios is None:
            return []

        return [
            {
                "id": i,
                "title": scenario["scenario"]
            }
            for i, scenario in enumerate(self.scenarios)
        ]

    def get_analysis_only(self, scenario_id: int) -> Optional[Dict[str, Any]]:
        """Get analysis object only (no wrapper) for scenario."""
        if self.scenarios is None:
            return None

        if scenario_id < 0 or scenario_id >= len(self.scenarios):
            return None

        return self.scenarios[scenario_id]["analysis"]
