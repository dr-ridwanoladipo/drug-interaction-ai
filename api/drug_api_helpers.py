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

    def check_interaction_live(self, drug_query: str) -> Dict[str, Any]:
        """
        Perform live drug interaction lookup using direct KB matching.
        Uses check_polypharmacy_light for fast Tier 1/3 results (no LLM/embedding costs).
        """
        if self.kb_df is None or self.lookups is None:
            raise RuntimeError("Knowledge base not loaded")

        # Preprocess query: strip 'interaction' keyword and normalize separators
        text = drug_query.replace("interaction", "").replace("and", ",")
        drugs = [d.strip() for d in text.split(",") if d.strip()]

        # Perform direct KB lookup
        result = check_polypharmacy_light_orig(drugs, self.kb_df, self.lookups)
        return result

    def validate_data(self) -> Dict[str, bool]:
        """Validate that data is loaded."""
        return {
            'scenarios_loaded': self.scenarios is not None,
            'kb_loaded': self.kb_df is not None,
            'lookups_loaded': self.lookups is not None
        }

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of loaded data."""
        if self.scenarios is None:
            return {'error': 'Data not loaded'}

        return {
            'total_scenarios': len(self.scenarios),
            'scenarios_available': len(self.scenarios),
            'kb_interactions': len(self.kb_df) if self.kb_df is not None else 0,
            'live_lookup_enabled': self.kb_df is not None and self.lookups is not None
        }


# Global instance
data_service = DrugDataService()


def initialize_data_service() -> bool:
    """Initialize the global data service instance."""
    return data_service.load_data()


def get_data_service() -> DrugDataService:
    """Get the global data service instance."""
    return data_service
