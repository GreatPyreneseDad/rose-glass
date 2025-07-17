from typing import List, Dict
import pandas as pd
from .gct_synergy import GCTSynergyEngine


class RoadmapOptimizer:
    """Generate R&D roadmaps based on synergy scores and readiness."""

    def __init__(self, synergy_engine: GCTSynergyEngine):
        self.synergy_engine = synergy_engine

    def generate_roadmap(self, seed_domains: List[str], horizon_years: int) -> List[Dict]:
        """Create a time-phased roadmap for technology convergence."""
        if not seed_domains:
            return []

        roadmap = []
        for year in range(1, horizon_years + 1):
            domain = seed_domains[(year - 1) % len(seed_domains)]
            roadmap.append({"year": year, "focus_domain": domain})
        return roadmap

    def visualize_roadmap(self, roadmap: List[Dict]):
        """Visualize roadmap (placeholder)."""
        for entry in roadmap:
            print(f"Year {entry['year']}: focus on {entry['focus_domain']}")
