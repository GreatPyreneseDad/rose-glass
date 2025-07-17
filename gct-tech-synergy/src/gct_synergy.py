from typing import List, Dict
import pandas as pd


class GCTSynergyEngine:
    """Calculate coherence-based synergy scores between technologies."""

    def __init__(self, weights: Dict[str, float]):
        self.weights = weights

    def score_pairwise(self, domain_features: pd.DataFrame) -> pd.DataFrame:
        """Compute pairwise coherence scores between technology domains."""
        if domain_features.empty:
            return pd.DataFrame()

        import numpy as np

        df = domain_features.set_index("domain")
        domains = df.index.tolist()
        vectors = df.values

        # cosine similarity matrix
        norm = np.linalg.norm(vectors, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        normalized = vectors / norm
        sim = normalized @ normalized.T

        return pd.DataFrame(sim, index=domains, columns=domains)

    def cluster_synergies(self, score_matrix: pd.DataFrame) -> List[List[str]]:
        """Group domains into high-coherence clusters."""
        if score_matrix.empty:
            return []

        threshold = 0.8
        visited = set()
        clusters = []
        for domain in score_matrix.index:
            if domain in visited:
                continue
            cluster = [domain]
            visited.add(domain)
            for other in score_matrix.columns:
                if other not in visited and score_matrix.loc[domain, other] >= threshold:
                    cluster.append(other)
                    visited.add(other)
            clusters.append(cluster)

        return clusters
