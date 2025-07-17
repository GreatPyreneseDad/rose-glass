from typing import List, Dict
import pandas as pd


class TechFeatureExtractor:
    """Extracts semantic and maturity features from technology documents."""

    def __init__(self, embedding_model: str):
        self.embedding_model = embedding_model
        # Placeholder for loading embedding model

    def embed_documents(self, docs: List[dict]) -> pd.DataFrame:
        """Return DataFrame of embedded documents and metadata."""
        import numpy as np

        rows = []
        for doc in docs:
            vector = np.random.normal(size=3)
            rows.append({
                "doc_id": doc["id"],
                "domain": doc["domain"],
                "e0": vector[0],
                "e1": vector[1],
                "e2": vector[2],
            })

        return pd.DataFrame(rows)

    def compute_domain_metrics(self, embeddings: pd.DataFrame) -> pd.DataFrame:
        """Aggregate document embeddings into per-domain features."""
        if embeddings.empty:
            return embeddings

        numeric = [c for c in embeddings.columns if c.startswith("e")]
        aggregated = embeddings.groupby("domain")[numeric].mean().reset_index()
        return aggregated
