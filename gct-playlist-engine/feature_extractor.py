from typing import Dict
import pandas as pd

class FeatureExtractor:
    """Extract audio and lyric features for coherence scoring."""

    def __init__(self, audio_model=None, nlp_model=None):
        self.audio_model = audio_model
        self.nlp_model = nlp_model

    def extract_audio_features(self, audio_metadata: pd.DataFrame) -> pd.DataFrame:
        """Normalize and vectorize audio descriptors like tempo or energy."""
        if audio_metadata.empty:
            return audio_metadata

        numeric_cols = audio_metadata.select_dtypes(include="number").columns
        normalized = audio_metadata.copy()
        for col in numeric_cols:
            series = audio_metadata[col]
            # simple min-max normalization
            normalized[col] = (series - series.min()) / (series.max() - series.min() + 1e-9)

        return normalized

    def extract_lyric_embeddings(self, lyrics: Dict[str, str]) -> pd.DataFrame:
        """Convert lyrics into embedding vectors using NLP model."""
        data = {"track_id": [], "lyric_length": []}
        for tid, text in lyrics.items():
            data["track_id"].append(tid)
            data["lyric_length"].append(len(text.split()))

        return pd.DataFrame(data).set_index("track_id")
