from typing import Dict
import pandas as pd

class FeatureExtractor:
    """Extract audio and lyric features for coherence scoring."""

    def __init__(self, audio_model=None, nlp_model=None):
        self.audio_model = audio_model
        self.nlp_model = nlp_model

    def extract_audio_features(self, audio_metadata: pd.DataFrame) -> pd.DataFrame:
        """Normalize and vectorize audio descriptors like tempo or energy."""
        raise NotImplementedError

    def extract_lyric_embeddings(self, lyrics: Dict[str, str]) -> pd.DataFrame:
        """Convert lyrics into embedding vectors using NLP model."""
        raise NotImplementedError
