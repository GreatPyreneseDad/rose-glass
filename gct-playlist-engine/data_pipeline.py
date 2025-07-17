from typing import List, Dict
import pandas as pd

class DataPipeline:
    """Load audio metadata and lyrics for playlist generation."""

    def __init__(self, source_config: Dict):
        self.source_config = source_config

    def fetch_audio_metadata(self, track_ids: List[str]) -> pd.DataFrame:
        """Retrieve audio features and metadata for the given track IDs."""
        # This reference implementation generates synthetic metadata so that
        # the rest of the playlist engine can operate without external
        # dependencies. In a real system this would call a music catalog API.

        import numpy as np

        data = {
            "track_id": track_ids,
            "tempo": np.random.uniform(80, 160, size=len(track_ids)),
            "energy": np.random.uniform(0, 1, size=len(track_ids)),
            "valence": np.random.uniform(0, 1, size=len(track_ids)),
        }
        return pd.DataFrame(data).set_index("track_id")

    def fetch_lyrics(self, track_ids: List[str]) -> Dict[str, str]:
        """Retrieve lyrics text for semantic analysis."""
        # In lieu of real lyric data we return a simple placeholder string for
        # each track.  Downstream components can still operate on the resulting
        # dictionary for testing purposes.

        return {tid: f"Lyrics for {tid}" for tid in track_ids}
