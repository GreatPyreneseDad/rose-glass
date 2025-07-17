from typing import List, Tuple, Dict

class PlaylistOptimizer:
    """Generate coherent playlists using the GCT model."""

    def __init__(self, gct_model, user_profile: Dict):
        self.gct_model = gct_model
        self.user_profile = user_profile

    def score_transitions(self, track_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], float]:
        """Return coherence scores for track transitions."""
        if not hasattr(self, "features"):
            return {}

        coherence = self.gct_model.compute_coherence(self.features)["coherence"]
        scores = {}
        for a, b in track_pairs:
            if a in coherence and b in coherence:
                scores[(a, b)] = float(abs(coherence[b] - coherence[a]))
        return scores

    def build_playlist(self, seed_track: str, length: int) -> List[str]:
        """Construct a playlist maximizing cumulative coherence."""
        if not hasattr(self, "features"):
            return []

        coherence = self.gct_model.compute_coherence(self.features)["coherence"]
        available = [t for t in coherence.index if t != seed_track]
        sorted_tracks = coherence.loc[available].sort_values(ascending=False).index.tolist()
        playlist = [seed_track] + sorted_tracks[: max(length - 1, 0)]
        return playlist

    def set_features(self, features_df):
        """Store features used for optimization."""
        self.features = features_df
