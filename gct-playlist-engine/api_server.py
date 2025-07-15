from fastapi import FastAPI
from .data_pipeline import DataPipeline
from .feature_extractor import FeatureExtractor
from .gct_model import GCTModel
from .playlist_optimizer import PlaylistOptimizer

app = FastAPI()

@app.post("/generate-playlist")
def generate_playlist(request: dict):
    """Return a coherent playlist based on the request parameters."""
    # 1. Load user profile (placeholder)
    user_profile = {}

    pipeline = DataPipeline(source_config={})
    extractor = FeatureExtractor(audio_model=None, nlp_model=None)
    gct_model = GCTModel()
    optimizer = PlaylistOptimizer(gct_model, user_profile)

    # TODO: fetch data, compute features, and build playlist
    raise NotImplementedError
