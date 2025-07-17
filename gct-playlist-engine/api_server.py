from fastapi import FastAPI
from .data_pipeline import DataPipeline
from .feature_extractor import FeatureExtractor
from .gct_model import GCTModel
from .playlist_optimizer import PlaylistOptimizer

app = FastAPI()

@app.post("/generate-playlist")
def generate_playlist(request: dict):
    """Return a coherent playlist based on the request parameters."""
    track_ids = request.get("track_ids", [])
    length = int(request.get("length", len(track_ids) or 0))
    user_profile = request.get("user_profile", {})

    pipeline = DataPipeline(source_config={})
    extractor = FeatureExtractor()
    gct_model = GCTModel()
    optimizer = PlaylistOptimizer(gct_model, user_profile)

    if not track_ids:
        return {"playlist": [], "transition_scores": {}}

    metadata = pipeline.fetch_audio_metadata(track_ids)
    lyrics = pipeline.fetch_lyrics(track_ids)

    audio_features = extractor.extract_audio_features(metadata)
    lyric_features = extractor.extract_lyric_embeddings(lyrics)
    features = audio_features.join(lyric_features, how="outer").fillna(0)

    optimizer.set_features(features)

    seed = track_ids[0]
    playlist = optimizer.build_playlist(seed_track=seed, length=length)
    scores = optimizer.score_transitions([(playlist[i], playlist[i + 1]) for i in range(len(playlist) - 1)])

    return {"playlist": playlist, "transition_scores": scores}
