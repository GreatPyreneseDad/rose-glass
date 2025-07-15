from fastapi import FastAPI
from typing import List, Dict
from .etl_pipeline import ETLPipeline
from .feature_extractor import TechFeatureExtractor
from .gct_synergy import GCTSynergyEngine
from .roadmap_optimizer import RoadmapOptimizer

app = FastAPI()


@app.post("/analyze-convergence")
def analyze_convergence(request: Dict):
    """Analyze technology convergence based on provided seed domains."""
    seed_domains: List[str] = request.get("seed_domains", [])
    horizon_years: int = int(request.get("horizon_years", 5))

    etl = ETLPipeline(db_config={})
    docs = etl.fetch_tech_documents()

    extractor = TechFeatureExtractor(embedding_model="all-MiniLM-L6-v2")
    embeddings = extractor.embed_documents(docs)
    domain_features = extractor.compute_domain_metrics(embeddings)

    engine = GCTSynergyEngine(weights={"psi": 1, "rho": 1, "q_opt": 1, "flow": 1, "alpha": 1})
    scores = engine.score_pairwise(domain_features)
    clusters = engine.cluster_synergies(scores)

    optimizer = RoadmapOptimizer(engine)
    roadmap = optimizer.generate_roadmap(seed_domains, horizon_years)

    return {
        "pairwise_scores": scores.to_dict(),
        "clusters": clusters,
        "roadmap": roadmap,
    }
