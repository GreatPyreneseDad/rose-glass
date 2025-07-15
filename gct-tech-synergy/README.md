# GCT Tech Synergy Analyzer

This module prototypes a convergence analysis tool using Grounded Coherence Theory (GCT). It exposes a FastAPI service that computes coherence-based synergy scores between technology domains and produces a simple R&D roadmap.

## Components

- **ETLPipeline** – stubs document ingestion from scholarly APIs such as Semantic Scholar and USPTO.
- **TechFeatureExtractor** – embeds documents and aggregates features per technology domain.
- **GCTSynergyEngine** – calculates pairwise coherence metrics and clusters promising tech combinations.
- **RoadmapOptimizer** – generates a basic investment roadmap based on synergy scores.
- **api_service** – FastAPI endpoint `/analyze-convergence` returning scores, clusters, and a roadmap.

## Next Steps

1. Integrate real API calls to scholarly databases for fresh literature and patent data.
2. Implement maturity scoring models (e.g., TRL scales, publication velocity).
3. Calibrate GCT weight parameters through expert workshops.
4. Pilot the workflow on a small set of seed domains (e.g., AI and Bioinformatics) and evaluate results.
