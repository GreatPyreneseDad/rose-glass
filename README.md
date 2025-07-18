# Grounded Coherence Theory (GCT) - Implementation Suite

This repository contains implementations of **Grounded Coherence Theory (GCT)**, a psychometric and derivative-driven model of human decision dynamics. GCT models how coherence in beliefs, emotions, and social belonging evolves over time.

## 🧠 Projects

### 1. [GCT Market Sentiment Analysis](./gct-market-sentiment/)
A financial market sentiment engine that applies GCT to detect coherence shifts in market narratives and predict potential price movements.

**Key Features:**
- Real-time financial news analysis via Tiingo API
- NLP extraction of GCT variables (ψ, ρ, q, f)
- Coherence trajectory tracking with derivatives
- Bullish/bearish signal generation
- Interactive Streamlit dashboard

[Full Documentation →](./gct-market-sentiment/README.md)

### 2. [GCT Stock Sentiment Analyzer](./gct-stock-analyzer/)
A standalone script that evaluates a single stock and stores historical data to CSV for coherence-based sentiment metrics.

### 3. [SoulMath Moderation System](./soulmath-moderation-system/)
A Reddit content moderation system using GCT to analyze discourse quality and identify toxic behavior patterns.

**Key Features:**
- Reddit content scraping with Scrapy
- Coherence-based toxicity detection
- Community discourse health metrics
- React-based moderation dashboard

### 4. [GCT Login Service](./gct-login-service/)
A lightweight Express server that exposes a login page with links for Google and Apple authentication. It records crash logs under `gct-login-service/logs/`.

## 📚 Theory Documentation

- [GCT Whitepaper](./docs/WHITEPAPER.md) - Comprehensive theory and mathematical framework
- [Market Sentiment Specification](./docs/SPEC-1-Market-Sentiment-Engine.md) - Design document for financial applications

## 🔬 Core GCT Model

The enhanced GCT model (eGCT) incorporates:

```
C = ψ + ρ·ψ + q_opt + f·ψ + α·ρ·q_opt

where:
- ψ (psi): Clarity/precision of narrative
- ρ (rho): Reflective depth/wisdom
- q_opt: Optimized emotional charge
- f: Social belonging signal
- α: Coupling strength
```

## 🚀 Getting Started

Each project has its own setup instructions:

1. **Market Sentiment**: See [setup guide](./gct-market-sentiment/README.md#-quick-start)
2. **Moderation System**: See [setup guide](./soulmath-moderation-system/README.md)

## 📚 API Documentation

Both the market sentiment engine and moderation backend expose OpenAPI-compliant
endpoints. When running either service locally, navigate to `/docs` to view
interactive Swagger documentation or `/openapi.json` for the raw specification.

## 🛠 CI/CD & Containers

Automated GitHub Actions run on each commit to lint, format, audit, and test both
projects. A second workflow builds and publishes Docker images to GitHub Container
Registry so deployments stay reproducible. For GitLab users, a `.gitlab-ci.yml`
pipeline is also provided to mirror these steps. Dockerfiles are available so
each service can be run locally using `docker-compose up`. This launches the
market sentiment dashboard on port `8501` and the moderation dashboard on port
`3000`.

## 🤝 Contributing

We welcome contributions to GCT implementations! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](./LICENSE) file for details.

## 🔗 Links

- Theory development: [GCT Research](https://github.com/GreatPyreneseDad/GCT)
- Related work: [Emotional Superposition](./docs/WHITEPAPER.md#emotional-superposition)

---

*Grounded Coherence Theory: Bridging psychology, systems theory, and practical applications*
