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

### 2. [SoulMath Moderation System](./soulmath-moderation-system/)
A Reddit content moderation system using GCT to analyze discourse quality and identify toxic behavior patterns.

**Key Features:**
- Reddit content scraping with Scrapy
- Coherence-based toxicity detection
- Community discourse health metrics
- React-based moderation dashboard

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