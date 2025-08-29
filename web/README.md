# Rose Glass Dream Analysis Web Application

A public-facing web application that provides deep, culturally-aware dream analysis using the Rose Glass translation framework. Similar to how Claude Projects can be informed by repositories, this app gives users access to sophisticated dream interpretation.

## Features

- **Multi-Cultural Dream Translation**: View dreams through different cultural lenses (Medieval Islamic, Indigenous Oral, Buddhist Contemplative, etc.)
- **Pattern Recognition**: Identifies narrative flow, symbolic density, emotional charge, and relational content
- **Symbol Analysis**: Detects universal dream symbols with culture-specific interpretations
- **Privacy-First Design**: No dream content is permanently stored
- **LLM Enhancement** (optional): Integrates with Claude or GPT-4 for additional insights
- **Beautiful Dark UI**: Designed for comfortable dream journaling

## Quick Start

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/GreatPyreneseDad/rose-glass.git
cd rose-glass/web
```

2. Run the startup script:
```bash
./start.sh
```

3. Visit http://localhost:8000 in your browser

### Docker Deployment

```bash
docker-compose up -d
```

## Configuration

Create a `.env` file (copy from `.env.example`):

```env
# Optional - the app works without these
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
DEFAULT_LLM=claude
```

## How It Works

1. **User enters their dream**: The interface provides a text area for detailed dream descriptions

2. **Pattern extraction**: The Rose Glass framework identifies four key patterns:
   - Ψ (Psi): Narrative coherence and flow
   - ρ (Rho): Symbolic and archetypal density
   - q: Emotional intensity and charge
   - f: Social/relational content

3. **Cultural translation**: Dreams are viewed through selected cultural lenses, each revealing different aspects

4. **Symbol identification**: Universal symbols are detected and given culture-specific meanings

5. **Insights generation**: The system provides:
   - Pattern-based insights
   - Symbol interpretations
   - Cultural perspectives
   - Optional LLM-enhanced therapeutic insights

## Privacy & Ethics

- **No storage**: Dream content is never permanently stored
- **Session data**: Temporary sessions expire after 1 hour
- **No profiling**: The system never attempts to profile users
- **Cultural respect**: Multiple interpretations honor diverse perspectives

## API Endpoints

- `GET /`: Main web interface
- `POST /api/analyze_dream`: Analyze a dream
- `GET /api/lenses`: List available cultural lenses
- `GET /api/session/{id}`: Retrieve recent analysis (1 hour window)
- `GET /api/health`: Health check

## Example Dream Analysis

Input:
> "I was flying over a vast ocean, then suddenly found myself underwater breathing normally. My grandmother appeared as a young woman and handed me a golden key."

Output includes:
- Pattern scores (narrative flow, symbols, emotion, social)
- Detection of key symbols (flying, water, grandmother, key)
- Multiple cultural interpretations
- Insights about transformation and ancestral wisdom

## Therapeutic Value

Users and therapists have found this tool valuable for:
- Identifying recurring dream themes
- Understanding cultural influences on dream interpretation
- Exploring multiple perspectives on dream symbolism
- Facilitating dream journaling and reflection

## Development

### Adding New Cultural Lenses

See `/src/cultural_calibrations/calibration_development_guide.md` for instructions on adding new cultural perspectives.

### Customizing the UI

The interface uses inline CSS for easy customization. Modify the styles in `dream_app.py`.

### Extending Analysis

To add new dream analysis features:
1. Extend `RoseGlassDreamInterpreter` in `src/dream/rose_glass_dream_interpreter.py`
2. Update the API endpoint in `dream_app.py`
3. Add UI elements to display new insights

## Deployment Options

### Heroku
```bash
heroku create your-rose-glass-app
heroku config:set ANTHROPIC_API_KEY=your_key
git push heroku main
```

### AWS/GCP/Azure
Use the provided Dockerfile with your preferred container service.

### Vercel/Netlify
For serverless deployment, adapt the FastAPI app to work with serverless functions.

## Credits

- Based on Grounded Coherence Theory and the Rose Glass translation framework
- Inspired by Jungian dream analysis and cross-cultural dream interpretation
- Built with privacy and cultural dignity as core values

## License

MIT License - See LICENSE file

---

*"Dreams are personal letters from the unconscious. The Rose Glass helps us read them through many eyes, honoring their mystery while seeking understanding."*