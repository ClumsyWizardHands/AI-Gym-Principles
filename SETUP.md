# AI Principles Gym - Setup Instructions

## Quick Start

1. **Activate Virtual Environment**
   ```bash
   cd ai-principles-gym
   # On Windows:
   venv\Scripts\activate
   # On Unix/macOS:
   source venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create Environment File**
   ```bash
   copy .env.example .env
   # Edit .env with your settings if needed
   ```

4. **Verify Installation**
   ```bash
   python example_usage.py
   ```

5. **Run Tests**
   ```bash
   pytest tests/test_setup.py -v
   ```

## Project Structure Overview

```
ai-principles-gym/
├── src/                    # Source code
│   ├── core/              # Core functionality (config, logging, models)
│   ├── scenarios/         # Scenario engine and archetypes
│   ├── adapters/          # Multi-framework AI adapters
│   └── api/               # FastAPI REST service
├── tests/                 # Test suite
├── deployment/            # Deployment configurations
├── client/                # Client libraries
├── memory-bank/           # Project documentation and context
├── .env.example           # Environment configuration template
├── requirements.txt       # Python dependencies
├── pyproject.toml        # Modern Python project config
└── README.md             # Project documentation
```

## Key Features Configured

✅ **Virtual Environment** - Isolated Python environment  
✅ **Structured Logging** - JSON-formatted logs with structlog  
✅ **Configuration Management** - Environment-based config with Pydantic  
✅ **Type Safety** - Strict mypy configuration  
✅ **Code Quality** - Black formatter, flake8 linter  
✅ **Testing** - pytest with async support  
✅ **Memory Bank** - Project context documentation  

## Next Development Steps

1. Implement core domain models in `src/core/models.py`
2. Create database schema in `src/core/database.py`
3. Build principle inference engine in `src/core/inference.py`
4. Develop scenario engine in `src/scenarios/engine.py`
5. Create FastAPI endpoints in `src/api/main.py`

## Environment Variables

The most important settings to configure in `.env`:

- `DATABASE_URL` - Database connection string
- `MIN_PATTERN_LENGTH` - Minimum actions for principle inference
- `CONSISTENCY_THRESHOLD` - Required consistency score
- `API_PORT` - Port for the API server

See `.env.example` for all available settings with documentation.
