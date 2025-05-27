# AI Principles Gym

Training AI agents to develop behavioral principles through experience (not rules).

## Overview

AI Principles Gym is a framework for training AI agents to develop their own behavioral principles through experiential learning rather than following predefined rules. The system observes agent actions, identifies patterns, and infers underlying principles that guide behavior.

## Features

- **Experience-Based Learning**: Agents develop principles through interaction, not programmed rules
- **Pattern Recognition**: Advanced temporal pattern matching using DTW (Dynamic Time Warping)
- **Multi-Framework Support**: Adapters for various AI frameworks
- **Real-time Monitoring**: Track principle emergence and behavioral consistency
- **Structured Logging**: Comprehensive logging with structlog for debugging and analysis

## Installation

### Prerequisites

- Python 3.11 or higher
- Virtual environment (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-org/ai-principles-gym.git
cd ai-principles-gym
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix/macOS:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy environment configuration:
```bash
cp .env.example .env
# Edit .env with your specific settings
```

5. Initialize the database:
```bash
python -m src.core.database init
```

## Quick Start

### Running the API Server

```bash
uvicorn src.api.main:app --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Project Structure

```
ai-principles-gym/
├── src/
│   ├── core/           # Core models, inference engine, tracking, database
│   ├── scenarios/      # Scenario archetypes and engine
│   ├── adapters/       # Multi-framework support
│   └── api/           # FastAPI service
├── tests/             # Test suite
├── deployment/        # Deployment configurations
├── client/            # Client libraries/examples
└── memory-bank/       # Project memory and context
```

## Configuration

Key configuration parameters (see `.env.example` for full list):

- `MIN_PATTERN_LENGTH`: Minimum actions required for principle inference (default: 20)
- `CONSISTENCY_THRESHOLD`: Required consistency for principle acceptance (default: 0.85)
- `ENTROPY_THRESHOLD`: Maximum behavioral entropy before flagging (default: 0.7)
- `MAX_SCENARIOS_PER_SESSION`: Maximum scenarios per training session (default: 500)

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black src tests
```

### Type Checking

```bash
mypy src
```

### Linting

```bash
flake8 src tests
```

## Architecture

The system consists of several key components:

1. **Core Engine**: Handles principle inference and pattern matching
2. **Scenario Engine**: Generates training scenarios for agents
3. **Adapters**: Integrate with various AI frameworks
4. **API Service**: FastAPI-based REST API for interaction
5. **Database**: SQLite/PostgreSQL for storing principles and patterns

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with FastAPI, Pydantic, and structlog
- Uses DTAIDistance for temporal pattern matching
- Inspired by principles-based AI research
