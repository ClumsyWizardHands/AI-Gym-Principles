# AI Principles Gym Python Client

A Python client library for easy integration with the AI Principles Gym training system.

## Installation

### From PyPI (when published)
```bash
pip install principles-gym-client
```

### From source
```bash
cd client/python
pip install -e .
```

## Features

- **Synchronous and Asynchronous Support**: Choose between `PrinciplesGymClient` for synchronous operations or `AsyncPrinciplesGymClient` for async/await patterns
- **Automatic Retry Logic**: Built-in exponential backoff for handling transient failures
- **Progress Tracking**: Real-time progress callbacks during training sessions
- **Comprehensive Error Handling**: Specific exceptions for different error scenarios
- **Type Safety**: Full type hints and Pydantic models for responses
- **Rate Limit Handling**: Automatic retry-after handling for rate limits

## Quick Start

### Synchronous Usage

```python
from principles_gym_client import PrinciplesGymClient

# Initialize client
client = PrinciplesGymClient(base_url="http://localhost:8000")

# Generate API key
api_key = client.generate_api_key("user123")

# Register an agent
agent_response = client.register_agent(
    agent_id="my-agent",
    framework="openai",
    config={
        "model": "gpt-4",
        "temperature": 0.7
    }
)

# Start training
session_id = client.start_training(
    agent_id=agent_response["agent_id"],
    num_scenarios=50
)

# Wait for completion with progress updates
def progress_callback(progress, completed, total):
    print(f"Progress: {progress:.1%} ({completed}/{total})")

client.wait_for_completion(
    session_id=session_id,
    progress_callback=progress_callback
)

# Get results
report = client.get_report(session_id)
print(f"Discovered {len(report['principles'])} principles")
```

### Asynchronous Usage

```python
import asyncio
from principles_gym_client import AsyncPrinciplesGymClient

async def main():
    async with AsyncPrinciplesGymClient() as client:
        # Generate API key
        api_key = await client.generate_api_key("user123")
        
        # Register agent
        agent = await client.register_agent(
            agent_id="my-claude-agent",
            framework="anthropic",
            config={"model": "claude-3-opus-20240229"}
        )
        
        # Start training
        session_id = await client.start_training(
            agent_id=agent["agent_id"],
            num_scenarios=100,
            scenario_types=["LOYALTY", "SCARCITY", "BETRAYAL"]
        )
        
        # Wait and get results
        await client.wait_for_completion(session_id)
        report = await client.get_report(session_id)
        
        print(report["summary"])

asyncio.run(main())
```

## API Reference

### Client Initialization

```python
PrinciplesGymClient(
    base_url: str = "http://localhost:8000",
    api_key: Optional[str] = None,
    timeout: int = 30,
    max_retries: int = 3,
    retry_delay: float = 1.0
)
```

### Core Methods

#### `generate_api_key(user_id: str, usage_limit: Optional[int] = None, expires_in_days: Optional[int] = None) -> str`
Generate a new API key for authentication.

#### `register_agent(agent_id: str, framework: str, config: Dict, name: Optional[str] = None, description: Optional[str] = None) -> Dict`
Register a new agent with the system.

**Supported frameworks:**
- `openai` - OpenAI GPT models
- `anthropic` - Anthropic Claude models
- `langchain` - LangChain agents
- `custom` - Custom Python functions

#### `start_training(agent_id: str, num_scenarios: int = 50, scenario_types: Optional[List[str]] = None, adaptive: bool = True) -> str`
Start a training session for the agent.

**Scenario types:**
- `LOYALTY` - Test loyalty vs self-interest
- `SCARCITY` - Resource allocation decisions
- `BETRAYAL` - Trust and betrayal scenarios
- `TRADEOFFS` - Multi-objective optimization
- `TIME_PRESSURE` - Decisions under time constraints
- `OBEDIENCE_AUTONOMY` - Following vs questioning orders
- `INFO_ASYMMETRY` - Decisions with incomplete information
- `REPUTATION_MGMT` - Reputation vs immediate gains
- `POWER_DYNAMICS` - Power use and restraint
- `MORAL_HAZARD` - Risk-taking with shared consequences

#### `wait_for_completion(session_id: str, poll_interval: int = 5, progress_callback: Optional[Callable] = None, timeout: Optional[int] = None)`
Wait for a training session to complete.

#### `get_report(session_id: str) -> Dict`
Get the final report with discovered principles.

## Error Handling

```python
from principles_gym_client import (
    APIError,
    AuthenticationError,
    RateLimitError,
    ResourceNotFoundError,
    TrainingError
)

try:
    client.start_training(agent_id="invalid-id")
except ResourceNotFoundError as e:
    print(f"Agent not found: {e}")
except RateLimitError as e:
    print(f"Rate limited: {e}")
except APIError as e:
    print(f"API error: {e}")
```

## Advanced Usage

### Custom Progress Tracking

```python
class TrainingMonitor:
    def __init__(self):
        self.start_time = time.time()
    
    def on_progress(self, progress, completed, total):
        elapsed = time.time() - self.start_time
        eta = elapsed / progress - elapsed if progress > 0 else 0
        print(f"[{completed}/{total}] {progress:.1%} - ETA: {eta:.0f}s")

monitor = TrainingMonitor()
client.wait_for_completion(
    session_id,
    progress_callback=monitor.on_progress
)
```

### Batch Training

```python
async def train_multiple_agents(agents: List[Dict]):
    async with AsyncPrinciplesGymClient() as client:
        # Generate API key once
        await client.generate_api_key("batch-user")
        
        # Register all agents
        agent_ids = []
        for agent in agents:
            response = await client.register_agent(**agent)
            agent_ids.append(response["agent_id"])
        
        # Start training sessions concurrently
        sessions = await asyncio.gather(*[
            client.start_training(agent_id, num_scenarios=50)
            for agent_id in agent_ids
        ])
        
        # Wait for all to complete
        await asyncio.gather(*[
            client.wait_for_completion(session_id)
            for session_id in sessions
        ])
        
        # Get all reports
        reports = await asyncio.gather(*[
            client.get_report(session_id)
            for session_id in sessions
        ])
        
        return reports
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
