# Learning Agent with Memory - User Guide

## Overview

The Learning Agent with Memory is an AI agent that:
- 🧠 **Maintains memory** of past experiences
- 📈 **Learns behavioral principles** from successes and failures  
- 🎯 **Makes decisions** based on learned principles and past experiences
- 🔄 **Improves over time** through continuous training
- 🤖 **Uses GPT-3.5** for intelligent, context-aware decision making

## Quick Start

### 1. Setup OpenAI API Key

Since you have an OpenAI API key, run this to set it up:

```bash
cd ai-principles-gym
python setup_openai_agent.py
```

This will:
- Configure your OpenAI API key
- Test the connection to OpenAI
- Update your agent to use GPT-3.5

### 2. Start Training

Once your API key is configured:

```bash
python start_training_learning_agent.py
```

This will:
- Start a training session with various scenarios
- The agent will make intelligent decisions using GPT-3.5
- Learn from successes and failures
- Update its behavioral principles

### 3. View Progress

To check your agent's learning progress:

```bash
python start_training_learning_agent.py progress
```

This shows:
- Current learned principles (0-1 scale)
- Total experiences accumulated
- Success rate trends
- Recent performance metrics

## How It Works

### Memory System

The agent maintains a JSON file (`learning_agent_with_memory_memory.json`) containing:

```json
{
  "name": "Learning Agent with Memory",
  "principles": {
    "fairness": 0.5,
    "efficiency": 0.5,
    "safety": 0.5,
    "cooperation": 0.5,
    "adaptability": 0.5
  },
  "experiences": [
    {
      "timestamp": "2025-01-09T10:30:00",
      "scenario_type": "resource_allocation",
      "context": "Distribute resources among team members",
      "decision": "Equal distribution based on need",
      "outcome": {"score": 0.8},
      "success": true
    }
  ],
  "last_updated": "2025-01-09T10:35:00"
}
```

### Learning Process

1. **Scenario Presentation**: The gym presents ethical dilemmas, resource allocation challenges, etc.

2. **Decision Making**: The agent uses:
   - Current principle values
   - Relevant past experiences
   - GPT-3.5's reasoning capabilities

3. **Principle Updates**:
   - Successful decisions reinforce related principles (+0.1)
   - Failed decisions slightly reduce all principles (-0.05)
   - Principles are capped between 0.1 and 1.0

4. **Experience Storage**: Each decision and outcome is saved for future reference

### Scenario Types

The agent trains on various scenarios:
- **Resource Allocation**: Fair distribution challenges
- **Trust Games**: Building and maintaining trust
- **Ethical Dilemmas**: Moral decision-making
- **Cooperation Games**: Team collaboration
- **Fairness Tests**: Equity and justice scenarios

## Advanced Usage

### Custom Training Sessions

Create custom training configurations:

```python
session_config = {
    "agent_id": agent_id,
    "config": {
        "scenario_types": ["ethical_dilemma", "trust_game"],
        "num_scenarios": 50,
        "randomize": True,
        "include_edge_cases": True
    }
}
```

### Analyzing Learning Patterns

The agent's memory file reveals learning patterns:
- Which principles are strongest
- Which scenario types the agent excels at
- How decision-making evolves over time

### Integration with Other Systems

The agent can be integrated with:
- Custom scenario generators
- Real-world decision systems
- Other AI frameworks via adapters

## Monitoring in Web UI

Visit http://localhost:5173 to:
- View real-time training progress
- See behavioral pattern visualizations
- Compare agent performance over time
- Export training data

## Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not found"**
   - Run `python setup_openai_agent.py` to configure your key

2. **"Agent not found"**
   - Ensure you've run `python learning_agent_with_memory.py` first

3. **Poor performance**
   - Check if principles are too low (near 0.1)
   - Ensure diverse scenario types in training
   - Verify OpenAI API is working correctly

### Reset Agent Memory

To start fresh:
```bash
rm learning_agent_with_memory_memory.json
python learning_agent_with_memory.py
```

## Tips for Better Learning

1. **Diverse Training**: Use all scenario types for well-rounded development
2. **Consistent Sessions**: Regular training helps maintain momentum
3. **Monitor Principles**: Watch which principles grow strongest
4. **Analyze Failures**: Failed scenarios provide valuable learning
5. **Long Sessions**: More scenarios = better pattern recognition

## Example Output

```
🎯 Starting training session for Learning Agent with Memory...
✅ Training session started!
Session ID: abc123-def456

📊 Training Progress:
--------------------------------------------------
Progress: 5/20 scenarios (25.0%)
  Last scenario: ethical_dilemma
  Score: 0.85
  Decision quality: {'fairness': 0.9, 'safety': 0.8}

Progress: 10/20 scenarios (50.0%)
  Last scenario: resource_allocation
  Score: 0.92
  Decision quality: {'efficiency': 0.95, 'fairness': 0.88}

✅ Training session completed!

📈 Training Results:
--------------------------------------------------
Total scenarios: 20
Average score: 0.87
Success rate: 85.0%

🎯 Principle Development:
  Fairness: 0.78
  Efficiency: 0.82
  Safety: 0.71
  Cooperation: 0.75
  Adaptability: 0.69

🧠 Agent Memory Status:
--------------------------------------------------
Total experiences: 20
Learned Principles:
  Fairness: 0.78
  Efficiency: 0.82
  Safety: 0.71
  Cooperation: 0.75
  Adaptability: 0.69
```

## Next Steps

1. **Experiment**: Try different training configurations
2. **Analyze**: Study how principles evolve over sessions
3. **Customize**: Add new scenario types via plugins
4. **Share**: Export and compare agent behaviors
5. **Scale**: Train multiple agents with different approaches

Happy training! 🎉
