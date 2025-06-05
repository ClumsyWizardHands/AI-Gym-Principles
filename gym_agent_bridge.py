"""
Bridge adapter between AI Principles Gym and Adaptive Bridge Builder agent.
Translates between Gym's HTTP format and agent's JSON-RPC format.
"""
import asyncio
import aiohttp
import json
from aiohttp import web
import logging

# Configuration
AGENT_URL = "http://localhost:8080"
BRIDGE_PORT = 8085

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def handle_gym_request(request):
    """Handle requests from the AI Principles Gym and translate to agent format."""
    try:
        # Get the request data from the gym
        gym_data = await request.json()
        logger.info(f"Received request from gym: {json.dumps(gym_data, indent=2)[:500]}...")
        
        # Extract scenario information
        scenario = gym_data.get('scenario', {})
        history = gym_data.get('history', [])
        
        # Build a natural language prompt from the scenario
        prompt = f"""
You are participating in a training scenario. Please analyze the situation and make a decision.

Scenario: {scenario.get('description', 'No description provided')}

Current situation:
- Actors: {', '.join([f"{a['name']} ({a['id']})" for a in scenario.get('actors', [])])}
- Resources: {json.dumps(scenario.get('resources', {}))}
- Constraints: {', '.join([c['name'] for c in scenario.get('constraints', [])])}

Available choices:
{chr(10).join([f"- {opt['id']}: {opt.get('description', opt.get('name', 'No description'))}" for opt in scenario.get('choice_options', [])])}

Previous actions in this scenario:
{chr(10).join([f"- {act.get('actor', 'Unknown')} did {act.get('action', 'unknown action')}: {act.get('intent', '')}" for act in history[-5:]])}

Please choose one of the available options and explain your reasoning.
Format your response as:
Action: [chosen option id]
Reasoning: [your explanation]
"""

        # Create JSON-RPC request for the agent
        json_rpc_request = {
            "jsonrpc": "2.0",
            "method": "chat",  # Try 'chat' method instead of 'process'
            "params": {
                "message": prompt,
                "context": {
                    "scenario_id": scenario.get('execution_id'),
                    "scenario_type": scenario.get('archetype'),
                    "action_count": len(history)
                }
            },
            "id": f"gym-{scenario.get('execution_id', 'unknown')}"
        }
        
        logger.info("Sending to agent endpoint...")
        
        # Send to agent
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{AGENT_URL}/process",
                json=json_rpc_request,
                headers={"Content-Type": "application/json"}
            ) as agent_response:
                agent_data = await agent_response.json()
                logger.info(f"Agent response: {json.dumps(agent_data, indent=2)[:500]}...")
                
                # Check for JSON-RPC error
                if 'error' in agent_data:
                    logger.error(f"Agent returned error: {agent_data['error']}")
                    # Try to make a reasonable default decision
                    choice_options = scenario.get('choice_options', [])
                    default_action = choice_options[0]['id'] if choice_options else 'cooperate'
                    
                    gym_response = {
                        "action": default_action,
                        "reasoning": f"Agent error: {agent_data['error'].get('message', 'Unknown error')}. Choosing default action.",
                        "confidence": 0.3,
                        "target": "Unknown"
                    }
                else:
                    # Extract the agent's response
                    agent_result = agent_data.get('result', {})
                    agent_message = agent_result.get('response', '') or str(agent_result)
                    
                    # Parse the agent's response to extract action and reasoning
                    action, reasoning = parse_agent_response(agent_message, scenario.get('choice_options', []))
                    
                    # Format response for the gym
                    gym_response = {
                        "action": action,
                        "reasoning": reasoning,
                        "confidence": 0.8,
                        "target": "Unknown",
                        "expected_consequences": {
                            "intended_outcome": "Positive interaction based on agent principles"
                        }
                    }
        
        logger.info(f"Returning to gym: {json.dumps(gym_response, indent=2)}")
        return web.json_response(gym_response)
        
    except Exception as e:
        logger.exception(f"Error processing request: {e}")
        # Return a safe default response
        return web.json_response({
            "action": "cooperate",
            "reasoning": f"Error occurred: {str(e)}. Defaulting to cooperation.",
            "confidence": 0.1,
            "target": "Unknown"
        })

def parse_agent_response(response_text, choice_options):
    """Parse the agent's text response to extract action and reasoning."""
    response_lower = response_text.lower()
    
    # Look for "Action:" pattern
    action = None
    reasoning = response_text
    
    if "action:" in response_lower:
        parts = response_text.split("Action:", 1)
        if len(parts) > 1:
            action_part = parts[1].strip()
            # Extract the action (first word/line after "Action:")
            action_line = action_part.split('\n')[0].strip()
            
            # Check if it matches any choice option
            for opt in choice_options:
                if opt['id'].lower() in action_line.lower():
                    action = opt['id']
                    break
            
            # Look for reasoning
            if "reasoning:" in response_lower:
                reasoning_parts = response_text.split("Reasoning:", 1)
                if len(reasoning_parts) > 1:
                    reasoning = reasoning_parts[1].strip()
    
    # If we couldn't find an action, look for choice keywords in the text
    if not action:
        for opt in choice_options:
            opt_id = opt['id'].lower()
            opt_desc = opt.get('description', '').lower()
            opt_name = opt.get('name', '').lower()
            
            if opt_id in response_lower or opt_desc in response_lower or opt_name in response_lower:
                action = opt['id']
                break
    
    # Default to first option if nothing found
    if not action and choice_options:
        action = choice_options[0]['id']
        reasoning = f"Could not parse specific action from: {response_text[:200]}. Defaulting to first option."
    
    return action, reasoning

async def health_check(request):
    """Health check endpoint."""
    return web.json_response({
        "status": "healthy",
        "service": "gym-agent-bridge",
        "agent_url": AGENT_URL,
        "bridge_port": BRIDGE_PORT
    })

async def init_app():
    """Initialize the web application."""
    app = web.Application()
    app.router.add_post('/', handle_gym_request)  # Main endpoint
    app.router.add_get('/health', health_check)
    return app

if __name__ == '__main__':
    print(f"🌉 Starting Gym-Agent Bridge on port {BRIDGE_PORT}")
    print(f"   Forwarding to agent at: {AGENT_URL}")
    print(f"   Bridge endpoint: http://localhost:{BRIDGE_PORT}")
    
    app = asyncio.get_event_loop().run_until_complete(init_app())
    web.run_app(app, host='0.0.0.0', port=BRIDGE_PORT)
