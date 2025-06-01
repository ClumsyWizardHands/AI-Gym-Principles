"""Decision-making wrapper for the Adaptive Bridge Builder.

This wrapper adds scenario processing capabilities to the Bridge Builder,
allowing it to work with the AI Principles Gym.
"""

from flask import Flask, request, jsonify
import requests
import json
import uuid
import random

app = Flask(__name__)

# The actual Adaptive Bridge Builder endpoint
AGENT_URL = "http://localhost:8080/process"

# Store agent card info
agent_card = None

def get_agent_card():
    """Get the agent card from the Bridge Builder."""
    global agent_card
    if not agent_card:
        try:
            json_rpc_request = {
                "jsonrpc": "2.0",
                "method": "getAgentCard",
                "params": {},
                "id": str(uuid.uuid4())
            }
            response = requests.post(
                AGENT_URL,
                json=json_rpc_request,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            if response.status_code == 200:
                result = response.json()
                if "result" in result:
                    agent_card = result["result"]
        except:
            pass
    return agent_card

def make_decision(scenario):
    """Make a decision based on the scenario.
    
    Since the Bridge Builder doesn't have decision-making capabilities,
    we'll implement a simple principle-based decision maker here.
    """
    
    # Extract scenario details
    options = scenario.get("choice_options", [])
    if not options:
        return "option_a", "No options provided", 0.1
    
    # Get agent principles from card if available
    card = get_agent_card()
    principles = []
    if card and "agent" in card:
        principles = card["agent"].get("principles", [])
    
    # Simple decision logic based on principles
    # If no principles, use the Bridge Builder's core purpose: connecting and facilitating
    if not principles:
        principles = [
            "Build bridges between different systems",
            "Facilitate communication and understanding",
            "Promote cooperation over conflict",
            "Find balanced solutions"
        ]
    
    # Analyze options based on principles
    best_option = options[0]
    best_score = 0
    reasoning_parts = []
    
    for option in options:
        score = 0
        option_desc = option.get("description", "").lower()
        
        # Score based on bridge-building principles
        if any(word in option_desc for word in ["cooperate", "collaborate", "bridge", "connect"]):
            score += 0.3
            reasoning_parts.append(f"{option['name']} promotes cooperation")
        
        if any(word in option_desc for word in ["balanced", "fair", "equitable"]):
            score += 0.2
            reasoning_parts.append(f"{option['name']} seeks balance")
            
        if any(word in option_desc for word in ["communicate", "facilitate", "enable"]):
            score += 0.2
            reasoning_parts.append(f"{option['name']} facilitates communication")
        
        # Avoid options that seem divisive
        if any(word in option_desc for word in ["conflict", "oppose", "against", "compete"]):
            score -= 0.2
            reasoning_parts.append(f"{option['name']} might create conflict")
        
        if score > best_score:
            best_score = score
            best_option = option
    
    # Generate reasoning
    if reasoning_parts:
        reasoning = "As a bridge builder, " + "; ".join(reasoning_parts[:2])
    else:
        reasoning = "Choosing based on bridge-building principles of cooperation and balance"
    
    # Calculate confidence (0.5 base + score adjustments)
    confidence = min(0.9, max(0.3, 0.5 + best_score))
    
    return best_option["id"], reasoning, confidence

@app.route('/wrapper', methods=['POST'])
def wrapper():
    """Process scenarios and make decisions."""
    try:
        # Get the gym's request
        gym_data = request.json
        scenario = gym_data.get("scenario", {})
        
        # Make a decision
        action_id, reasoning, confidence = make_decision(scenario)
        
        # Log the decision via echo method (for learning/tracking)
        try:
            echo_request = {
                "jsonrpc": "2.0",
                "method": "echo",
                "params": {
                    "decision": {
                        "scenario_id": scenario.get("execution_id"),
                        "action": action_id,
                        "reasoning": reasoning,
                        "confidence": confidence
                    }
                },
                "id": str(uuid.uuid4())
            }
            requests.post(AGENT_URL, json=echo_request, timeout=2)
        except:
            pass  # Don't fail if echo doesn't work
        
        # Return the decision to the gym
        return jsonify({
            "action": action_id,
            "reasoning": reasoning,
            "confidence": confidence
        })
        
    except Exception as e:
        print(f"Wrapper error: {str(e)}")
        # Return a safe default response
        return jsonify({
            "action": "option_a",
            "reasoning": f"Error in decision process: {str(e)}",
            "confidence": 0.1
        })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    card = get_agent_card()
    return jsonify({
        "status": "healthy",
        "agent_connected": card is not None,
        "wrapper_version": "1.0.0"
    })

if __name__ == '__main__':
    print("🤖 Decision Wrapper for Adaptive Bridge Builder")
    print("   Starting on http://localhost:8091/wrapper")
    print("   This adds decision-making capabilities to the Bridge Builder")
    
    # Try to get agent card on startup
    card = get_agent_card()
    if card:
        print(f"   Connected to: {card.get('agent', {}).get('name', 'Unknown Agent')}")
    else:
        print("   Warning: Could not connect to Bridge Builder at startup")
    
    app.run(host='0.0.0.0', port=8091, debug=True)
