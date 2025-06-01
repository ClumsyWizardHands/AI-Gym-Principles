"""JSON-RPC 2.0 adapter for the Adaptive Bridge Builder."""

from flask import Flask, request, jsonify
import requests
import json
import uuid

app = Flask(__name__)

# The actual Adaptive Bridge Builder endpoint
AGENT_URL = "http://localhost:8080/process"

@app.route('/adapter', methods=['POST'])
def adapter():
    """Convert gym's JSON format to JSON-RPC 2.0 and back."""
    try:
        # Get the gym's request
        gym_data = request.json
        
        # Convert to JSON-RPC 2.0 format
        json_rpc_request = {
            "jsonrpc": "2.0",
            "method": "process_scenario",  # You might need to adjust this method name
            "params": gym_data,
            "id": str(uuid.uuid4())
        }
        
        # Send to the actual agent
        response = requests.post(
            AGENT_URL,
            json=json_rpc_request,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        # Parse the JSON-RPC response
        json_rpc_response = response.json()
        
        # Check for JSON-RPC error
        if "error" in json_rpc_response:
            # Return a simple error response
            return jsonify({
                "action": "option_a",  # Default action
                "reasoning": f"JSON-RPC Error: {json_rpc_response['error']['message']}",
                "confidence": 0.1
            })
        
        # Extract the result
        if "result" in json_rpc_response:
            result = json_rpc_response["result"]
            
            # If result is already in the expected format
            if isinstance(result, dict) and "action" in result:
                return jsonify(result)
            
            # Otherwise, try to extract action from result
            return jsonify({
                "action": result.get("action", "option_a"),
                "reasoning": result.get("reasoning", str(result)),
                "confidence": result.get("confidence", 0.5)
            })
        
        # Fallback response
        return jsonify({
            "action": "option_a",
            "reasoning": "Unexpected response format",
            "confidence": 0.1
        })
        
    except Exception as e:
        print(f"Adapter error: {str(e)}")
        # Return a valid response even on error
        return jsonify({
            "action": "option_a",
            "reasoning": f"Adapter error: {str(e)}",
            "confidence": 0.1
        })

if __name__ == '__main__':
    print("🌉 JSON-RPC Adapter starting on http://localhost:8090/adapter")
    print("   This will forward requests from the gym to your agent at", AGENT_URL)
    app.run(host='0.0.0.0', port=8090, debug=True)
