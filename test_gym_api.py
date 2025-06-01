#!/usr/bin/env python3
"""
Test HTTP agent integration using the correct AI Principles Gym API endpoints.
"""

import asyncio
import json
import time
import requests
from typing import Dict, Any, Optional


class GymAPIClient:
    """Client for interacting with AI Principles Gym API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "X-API-Key": "test-key"
        })
    
    def test_connection(self) -> bool:
        """Test connection to the gym API."""
        try:
            response = self.session.get(f"{self.base_url}/api/health")
            return response.status_code == 200
        except:
            return False
    
    def register_agent(self, name: str, framework: str, config: Dict[str, Any]) -> Optional[Dict]:
        """Register an agent with the gym."""
        try:
            response = self.session.post(
                f"{self.base_url}/api/agents/register",  # Correct endpoint
                json={
                    "name": name,
                    "framework": framework,
                    "config": config
                }
            )
            if response.status_code == 201:  # 201 Created
                return response.json()
            else:
                print(f"Failed to register agent: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Error registering agent: {e}")
            return None
    
    def start_training(self, agent_id: str, num_scenarios: int = 10) -> Optional[Dict]:
        """Start a training session."""
        try:
            response = self.session.post(
                f"{self.base_url}/api/training/start",
                json={
                    "agent_id": agent_id,
                    "scenario_types": [],  # Empty = all types
                    "num_scenarios": num_scenarios,
                    "adaptive": True,
                    "use_branching": False
                }
            )
            if response.status_code == 202:  # 202 Accepted
                return response.json()
            else:
                print(f"Failed to start training: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Error starting training: {e}")
            return None
    
    def get_training_status(self, session_id: str) -> Optional[Dict]:
        """Get training session status."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/training/status/{session_id}"
            )
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def get_training_report(self, session_id: str) -> Optional[Dict]:
        """Get training session report."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/reports/{session_id}"
            )
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None


def test_http_agent_connection(endpoint: str) -> bool:
    """Test if HTTP agent is responding."""
    print(f"\n📡 Testing connection to HTTP agent at {endpoint}...")
    
    test_request = {
        "scenario": {
            "id": "test",
            "description": "Connection test",
            "choice_options": [
                {"id": "option1", "description": "Test option"}
            ]
        },
        "history": []
    }
    
    try:
        response = requests.post(endpoint, json=test_request, timeout=5)
        if response.status_code == 200:
            print(f"✅ Agent responded")
            return True
        else:
            print(f"❌ Agent returned status {response.status_code}: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to agent at {endpoint}")
        return False
    except Exception as e:
        print(f"❌ Error testing agent: {e}")
        return False


async def main():
    """Main test function."""
    print("🎯 AI Principles Gym - HTTP Agent Full Integration Test")
    print("="*60)
    
    # Configuration
    AGENT_ENDPOINT = "http://localhost:8080/process"
    AGENT_NAME = "My HTTP Agent"
    NUM_SCENARIOS = 10
    
    # Check if HTTP agent is running
    if not test_http_agent_connection(AGENT_ENDPOINT):
        print("\n⚠️ Please make sure your HTTP agent is running at", AGENT_ENDPOINT)
        print("The agent should accept POST requests with scenario data.")
        return
    
    # Initialize API client
    print("\n🏋️ Connecting to AI Principles Gym...")
    client = GymAPIClient()
    
    if not client.test_connection():
        print("❌ Cannot connect to AI Principles Gym API at", client.base_url)
        print("Make sure the gym is running (use dev-start.bat)")
        return
    
    print("✅ Connected to AI Principles Gym")
    
    # Register agent
    print(f"\n📝 Registering agent '{AGENT_NAME}'...")
    agent_response = client.register_agent(
        name=AGENT_NAME,
        framework="http",
        config={
            "endpoint": AGENT_ENDPOINT,
            "timeout": 30
        }
    )
    
    if not agent_response:
        print("❌ Failed to register agent")
        return
    
    agent_id = agent_response["agent_id"]
    print(f"✅ Agent registered with ID: {agent_id}")
    
    # Start training
    print(f"\n🎮 Starting training session with {NUM_SCENARIOS} scenarios...")
    training_response = client.start_training(
        agent_id=agent_id,
        num_scenarios=NUM_SCENARIOS
    )
    
    if not training_response:
        print("❌ Failed to start training")
        return
    
    session_id = training_response["session_id"]
    print(f"✅ Training session started with ID: {session_id}")
    print(f"   Estimated duration: {training_response['estimated_duration_seconds']} seconds")
    
    # Monitor progress
    print("\n📊 Monitoring training progress...")
    print("-" * 60)
    
    last_progress = 0
    while True:
        status = client.get_training_status(session_id)
        if not status:
            print("\n❌ Failed to get session status")
            break
        
        progress = status.get("progress", 0)
        state = status.get("status", "unknown")
        scenarios_completed = status.get("scenarios_completed", 0)
        scenarios_total = status.get("scenarios_total", NUM_SCENARIOS)
        
        # Update progress bar
        if progress > last_progress or state != "running":
            bar_length = 40
            filled_length = int(bar_length * progress)
            bar = "█" * filled_length + "-" * (bar_length - filled_length)
            print(f"\rProgress: [{bar}] {progress*100:.1f}% ({scenarios_completed}/{scenarios_total}) - {state}", end="", flush=True)
            last_progress = progress
        
        if state == "completed":
            print("\n✅ Training completed!")
            break
        elif state == "failed":
            print(f"\n❌ Training failed: {status.get('error_message', 'Unknown error')}")
            return
        
        await asyncio.sleep(2)  # Check every 2 seconds
    
    # Get results
    print("\n📈 Retrieving training report...")
    report = client.get_training_report(session_id)
    
    if not report:
        print("❌ Failed to get training report")
        print("Note: Report may not be available immediately after completion.")
        return
    
    # Display results
    print("\n" + "="*60)
    print("📊 TRAINING RESULTS")
    print("="*60)
    
    print(f"\n📋 Summary:")
    print(f"  • Session ID: {session_id}")
    print(f"  • Agent: {AGENT_NAME} ({agent_id})")
    print(f"  • Scenarios completed: {report.get('scenarios_completed', 0)}")
    print(f"  • Duration: {report.get('duration_seconds', 0):.1f} seconds")
    
    if "principles_discovered" in report:
        principles = report["principles_discovered"]
        if principles:
            print(f"\n🧠 DISCOVERED BEHAVIORAL PRINCIPLES ({len(principles)} total):")
            print("-" * 60)
            
            for principle in principles:
                print(f"\n  📌 {principle['name']}")
                print(f"     {principle['description']}")
                print(f"     Strength: {principle['strength']:.2f} | Consistency: {principle['consistency']:.2f}")
                print(f"     Evidence: {principle['evidence_count']} observations")
                if principle.get('contexts'):
                    print(f"     Contexts: {', '.join(principle['contexts'])}")
    
    if "behavioral_entropy" in report:
        entropy = report["behavioral_entropy"]
        print(f"\n🔬 Behavioral Analysis:")
        print(f"  • Entropy: {entropy:.3f}", end=" ")
        if entropy < 0.3:
            print("(Highly principled/consistent)")
        elif entropy < 0.7:
            print("(Moderately consistent)")
        else:
            print("(Inconsistent/exploratory)")
    
    if "consistency_score" in report:
        print(f"  • Overall Consistency: {report['consistency_score']:.2%}")
    
    if "summary" in report:
        print(f"\n📝 Summary:")
        print(f"  {report['summary']}")
    
    print("\n" + "="*60)
    print("✅ Test completed successfully!")
    print(f"🌐 View detailed results in the web UI: http://localhost:5173")
    

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
