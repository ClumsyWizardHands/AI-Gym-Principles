#!/usr/bin/env python3
"""
Test HTTP agent integration using the AI Principles Gym API.
This script uses the REST API to:
1. Register your HTTP agent
2. Create a training session
3. Run scenarios
4. Get results
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
            "X-API-Key": "test-key"  # Add your API key if needed
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
                f"{self.base_url}/api/agents",
                json={
                    "name": name,
                    "framework": framework,
                    "config": config
                }
            )
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to register agent: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Error registering agent: {e}")
            return None
    
    def create_training_session(self, agent_id: str, scenario_type: str, num_scenarios: int) -> Optional[Dict]:
        """Create a new training session."""
        try:
            response = self.session.post(
                f"{self.base_url}/api/training/sessions",
                json={
                    "agent_id": agent_id,
                    "scenario_type": scenario_type,
                    "num_scenarios": num_scenarios,
                    "config": {
                        "include_ethical": True,
                        "include_strategic": True,
                        "randomize": True
                    }
                }
            )
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to create session: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Error creating session: {e}")
            return None
    
    def start_training(self, session_id: str) -> bool:
        """Start a training session."""
        try:
            response = self.session.post(
                f"{self.base_url}/api/training/sessions/{session_id}/start"
            )
            return response.status_code == 200
        except:
            return False
    
    def get_session_status(self, session_id: str) -> Optional[Dict]:
        """Get training session status."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/training/sessions/{session_id}"
            )
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def get_session_report(self, session_id: str) -> Optional[Dict]:
        """Get training session report."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/training/sessions/{session_id}/report"
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
            data = response.json()
            print(f"✅ Agent responded: {data}")
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
    print("🎯 AI Principles Gym - HTTP Agent Integration Test")
    print("="*60)
    
    # Configuration
    AGENT_ENDPOINT = "http://localhost:8080/process"
    AGENT_NAME = "My HTTP Agent"
    NUM_SCENARIOS = 10
    
    # Check if HTTP agent is running
    if not test_http_agent_connection(AGENT_ENDPOINT):
        print("\n⚠️ Please make sure your HTTP agent is running at", AGENT_ENDPOINT)
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
    
    agent_id = agent_response["id"]
    print(f"✅ Agent registered with ID: {agent_id}")
    
    # Create training session
    print(f"\n🎮 Creating training session with {NUM_SCENARIOS} scenarios...")
    session_response = client.create_training_session(
        agent_id=agent_id,
        scenario_type="mixed",
        num_scenarios=NUM_SCENARIOS
    )
    
    if not session_response:
        print("❌ Failed to create training session")
        return
    
    session_id = session_response["id"]
    print(f"✅ Training session created with ID: {session_id}")
    
    # Start training
    print("\n🚀 Starting training...")
    if not client.start_training(session_id):
        print("❌ Failed to start training")
        return
    
    print("✅ Training started!")
    
    # Monitor progress
    print("\n📊 Monitoring training progress...")
    print("-" * 60)
    
    last_progress = 0
    while True:
        status = client.get_session_status(session_id)
        if not status:
            print("❌ Failed to get session status")
            break
        
        progress = status.get("progress", 0)
        state = status.get("status", "unknown")
        
        # Update progress bar
        if progress > last_progress:
            bar_length = 40
            filled_length = int(bar_length * progress)
            bar = "█" * filled_length + "-" * (bar_length - filled_length)
            print(f"\rProgress: [{bar}] {progress*100:.1f}% - {state}", end="", flush=True)
            last_progress = progress
        
        if state == "completed":
            print("\n✅ Training completed!")
            break
        elif state == "failed":
            print("\n❌ Training failed!")
            break
        
        await asyncio.sleep(2)  # Check every 2 seconds
    
    # Get results
    print("\n📈 Retrieving training report...")
    report = client.get_session_report(session_id)
    
    if not report:
        print("❌ Failed to get training report")
        return
    
    # Display results
    print("\n" + "="*60)
    print("📊 TRAINING RESULTS")
    print("="*60)
    
    print(f"\n📋 Summary:")
    print(f"  • Session ID: {session_id}")
    print(f"  • Agent: {AGENT_NAME} ({agent_id})")
    print(f"  • Scenarios completed: {report.get('scenarios_completed', 0)}")
    print(f"  • Total decisions: {report.get('total_decisions', 0)}")
    
    if "behavioral_principles" in report:
        principles = report["behavioral_principles"]
        
        if "ethical" in principles:
            print("\n⚖️ Ethical Principles:")
            for principle in principles["ethical"]:
                print(f"  • {principle['description']} (strength: {principle['strength']:.2f})")
        
        if "strategic" in principles:
            print("\n♟️ Strategic Principles:")
            for principle in principles["strategic"]:
                print(f"  • {principle['description']} (strength: {principle['strength']:.2f})")
    
    if "consistency_metrics" in report:
        metrics = report["consistency_metrics"]
        print("\n📈 Consistency Metrics:")
        print(f"  • Overall: {metrics.get('overall', 0):.2%}")
        print(f"  • Ethical: {metrics.get('ethical', 0):.2%}")
        print(f"  • Strategic: {metrics.get('strategic', 0):.2%}")
    
    if "behavioral_entropy" in report:
        entropy = report["behavioral_entropy"]
        print(f"\n🔬 Behavioral Entropy: {entropy:.3f}")
        if entropy < 0.3:
            print("  → Highly consistent/principled behavior")
        elif entropy < 0.7:
            print("  → Moderately consistent behavior")
        else:
            print("  → Inconsistent/random behavior")
    
    print("\n" + "="*60)
    print("✅ Test completed successfully!")
    print(f"🌐 View detailed results in the web UI: http://localhost:5173/sessions/{session_id}")
    

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
