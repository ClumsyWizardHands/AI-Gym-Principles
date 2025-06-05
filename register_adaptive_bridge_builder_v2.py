"""Register the Adaptive Bridge Builder agent and start a thorough training session."""

import requests
import json
import time
from datetime import datetime

# Configuration
API_KEY = "sk-dev-key"  # Development mode accepts any key
GYM_URL = "http://localhost:8000"
AGENT_URL = "http://localhost:8080/process"  # Updated with /process endpoint

# Headers for API requests
headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

def register_agent():
    """Register the Adaptive Bridge Builder agent."""
    print("📝 Registering Adaptive Bridge Builder Agent...")
    print(f"   Agent ID: 4f397af3-4816-42c4-bd3b-4262083b6964")
    print(f"   Endpoint: {AGENT_URL}")
    
    registration_data = {
        "name": "Adaptive Bridge Builder",
        "framework": "http",
        "config": {
            "endpoint_url": AGENT_URL,  # Using /process endpoint
            "method": "POST",
            "headers": {
                "Content-Type": "application/json"
            },
            "timeout": 30,
            "request_format": "json",
            "response_format": "json"
        },
        "description": "An agent designed to facilitate communication and collaboration between different agents and systems using the A2A Protocol. It embodies the 'Empire of the Adaptive Hero' profile, serving as a connector that adapts to various communication needs while maintaining core principles of fairness, harmony, and adaptability."
    }
    
    try:
        response = requests.post(
            f"{GYM_URL}/api/agents/register",
            json=registration_data,
            headers=headers
        )
        
        if response.status_code == 201:
            result = response.json()
            agent_id = result["agent_id"]
            print(f"✅ Agent registered successfully!")
            print(f"   Gym Agent ID: {agent_id}")
            print(f"   Name: {result['name']}")
            print(f"   Framework: {result['framework']}")
            
            # Save agent ID for future use
            with open("adaptive_bridge_builder_agent_id.txt", "w") as f:
                f.write(agent_id)
                
            return agent_id
        else:
            print(f"❌ Failed to register agent: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error registering agent: {e}")
        return None

def start_training(agent_id):
    """Start a comprehensive training session."""
    print("\n🚀 Starting Thorough Training Session")
    print("=" * 50)
    
    # Training configuration - comprehensive session
    training_config = {
        "agent_id": agent_id,
        "scenario_types": [
            "LOYALTY",
            "SCARCITY", 
            "BETRAYAL",
            "TRADEOFFS",
            "TIME_PRESSURE",
            "OBEDIENCE_AUTONOMY",
            "INFO_ASYMMETRY",
            "REPUTATION_MGMT",
            "POWER_DYNAMICS",
            "MORAL_HAZARD"
        ],
        "num_scenarios": 50,  # Thorough training with 50 scenarios
        "adaptive": True,     # Enable adaptive difficulty
        "use_branching": True,  # Include branching scenarios
        "branching_types": ["trust_building", "resource_cascade"]
    }
    
    print(f"\n📋 Training Configuration:")
    print(f"   - Agent ID: {agent_id}")
    print(f"   - Scenarios: {training_config['num_scenarios']}")
    print(f"   - Scenario Types: {', '.join(training_config['scenario_types'])}")
    print(f"   - Adaptive Difficulty: {'Yes' if training_config['adaptive'] else 'No'}")
    print(f"   - Branching Scenarios: {'Yes' if training_config['use_branching'] else 'No'}")
    
    # Start training session
    try:
        response = requests.post(
            f"{GYM_URL}/api/training/start",
            json=training_config,
            headers=headers
        )
        
        if response.status_code == 202:
            result = response.json()
            session_id = result["session_id"]
            print(f"\n✅ Training Session Started!")
            print(f"   Session ID: {session_id}")
            print(f"   Status: {result['status']}")
            print(f"   Estimated Duration: {result['estimated_duration_seconds']} seconds")
            
            return session_id
        else:
            print(f"\n❌ Failed to start training: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"\n❌ Error starting training: {e}")
        return None

def monitor_training(session_id):
    """Monitor training progress."""
    print(f"\n📊 Monitoring Training Progress...")
    print("   Press Ctrl+C to stop monitoring (training will continue)")
    
    last_progress = -1
    
    try:
        while True:
            try:
                response = requests.get(
                    f"{GYM_URL}/api/training/status/{session_id}",
                    headers=headers
                )
                
                if response.status_code == 200:
                    status = response.json()
                    progress = int(status["progress"] * 100)
                    
                    # Only update if progress changed
                    if progress != last_progress:
                        print(f"\r   Progress: {progress}% | "
                              f"Scenarios: {status['scenarios_completed']}/{status['scenarios_total']} | "
                              f"Status: {status['status']}", end="", flush=True)
                        last_progress = progress
                    
                    if status["status"] == "completed":
                        print(f"\n\n🎉 Training Completed Successfully!")
                        return True
                    elif status["status"] == "failed":
                        print(f"\n\n❌ Training Failed: {status.get('error_message', 'Unknown error')}")
                        return False
                        
                time.sleep(2)  # Check every 2 seconds
                
            except KeyboardInterrupt:
                print(f"\n\n⚠️  Monitoring stopped. Training continues in background.")
                print(f"   Session ID: {session_id}")
                return None
                
    except Exception as e:
        print(f"\n\n❌ Error monitoring training: {e}")
        return False

def get_report(session_id):
    """Get the training report."""
    print(f"\n📈 Fetching Training Report...")
    
    try:
        response = requests.get(
            f"{GYM_URL}/api/reports/{session_id}",
            headers=headers
        )
        
        if response.status_code == 200:
            report = response.json()
            
            print("\n" + "=" * 60)
            print("🏆 TRAINING REPORT - Adaptive Bridge Builder")
            print("=" * 60)
            
            print(f"\n📊 Overview:")
            print(f"   - Session ID: {report['session_id']}")
            print(f"   - Duration: {report['duration_seconds']:.1f} seconds")
            print(f"   - Scenarios Completed: {report['scenarios_completed']}")
            print(f"   - Behavioral Entropy: {report['behavioral_entropy']:.3f}")
            print(f"   - Consistency Score: {report['consistency_score']:.3f}")
            
            print(f"\n🧠 Discovered Principles ({len(report['principles_discovered'])}):")
            for i, principle in enumerate(report['principles_discovered'], 1):
                print(f"\n   {i}. {principle['name']}")
                print(f"      Description: {principle['description']}")
                print(f"      Strength: {principle['strength']:.3f}")
                print(f"      Consistency: {principle['consistency']:.3f}")
                print(f"      Evidence Count: {principle['evidence_count']}")
                print(f"      Contexts: {', '.join(principle['contexts'])}")
            
            print(f"\n📝 Summary:")
            print(f"   {report['summary']}")
            
            # Save report to file
            filename = f"adaptive_bridge_builder_report_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n💾 Full report saved to: {filename}")
            
        else:
            print(f"\n❌ Failed to get report: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"\n❌ Error getting report: {e}")

def main():
    """Main function to register agent and run training session."""
    print("🎯 AI Principles Gym - Adaptive Bridge Builder Testing")
    print("=" * 60)
    print("   Agent: Adaptive Bridge Builder")
    print("   Internal ID: 4f397af3-4816-42c4-bd3b-4262083b6964")
    print("   Endpoint: http://localhost:8080/process")
    print("   Description: Empire of the Adaptive Hero")
    print("=" * 60)
    
    # Check if agent is accessible
    print("\n🔍 Checking agent health...")
    try:
        health_response = requests.get("http://localhost:8080/health", timeout=2)
        if health_response.status_code == 200:
            print("✅ Agent health check passed")
            print(f"   Response: {health_response.json()}")
        else:
            print(f"⚠️  Agent health check returned status {health_response.status_code}")
    except Exception as e:
        print(f"⚠️  Could not check agent health: {e}")
        print("   Proceeding anyway...")
    
    # Register agent
    agent_id = register_agent()
    
    if agent_id:
        # Brief pause before starting training
        time.sleep(2)
        
        # Start training
        session_id = start_training(agent_id)
        
        if session_id:
            # Monitor progress
            completed = monitor_training(session_id)
            
            # Get report if completed or if user wants it anyway
            if completed is not False:  # True or None (stopped monitoring)
                time.sleep(2)  # Brief pause before getting report
                get_report(session_id)
    
    print("\n✨ Testing session complete!")

if __name__ == "__main__":
    main()
