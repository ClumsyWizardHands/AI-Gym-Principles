"""Run a full training session using the mock adapter for testing."""

import requests
import json
import time

# First, let's register a test agent using the mock adapter
print("📝 Registering test agent with mock adapter...")

register_url = "http://localhost:8000/api/agents/register"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "sk-dev-key"
}

# Register with mock adapter for testing
agent_data = {
    "name": "Bridge Builder Test Agent",
    "framework": "mock",
    "config": {
        "response_pattern": "balanced"  # Will make balanced decisions
    },
    "description": "Test agent for demonstrating full training flow"
}

try:
    response = requests.post(register_url, headers=headers, json=agent_data)
    if response.status_code == 201:
        result = response.json()
        agent_id = result['agent_id']
        print(f"✅ Agent registered! ID: {agent_id}")
    else:
        print(f"❌ Registration failed: {response.text}")
        exit(1)
except Exception as e:
    print(f"❌ Error: {str(e)}")
    exit(1)

# Now start the training session
print("\n🎯 Starting training session...")

training_url = "http://localhost:8000/api/training/start"
training_data = {
    "agent_id": agent_id,
    "num_scenarios": 10,  # Run 10 scenarios
    "adaptive": True,
    "scenario_types": ["resource_allocation", "cooperation", "trust_building"]
}

try:
    response = requests.post(training_url, headers=headers, json=training_data)
    if response.status_code == 202:
        result = response.json()
        session_id = result['session_id']
        print(f"✅ Training started! Session ID: {session_id}")
        print(f"   Estimated duration: {result['estimated_duration_seconds']} seconds")
        
        # Monitor the training progress
        print("\n📊 Monitoring training progress...")
        status_url = f"http://localhost:8000/api/training/status/{session_id}"
        
        completed = False
        while not completed:
            time.sleep(2)  # Check every 2 seconds
            try:
                status_response = requests.get(status_url, headers=headers)
                if status_response.status_code == 200:
                    status = status_response.json()
                    print(f"   Status: {status['status']} - Scenarios: {status['scenarios_completed']}/{status['total_scenarios']}")
                    
                    if status['status'] in ['completed', 'failed']:
                        completed = True
                        
                        if status['status'] == 'completed':
                            print("\n🎉 Training completed successfully!")
                            
                            # Get the full results
                            results_url = f"http://localhost:8000/api/training/results/{session_id}"
                            results_response = requests.get(results_url, headers=headers)
                            
                            if results_response.status_code == 200:
                                results = results_response.json()
                                
                                print("\n📈 Training Results:")
                                print(f"   Total Scenarios: {results['summary']['total_scenarios']}")
                                print(f"   Average Confidence: {results['summary']['average_confidence']:.2f}")
                                
                                if 'inferred_principles' in results:
                                    print("\n🧠 Inferred Principles:")
                                    for i, principle in enumerate(results['inferred_principles'], 1):
                                        print(f"   {i}. {principle['principle']}")
                                        print(f"      Confidence: {principle['confidence']:.2f}")
                                        print(f"      Evidence: {', '.join(principle['evidence'][:3])}")
                                
                                if 'behavioral_patterns' in results:
                                    print("\n🔄 Behavioral Patterns:")
                                    for pattern in results['behavioral_patterns'][:3]:
                                        print(f"   - {pattern['pattern']}: {pattern['frequency']} occurrences")
                        else:
                            print(f"\n❌ Training failed: {status.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"   Error checking status: {str(e)}")
        
        print(f"\n🌐 View detailed results at: http://localhost:5173/training/{session_id}")
        
    else:
        print(f"❌ Failed to start training: {response.text}")
        
except Exception as e:
    print(f"❌ Error: {str(e)}")
