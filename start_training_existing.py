"""Start a training session with the already registered Decision Wrapper agent."""

import requests
import json
import time

# We already have the Decision Wrapper agent ID
agent_id = "aca139be-317c-4991-8b82-a12d438e34fa"

print(f"🎯 Starting training session for Decision Wrapper agent...")
print(f"   Agent ID: {agent_id}")

# API endpoint
url = "http://localhost:8000/api/training/start"

# Headers
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "sk-dev-key"
}

# Training configuration
training_data = {
    "agent_id": agent_id,
    "num_scenarios": 5,
    "adaptive": True
}

try:
    # Start the training session
    response = requests.post(url, headers=headers, json=training_data)
    
    if response.status_code == 202:  # Accepted
        result = response.json()
        session_id = result['session_id']
        print(f"\n✅ Training session started successfully!")
        print(f"   Session ID: {session_id}")
        print(f"   Status: {result['status']}")
        print(f"   Started at: {result['started_at']}")
        print(f"   Estimated duration: {result['estimated_duration_seconds']} seconds")
        
        # Save session ID
        with open("current_session_id.txt", "w") as f:
            f.write(session_id)
        
        # Monitor the training progress
        print("\n📊 Monitoring training progress...")
        status_url = f"http://localhost:8000/api/training/status/{session_id}"
        
        completed = False
        scenarios_completed = 0
        
        while not completed:
            time.sleep(2)  # Check every 2 seconds
            try:
                status_response = requests.get(status_url, headers=headers)
                if status_response.status_code == 200:
                    status = status_response.json()
                    
                    # Only print if there's progress
                    if status['scenarios_completed'] > scenarios_completed:
                        scenarios_completed = status['scenarios_completed']
                        print(f"   Progress: {scenarios_completed}/{status['total_scenarios']} scenarios - Status: {status['status']}")
                    
                    if status['status'] in ['completed', 'failed']:
                        completed = True
                        
                        if status['status'] == 'completed':
                            print("\n🎉 Training completed successfully!")
                            
                            # Get the results
                            results_url = f"http://localhost:8000/api/training/results/{session_id}"
                            results_response = requests.get(results_url, headers=headers)
                            
                            if results_response.status_code == 200:
                                results = results_response.json()
                                
                                print("\n📈 Training Summary:")
                                print(f"   Total Scenarios: {results['summary']['total_scenarios']}")
                                print(f"   Average Confidence: {results['summary']['average_confidence']:.2f}")
                                
                                if 'scenario_results' in results and results['scenario_results']:
                                    print("\n🎮 Sample Decisions:")
                                    for i, scenario in enumerate(results['scenario_results'][:3], 1):
                                        print(f"   {i}. {scenario['scenario_type']} - Action: {scenario['action_taken']}")
                                        print(f"      Reasoning: {scenario['reasoning']}")
                                
                                print(f"\n🌐 View detailed results at: http://localhost:5173/training/{session_id}")
                        else:
                            print(f"\n❌ Training failed: {status.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"   Error checking status: {str(e)}")
                
    else:
        print(f"\n❌ Failed to start training session. Status code: {response.status_code}")
        print(f"Response: {response.text}")
        
        # If it's a validation error, show details
        if response.status_code == 422:
            try:
                error_details = response.json()
                print("\nValidation errors:")
                for error in error_details.get('detail', []):
                    print(f"  - {error.get('loc', [])}: {error.get('msg', '')}")
            except:
                pass
        
except requests.exceptions.ConnectionError:
    print("❌ Could not connect to the AI Principles Gym API")
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
