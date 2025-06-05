"""Script to set up API key in production mode."""

import requests
import json

print("🔑 AI Principles Gym - Production Mode API Key Setup")
print("=" * 50)

print("\n⚠️  Important: Production mode requires generating API keys through the API.")
print("The API_KEY in .env is not used for validation in the current implementation.")

print("\n📝 Steps to set up your API key in production mode:")
print("1. Restart the server (the environment has been changed to production)")
print("2. Generate an API key using this script")
print("3. Use the generated key in all API requests")

input("\nPress Enter to continue...")

# Check if server is running
try:
    response = requests.get("http://localhost:8000/api/health", timeout=2)
    print("\n✅ Server is running!")
except:
    print("\n❌ Server is not running. Please restart it with the new production settings.")
    print("   Run: cd ai-principles-gym && python -m uvicorn src.api.app:app --reload")
    exit(1)

# Generate API key
print("\n🔑 Generating API key...")
print("Would you like to set usage limits for this key?")
print("1. No limits (recommended for development)")
print("2. Set usage limit")
print("3. Set expiration time")

choice = input("\nEnter choice (1-3): ").strip()

data = {}
if choice == "2":
    limit = input("Enter usage limit (number of requests): ")
    data["usage_limit"] = int(limit)
elif choice == "3":
    days = input("Enter expiration time in days: ")
    data["expires_in_days"] = int(days)

try:
    response = requests.post(
        "http://localhost:8000/api/keys",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 201:
        result = response.json()
        api_key = result["api_key"]
        
        print("\n✅ API Key Generated Successfully!")
        print(f"\n🔑 Your API Key: {api_key}")
        print(f"   Created at: {result['created_at']}")
        if result.get('expires_at'):
            print(f"   Expires at: {result['expires_at']}")
        if result.get('usage_limit'):
            print(f"   Usage limit: {result['usage_limit']} requests")
        
        # Save to file for convenience
        with open("production_api_key.txt", "w") as f:
            f.write(api_key)
        print("\n💾 API key saved to: production_api_key.txt")
        
        # Update test scripts
        print("\n🔧 Would you like to update test scripts to use this key? (y/n)")
        if input().lower() == 'y':
            # Update the test scripts
            scripts_to_update = [
                "register_decision_wrapper.py",
                "start_training_decision_wrapper.py",
                "test_decision_wrapper.py"
            ]
            
            for script in scripts_to_update:
                try:
                    with open(script, 'r') as f:
                        content = f.read()
                    
                    # Replace the API key
                    content = content.replace('"sk-dev-key"', f'"{api_key}"')
                    
                    with open(script, 'w') as f:
                        f.write(content)
                    
                    print(f"   ✅ Updated {script}")
                except Exception as e:
                    print(f"   ❌ Could not update {script}: {e}")
        
        print("\n📋 How to use your API key:")
        print(f'   Headers: {{"X-API-Key": "{api_key}", "Content-Type": "application/json"}}')
        
    else:
        print(f"\n❌ Failed to generate API key: {response.status_code}")
        print(f"   Response: {response.text}")
        
except Exception as e:
    print(f"\n❌ Error generating API key: {e}")
    print("   Make sure the server is running in production mode.")
