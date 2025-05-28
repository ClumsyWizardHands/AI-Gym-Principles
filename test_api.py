import requests

print("Testing API endpoints...")

# Test health endpoint
try:
    response = requests.get("http://localhost:8000/api/health")
    print(f"Health check: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Health check failed: {e}")

# Test agents endpoint
try:
    response = requests.get("http://localhost:8000/api/agents")
    print(f"\nAgents endpoint: {response.status_code}")
    if response.status_code != 200:
        print(f"Response: {response.text}")
except Exception as e:
    print(f"Agents endpoint failed: {e}")

# Check CORS headers
try:
    response = requests.options("http://localhost:8000/api/agents", 
                               headers={"Origin": "http://localhost:5174"})
    print(f"\nCORS check: {response.status_code}")
    print(f"Access-Control-Allow-Origin: {response.headers.get('Access-Control-Allow-Origin')}")
    print(f"Access-Control-Allow-Methods: {response.headers.get('Access-Control-Allow-Methods')}")
except Exception as e:
    print(f"CORS check failed: {e}")
