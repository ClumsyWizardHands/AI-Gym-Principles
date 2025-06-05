"""Test the new API key behavior in development mode."""
import requests
import json

BASE_URL = "http://localhost:8000"

print("Testing API key behavior in development mode...\n")

# Test 1: Request without API key header
print("1. Testing request without X-API-Key header:")
try:
    response = requests.get(f"{BASE_URL}/api/agents")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
except requests.exceptions.RequestException as e:
    print(f"   Error: {e}")

print()

# Test 2: Request with development API key
print("2. Testing request with development API key:")
headers = {"X-API-Key": "sk-dev-key"}
try:
    response = requests.get(f"{BASE_URL}/api/agents", headers=headers)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
except requests.exceptions.RequestException as e:
    print(f"   Error: {e}")

print()

# Test 3: Request with custom API key (should also work in dev mode)
print("3. Testing request with custom API key:")
headers = {"X-API-Key": "sk-custom-key-123"}
try:
    response = requests.get(f"{BASE_URL}/api/agents", headers=headers)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
except requests.exceptions.RequestException as e:
    print(f"   Error: {e}")

print("\nTest complete!")
