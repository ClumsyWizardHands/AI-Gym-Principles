import requests

# Test CORS for the /api/keys endpoint
url = "http://localhost:8000/api/keys"
headers = {
    "Origin": "http://localhost:5173",
    "Content-Type": "application/json"
}

print("Testing CORS for /api/keys endpoint...")
print("=" * 50)

# Test OPTIONS request (preflight)
print("\n1. Testing OPTIONS request (preflight):")
try:
    response = requests.options(
        url, 
        headers={
            **headers,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type,X-API-Key"
        }
    )
    print(f"Status Code: {response.status_code}")
    print("Response Headers:")
    cors_headers = {k: v for k, v in response.headers.items() if k.lower().startswith('access-control')}
    if cors_headers:
        for header, value in cors_headers.items():
            print(f"  {header}: {value}")
    else:
        print("  No CORS headers found!")
    
    # Check for the critical header
    if 'access-control-allow-origin' not in [h.lower() for h in response.headers.keys()]:
        print("  ⚠️  Missing Access-Control-Allow-Origin header!")
except Exception as e:
    print(f"Error: {e}")

# Test POST request
print("\n2. Testing POST request:")
try:
    response = requests.post(url, headers=headers, json={})
    print(f"Status Code: {response.status_code}")
    print("Response Headers:")
    cors_headers = {k: v for k, v in response.headers.items() if k.lower().startswith('access-control')}
    if cors_headers:
        for header, value in cors_headers.items():
            print(f"  {header}: {value}")
    else:
        print("  No CORS headers found!")
    
    # Check for the critical header
    if 'access-control-allow-origin' not in [h.lower() for h in response.headers.keys()]:
        print("  ⚠️  Missing Access-Control-Allow-Origin header!")
    
    print(f"Response Body: {response.text[:200]}")
except Exception as e:
    print(f"Error: {e}")

# Test GET request to /api/agents
print("\n3. Testing GET request to /api/agents:")
try:
    response = requests.get(
        "http://localhost:8000/api/agents",
        headers=headers
    )
    print(f"Status Code: {response.status_code}")
    print("Response Headers:")
    cors_headers = {k: v for k, v in response.headers.items() if k.lower().startswith('access-control')}
    if cors_headers:
        for header, value in cors_headers.items():
            print(f"  {header}: {value}")
    else:
        print("  No CORS headers found!")
    
    # Check for the critical header
    if 'access-control-allow-origin' not in [h.lower() for h in response.headers.keys()]:
        print("  ⚠️  Missing Access-Control-Allow-Origin header!")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 50)
print("CORS test complete.")

# Summary
print("\nSummary:")
print("If you see 'Missing Access-Control-Allow-Origin header!' warnings above,")
print("the CORS configuration is not working properly and browsers will block requests.")
