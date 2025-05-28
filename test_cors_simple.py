import requests

# Test without origin header
print("Test 1: Request without Origin header")
response = requests.get("http://localhost:8000/health")
print(f"Status: {response.status_code}")
print("Headers with 'access-control':")
for k, v in response.headers.items():
    if 'access-control' in k.lower():
        print(f"  {k}: {v}")
if not any('access-control' in k.lower() for k in response.headers.keys()):
    print("  No access-control headers found")

print("\n" + "="*50 + "\n")

# Test with origin header
print("Test 2: Request with Origin header")
response = requests.get("http://localhost:8000/health", headers={"Origin": "http://localhost:5173"})
print(f"Status: {response.status_code}")
print("Headers with 'access-control':")
for k, v in response.headers.items():
    if 'access-control' in k.lower():
        print(f"  {k}: {v}")
if not any('access-control' in k.lower() for k in response.headers.keys()):
    print("  No access-control headers found")

print("\n" + "="*50 + "\n")

# Test OPTIONS preflight
print("Test 3: OPTIONS preflight request")
response = requests.options("http://localhost:8000/health", headers={
    "Origin": "http://localhost:5173",
    "Access-Control-Request-Method": "GET"
})
print(f"Status: {response.status_code}")
print("Headers with 'access-control':")
for k, v in response.headers.items():
    if 'access-control' in k.lower():
        print(f"  {k}: {v}")
if not any('access-control' in k.lower() for k in response.headers.keys()):
    print("  No access-control headers found")
