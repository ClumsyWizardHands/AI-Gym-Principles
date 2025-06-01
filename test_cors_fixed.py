#!/usr/bin/env python3
"""Test CORS configuration after fix."""

import requests
import json

def test_cors():
    """Test CORS headers are properly set."""
    
    # Test preflight OPTIONS request
    print("Testing CORS preflight request...")
    
    try:
        response = requests.options(
            'http://localhost:8000/api/agents',
            headers={
                'Origin': 'http://localhost:5173',
                'Access-Control-Request-Method': 'POST',
                'Access-Control-Request-Headers': 'content-type,x-api-key'
            }
        )
        
        print(f"OPTIONS Status: {response.status_code}")
        print("CORS Headers:")
        for header, value in response.headers.items():
            if 'access-control' in header.lower():
                print(f"  {header}: {value}")
        
        # Test actual POST request
        print("\nTesting actual POST request...")
        response = requests.post(
            'http://localhost:8000/api/agents',
            json={
                "name": "Test Agent",
                "framework": "http",
                "config": {
                    "endpoint": "http://localhost:8080/process"
                }
            },
            headers={
                'Origin': 'http://localhost:5173',
                'Content-Type': 'application/json',
                'X-API-Key': 'test-key'
            }
        )
        
        print(f"\nPOST Status: {response.status_code}")
        print("Response Headers:")
        for header, value in response.headers.items():
            if 'access-control' in header.lower():
                print(f"  {header}: {value}")
        
        if response.status_code == 200:
            print("\n✅ CORS is working correctly!")
        else:
            print(f"\n❌ Request failed: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to server. Make sure it's running on http://localhost:8000")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    test_cors()
