"""Test if the API server is running and accessible."""
import requests
import sys

def check_server_status():
    """Check if the API server is running."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server is running!")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"❌ API server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ API server is not running or not accessible at http://localhost:8000")
        print("Please start the server with: python -m uvicorn src.api.app:app --reload")
        return False
    except Exception as e:
        print(f"❌ Error checking server status: {e}")
        return False

if __name__ == "__main__":
    is_running = check_server_status()
    sys.exit(0 if is_running else 1)
