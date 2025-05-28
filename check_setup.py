#!/usr/bin/env python3
"""
Quick diagnostic script to check if the Principle AI Gym is set up correctly.
"""

import sys
import subprocess
import socket
import json
import os
from pathlib import Path

def check_port(host, port):
    """Check if a port is open."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        return result == 0

def check_backend():
    """Check if backend is running."""
    print("🔍 Checking backend (API)...")
    if check_port('localhost', 8000):
        print("✅ Backend is running on http://localhost:8000")
        # Try to hit the health endpoint
        try:
            import requests
            response = requests.get('http://localhost:8000/api/health', timeout=2)
            if response.status_code == 200:
                print("✅ Backend health check passed")
            else:
                print(f"⚠️  Backend returned status {response.status_code}")
        except Exception as e:
            print(f"⚠️  Could not reach backend health endpoint: {e}")
    else:
        print("❌ Backend is NOT running")
        print("   Run: cd ai-principles-gym && python -m src.api.app")
    return check_port('localhost', 8000)

def check_frontend():
    """Check if frontend is running."""
    print("\n🔍 Checking frontend...")
    if check_port('localhost', 5173):
        print("✅ Frontend is running on http://localhost:5173")
    else:
        print("❌ Frontend is NOT running")
        print("   Run: cd ai-principles-gym/frontend && npm run dev")
    return check_port('localhost', 5173)

def check_database():
    """Check if database file exists."""
    print("\n🔍 Checking database...")
    db_path = Path("data/gym.db")
    if db_path.exists():
        print(f"✅ Database exists at {db_path}")
    else:
        print("⚠️  Database does not exist (will be created on first run)")
    return True

def check_env_files():
    """Check if environment files exist."""
    print("\n🔍 Checking environment files...")
    backend_env = Path(".env")
    frontend_env = Path("frontend/.env")
    
    if backend_env.exists():
        print("✅ Backend .env file exists")
    else:
        print("❌ Backend .env file missing")
        print("   Copy from .env.example: cp .env.example .env")
    
    if frontend_env.exists():
        print("✅ Frontend .env file exists")
        # Check API URL
        with open(frontend_env, 'r') as f:
            content = f.read()
            if 'VITE_API_URL' in content:
                print("✅ VITE_API_URL is configured")
            else:
                print("⚠️  VITE_API_URL not found in frontend/.env")
    else:
        print("❌ Frontend .env file missing")
        print("   Create with: echo 'VITE_API_URL=http://localhost:8000' > frontend/.env")
    
    return backend_env.exists() and frontend_env.exists()

def check_http_adapter():
    """Check if HTTP adapter files exist."""
    print("\n🔍 Checking HTTP adapter implementation...")
    files_to_check = [
        ("src/adapters/http_adapter.py", "HTTP adapter implementation"),
        ("frontend/src/api/endpoints.ts", "Frontend endpoints (with 'http' option)"),
        ("frontend/src/pages/AgentsPage.tsx", "Agent registration page (with HTTP fields)"),
        ("frontend/src/api/types.ts", "TypeScript types (with 'http' framework)")
    ]
    
    all_good = True
    for file_path, description in files_to_check:
        path = Path(file_path)
        if path.exists():
            print(f"✅ {description} exists")
            # Check for 'http' in the file
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'http' in content.lower():
                    print(f"   └─ Contains 'http' references ✓")
                else:
                    print(f"   └─ ⚠️  No 'http' references found")
                    all_good = False
        else:
            print(f"❌ {description} missing!")
            all_good = False
    
    return all_good

def check_node_modules():
    """Check if node modules are installed."""
    print("\n🔍 Checking frontend dependencies...")
    node_modules = Path("frontend/node_modules")
    if node_modules.exists() and node_modules.is_dir():
        print("✅ Node modules installed")
    else:
        print("❌ Node modules NOT installed")
        print("   Run: cd frontend && npm install")
    return node_modules.exists()

def check_python_deps():
    """Check if Python dependencies are installed."""
    print("\n🔍 Checking Python dependencies...")
    try:
        import fastapi
        import sqlalchemy
        import structlog
        import aiohttp
        print("✅ Core Python dependencies installed")
        return True
    except ImportError as e:
        print(f"❌ Missing Python dependency: {e}")
        print("   Run: pip install -r requirements.txt")
        return False

def main():
    """Run all checks."""
    print("🏋️ Principle AI Gym Diagnostic Tool")
    print("=" * 50)
    
    checks = [
        ("Environment files", check_env_files),
        ("Python dependencies", check_python_deps),
        ("Node modules", check_node_modules),
        ("Database", check_database),
        ("HTTP adapter", check_http_adapter),
        ("Backend server", check_backend),
        ("Frontend server", check_frontend),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            results.append((name, check_func()))
        except Exception as e:
            print(f"❌ Error checking {name}: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 50)
    print("📊 Summary:")
    all_passed = all(result[1] for result in results)
    
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"{status} {name}")
    
    if all_passed:
        print("\n🎉 Everything looks good!")
        print("You should be able to access the app at http://localhost:5173")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\n🚀 Quick start commands:")
        print("1. cd ai-principles-gym")
        print("2. pip install -r requirements.txt")
        print("3. cd frontend && npm install && cd ..")
        print("4. cp .env.example .env (if needed)")
        print("5. echo 'VITE_API_URL=http://localhost:8000' > frontend/.env (if needed)")
        print("6. In one terminal: python -m src.api.app")
        print("7. In another terminal: cd frontend && npm run dev")

if __name__ == "__main__":
    main()
