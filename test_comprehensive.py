"""
Comprehensive test suite for AI Principles Gym
Tests both backend API and frontend functionality
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import Dict, Any
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

class ComprehensiveTest:
    def __init__(self):
        self.api_base = "http://localhost:8000"
        self.frontend_base = "http://localhost:5173"
        self.api_key = None
        self.agent_id = None
        self.session_id = None
        self.results = {
            "backend": {},
            "frontend": {},
            "integration": {}
        }
        
    def log(self, message: str, color: str = RESET):
        """Print colored log message"""
        print(f"{color}{message}{RESET}")
        
    def log_test(self, category: str, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        status = f"{GREEN}✓ PASSED{RESET}" if passed else f"{RED}✗ FAILED{RESET}"
        self.log(f"  [{category}] {test_name}: {status}")
        if details:
            self.log(f"    → {details}", YELLOW)
        self.results[category][test_name] = {"passed": passed, "details": details}
        
    async def test_backend_health(self, session: aiohttp.ClientSession) -> bool:
        """Test backend health endpoint"""
        try:
            async with session.get(f"{self.api_base}/health") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self.log_test("backend", "Health Check", True, f"Status: {data.get('status')}")
                    return True
                else:
                    self.log_test("backend", "Health Check", False, f"Status code: {resp.status}")
                    return False
        except Exception as e:
            self.log_test("backend", "Health Check", False, str(e))
            return False
            
    async def test_backend_metrics(self, session: aiohttp.ClientSession) -> bool:
        """Test backend metrics endpoint"""
        try:
            async with session.get(f"{self.api_base}/metrics") as resp:
                if resp.status == 200:
                    # Prometheus format text, just check it's not empty
                    text = await resp.text()
                    self.log_test("backend", "Metrics Endpoint", True, "Prometheus metrics available")
                    return True
                else:
                    self.log_test("backend", "Metrics Endpoint", False, f"Status code: {resp.status}")
                    return False
        except Exception as e:
            self.log_test("backend", "Metrics Endpoint", False, str(e))
            return False
            
    async def test_api_key_generation(self, session: aiohttp.ClientSession) -> bool:
        """Test API key generation"""
        try:
            payload = {
                "user_id": "test_user_comprehensive",
                "usage_limit": 1000,
                "expires_in_days": 30
            }
            async with session.post(f"{self.api_base}/api/keys", json=payload) as resp:
                if resp.status == 201:  # 201 Created is the correct status for resource creation
                    data = await resp.json()
                    self.api_key = data.get("api_key")
                    self.log_test("backend", "API Key Generation", True, f"Key: {self.api_key[:20]}...")
                    return True
                else:
                    text = await resp.text()
                    self.log_test("backend", "API Key Generation", False, f"Status: {resp.status}, Body: {text}")
                    return False
        except Exception as e:
            self.log_test("backend", "API Key Generation", False, str(e))
            return False
            
    async def test_agent_registration(self, session: aiohttp.ClientSession) -> bool:
        """Test agent registration"""
        try:
            if not self.api_key:
                self.log_test("backend", "Agent Registration", False, "No API key available")
                return False
                
            headers = {"X-API-Key": self.api_key}
            payload = {
                "agent_id": f"test_agent_{int(time.time())}",
                "framework": "custom",
                "config": {
                    "name": "Test Agent",
                    "version": "1.0.0"
                }
            }
            
            async with session.post(
                f"{self.api_base}/api/agents/register", 
                json=payload,
                headers=headers
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self.agent_id = data.get("agent_id")
                    self.log_test("backend", "Agent Registration", True, f"Agent ID: {self.agent_id}")
                    return True
                else:
                    text = await resp.text()
                    self.log_test("backend", "Agent Registration", False, f"Status: {resp.status}, Body: {text}")
                    return False
        except Exception as e:
            self.log_test("backend", "Agent Registration", False, str(e))
            return False
            
    async def test_list_agents(self, session: aiohttp.ClientSession) -> bool:
        """Test listing agents"""
        try:
            if not self.api_key:
                self.log_test("backend", "List Agents", False, "No API key available")
                return False
                
            headers = {"X-API-Key": self.api_key}
            
            async with session.get(
                f"{self.api_base}/api/agents",
                headers=headers
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    agent_count = len(data.get("agents", []))
                    self.log_test("backend", "List Agents", True, f"Found {agent_count} agents")
                    return True
                else:
                    text = await resp.text()
                    self.log_test("backend", "List Agents", False, f"Status: {resp.status}, Body: {text}")
                    return False
        except Exception as e:
            self.log_test("backend", "List Agents", False, str(e))
            return False
            
    async def test_list_plugins(self, session: aiohttp.ClientSession) -> bool:
        """Test listing plugins"""
        try:
            if not self.api_key:
                self.log_test("backend", "List Plugins", False, "No API key available")
                return False
                
            headers = {"X-API-Key": self.api_key}
            
            async with session.get(
                f"{self.api_base}/api/plugins",
                headers=headers
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    plugin_count = len(data.get("plugins", []))
                    plugin_names = [p["name"] for p in data.get("plugins", [])]
                    self.log_test("backend", "List Plugins", True, 
                                f"Found {plugin_count} plugins: {', '.join(plugin_names[:3])}...")
                    return True
                else:
                    text = await resp.text()
                    self.log_test("backend", "List Plugins", False, f"Status: {resp.status}, Body: {text}")
                    return False
        except Exception as e:
            self.log_test("backend", "List Plugins", False, str(e))
            return False
            
    async def test_training_start(self, session: aiohttp.ClientSession) -> bool:
        """Test starting a training session"""
        try:
            if not self.api_key or not self.agent_id:
                self.log_test("backend", "Start Training", False, "Missing API key or agent ID")
                return False
                
            headers = {"X-API-Key": self.api_key}
            payload = {
                "agent_id": self.agent_id,
                "num_scenarios": 5,
                "config": {
                    "stress_level": 0.5,
                    "scenario_types": ["LOYALTY", "TRADEOFFS"]
                }
            }
            
            async with session.post(
                f"{self.api_base}/api/training/start",
                json=payload,
                headers=headers
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self.session_id = data.get("session_id")
                    self.log_test("backend", "Start Training", True, f"Session ID: {self.session_id}")
                    return True
                else:
                    text = await resp.text()
                    self.log_test("backend", "Start Training", False, f"Status: {resp.status}, Body: {text}")
                    return False
        except Exception as e:
            self.log_test("backend", "Start Training", False, str(e))
            return False
            
    async def test_training_status(self, session: aiohttp.ClientSession) -> bool:
        """Test checking training status"""
        try:
            if not self.api_key or not self.session_id:
                self.log_test("backend", "Training Status", False, "Missing API key or session ID")
                return False
                
            headers = {"X-API-Key": self.api_key}
            
            # Wait a bit for training to start
            await asyncio.sleep(2)
            
            async with session.get(
                f"{self.api_base}/api/training/status/{self.session_id}",
                headers=headers
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    status = data.get("status")
                    progress = data.get("progress", 0)
                    self.log_test("backend", "Training Status", True, 
                                f"Status: {status}, Progress: {progress}%")
                    return True
                else:
                    text = await resp.text()
                    self.log_test("backend", "Training Status", False, f"Status: {resp.status}, Body: {text}")
                    return False
        except Exception as e:
            self.log_test("backend", "Training Status", False, str(e))
            return False
            
    async def test_websocket_connection(self, session: aiohttp.ClientSession) -> bool:
        """Test WebSocket connection"""
        try:
            if not self.api_key:
                self.log_test("backend", "WebSocket Connection", False, "No API key available")
                return False
                
            ws_url = f"ws://localhost:8000/ws?api_key={self.api_key}"
            
            async with session.ws_connect(ws_url) as ws:
                # Send a test message
                await ws.send_str(json.dumps({
                    "type": "subscribe",
                    "channel": "test"
                }))
                
                # Wait for response with timeout
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=5.0)
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        self.log_test("backend", "WebSocket Connection", True, 
                                    f"Connected and received: {data.get('type', 'unknown')}")
                        await ws.close()
                        return True
                    else:
                        self.log_test("backend", "WebSocket Connection", False, 
                                    f"Unexpected message type: {msg.type}")
                        return False
                except asyncio.TimeoutError:
                    self.log_test("backend", "WebSocket Connection", False, "Connection timeout")
                    return False
                    
        except Exception as e:
            self.log_test("backend", "WebSocket Connection", False, str(e))
            return False
            
    async def test_frontend_health(self, session: aiohttp.ClientSession) -> bool:
        """Test frontend is serving"""
        try:
            async with session.get(self.frontend_base) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    if "AI Principles Gym" in text or "root" in text:
                        self.log_test("frontend", "Frontend Serving", True, "HTML content received")
                        return True
                    else:
                        self.log_test("frontend", "Frontend Serving", False, "Unexpected content")
                        return False
                else:
                    self.log_test("frontend", "Frontend Serving", False, f"Status code: {resp.status}")
                    return False
        except Exception as e:
            self.log_test("frontend", "Frontend Serving", False, str(e))
            return False
            
    async def test_frontend_assets(self, session: aiohttp.ClientSession) -> bool:
        """Test frontend can serve assets"""
        try:
            # Check if main JS file is accessible
            async with session.get(f"{self.frontend_base}/src/main.tsx") as resp:
                # Vite dev server should serve the file
                if resp.status in [200, 304]:  # 304 is not modified
                    self.log_test("frontend", "Asset Serving", True, "TypeScript files accessible")
                    return True
                else:
                    self.log_test("frontend", "Asset Serving", False, f"Status code: {resp.status}")
                    return False
        except Exception as e:
            self.log_test("frontend", "Asset Serving", False, str(e))
            return False
            
    def print_summary(self):
        """Print test summary"""
        self.log("\n" + "="*50, BLUE)
        self.log("TEST SUMMARY", BLUE)
        self.log("="*50, BLUE)
        
        total_tests = 0
        passed_tests = 0
        
        for category, tests in self.results.items():
            if tests:
                self.log(f"\n{category.upper()}:", YELLOW)
                for test_name, result in tests.items():
                    total_tests += 1
                    if result["passed"]:
                        passed_tests += 1
                        self.log(f"  ✓ {test_name}", GREEN)
                    else:
                        self.log(f"  ✗ {test_name}: {result['details']}", RED)
                        
        self.log(f"\nTotal: {passed_tests}/{total_tests} tests passed", 
                GREEN if passed_tests == total_tests else YELLOW)
        
        if passed_tests < total_tests:
            self.log("\nSome tests failed. Please check the logs above for details.", RED)
        else:
            self.log("\nAll tests passed! The system is working correctly.", GREEN)
            
    async def run_all_tests(self):
        """Run all tests"""
        self.log("Starting comprehensive test suite...\n", BLUE)
        
        # Test backend
        self.log("TESTING BACKEND API", YELLOW)
        self.log("-" * 30, YELLOW)
        
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Basic health checks
            await self.test_backend_health(session)
            await self.test_backend_metrics(session)
            
            # API functionality
            await self.test_api_key_generation(session)
            await self.test_agent_registration(session)
            await self.test_list_agents(session)
            await self.test_list_plugins(session)
            await self.test_training_start(session)
            await self.test_training_status(session)
            await self.test_websocket_connection(session)
            
        # Test frontend
        self.log("\nTESTING FRONTEND", YELLOW)
        self.log("-" * 30, YELLOW)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            await self.test_frontend_health(session)
            await self.test_frontend_assets(session)
            
        # Print summary
        self.print_summary()


async def main():
    """Main test runner"""
    tester = ComprehensiveTest()
    
    # Wait a moment for servers to be ready
    print("Waiting for servers to be ready...")
    await asyncio.sleep(2)
    
    try:
        await tester.run_all_tests()
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error during tests: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
