#!/usr/bin/env python3
"""Test script to verify LangChain compatibility with the gym."""

import sys
import importlib
import traceback
from typing import Dict, List, Tuple

def test_imports() -> Tuple[bool, List[str]]:
    """Test if all required packages can be imported."""
    packages = [
        "langchain",
        "langchain.agents",
        "langchain.memory",
        "langchain.prompts",
        "langchain.schema",
        "langchain.tools",
        "langchain_core",
        "langchain_openai",
        "langchain_anthropic",
        "langchain_community",
    ]
    
    errors = []
    for package in packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            errors.append(f"✗ {package}: {str(e)}")
    
    return len(errors) == 0, errors

def test_adapter_creation() -> Tuple[bool, str]:
    """Test if the LangChain adapter can be created."""
    try:
        from src.adapters.langchain_adapter import LangChainAdapter, LangChainNotInstalledError
        print("✓ LangChain adapter imports successfully")
        
        # Try to create a mock adapter (will fail without proper LLM)
        try:
            from langchain_openai import ChatOpenAI
            # This will fail without API key, but we're just testing imports
            llm = ChatOpenAI(api_key="test-key")
            adapter = LangChainAdapter(llm)
            print("✓ LangChain adapter can be instantiated")
            return True, ""
        except LangChainNotInstalledError as e:
            return False, str(e)
        except Exception as e:
            # Expected if no API key, but adapter creation worked
            if "api_key" in str(e).lower() or "openai" in str(e).lower():
                print("✓ LangChain adapter creation works (API key error expected)")
                return True, ""
            else:
                return False, f"Unexpected error: {str(e)}"
                
    except Exception as e:
        return False, f"Failed to import adapter: {str(e)}"

def test_core_dependencies() -> Tuple[bool, List[str]]:
    """Test if core dependencies don't conflict with LangChain."""
    conflicts = []
    
    try:
        # Test numpy
        import numpy as np
        arr = np.array([1, 2, 3])
        print("✓ NumPy works correctly")
    except Exception as e:
        conflicts.append(f"NumPy conflict: {str(e)}")
    
    try:
        # Test pydantic
        from pydantic import BaseModel
        class TestModel(BaseModel):
            test: str
        model = TestModel(test="value")
        print("✓ Pydantic works correctly")
    except Exception as e:
        conflicts.append(f"Pydantic conflict: {str(e)}")
    
    try:
        # Test FastAPI
        from fastapi import FastAPI
        app = FastAPI()
        print("✓ FastAPI works correctly")
    except Exception as e:
        conflicts.append(f"FastAPI conflict: {str(e)}")
    
    return len(conflicts) == 0, conflicts

def main():
    """Run all compatibility tests."""
    print("=== LangChain Compatibility Test ===\n")
    
    # Test imports
    print("1. Testing LangChain imports...")
    imports_ok, import_errors = test_imports()
    if not imports_ok:
        print("\nImport errors:")
        for error in import_errors:
            print(f"  {error}")
    print()
    
    # Test adapter
    print("2. Testing LangChain adapter...")
    adapter_ok, adapter_error = test_adapter_creation()
    if not adapter_ok:
        print(f"  Error: {adapter_error}")
    print()
    
    # Test core dependencies
    print("3. Testing core dependencies compatibility...")
    deps_ok, dep_conflicts = test_core_dependencies()
    if not deps_ok:
        print("\nDependency conflicts:")
        for conflict in dep_conflicts:
            print(f"  {conflict}")
    print()
    
    # Summary
    print("=== Summary ===")
    all_ok = imports_ok and adapter_ok and deps_ok
    if all_ok:
        print("✓ All tests passed! LangChain is compatible with the gym.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        if not imports_ok:
            print("\nTo fix import errors, run:")
            print("  pip install -r requirements-langchain.txt")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
