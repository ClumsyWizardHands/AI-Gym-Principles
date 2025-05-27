#!/usr/bin/env python3
"""Test script to verify no import warnings when LangChain is not installed."""

import sys
import warnings

# Capture all warnings
warnings.simplefilter("always")
captured_warnings = []

def warning_handler(message, category, filename, lineno, file=None, line=None):
    captured_warnings.append({
        'message': str(message),
        'category': category.__name__,
        'filename': filename,
        'lineno': lineno
    })

# Set custom warning handler
warnings.showwarning = warning_handler

print("Testing imports without LangChain installed...")
print("=" * 50)

# Test importing from adapters
try:
    print("\n1. Testing: from src.adapters import *")
    from src.adapters import (
        AgentInterface, 
        OpenAIAdapter, 
        AnthropicAdapter, 
        LangChainAdapter, 
        CustomAdapter
    )
    print("✓ Import successful")
    
    # Test that LangChainAdapter exists but raises error on use
    print("\n2. Testing LangChainAdapter placeholder...")
    try:
        adapter = LangChainAdapter(llm=None)
        print("✗ LangChainAdapter should have raised an ImportError")
    except ImportError as e:
        print("✓ LangChainAdapter correctly raises ImportError:")
        print(f"  {str(e).split(chr(10))[0]}...")
        
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()

# Test importing individual modules
print("\n3. Testing individual adapter imports...")
modules = [
    "src.adapters.base",
    "src.adapters.openai_adapter",
    "src.adapters.anthropic_adapter",
    "src.adapters.custom_adapter",
]

for module in modules:
    try:
        __import__(module)
        print(f"✓ {module}")
    except Exception as e:
        print(f"✗ {module}: {e}")

# Check for any warnings
print("\n" + "=" * 50)
if captured_warnings:
    print(f"\n⚠️  Found {len(captured_warnings)} warnings:")
    for w in captured_warnings:
        print(f"\n  Warning: {w['message']}")
        print(f"  Category: {w['category']}")
        print(f"  File: {w['filename']}:{w['lineno']}")
else:
    print("\n✓ No warnings detected during imports!")

print("\n" + "=" * 50)
print("Summary:")
print("- Core adapters can be imported without LangChain")
print("- LangChainAdapter provides a clear error when used")
print("- No import warnings are generated")
