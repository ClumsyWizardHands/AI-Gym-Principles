"""Quick setup script to configure custom API key validation."""

import os
import sys

def setup_custom_api_key():
    print("Setting up custom API key configuration...")
    
    # Ask for the API key
    api_key = input("Enter your custom API key (or press Enter to use 'your-api-key-here'): ").strip()
    if not api_key:
        api_key = "your-api-key-here"
    
    # Ask for environment mode
    print("\nSelect environment mode:")
    print("1. Development (accepts any API key)")
    print("2. Production (enforces specific API keys)")
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == "2":
        env_mode = "production"
        print("\n⚠️  Production mode selected. You'll need to generate API keys using the /keys endpoint.")
    else:
        env_mode = "development"
        print("\n✅ Development mode selected. Any API key will work.")
    
    # Update .env file
    env_path = ".env"
    
    with open(env_path, 'r') as f:
        lines = f.readlines()
    
    # Update the lines
    updated_lines = []
    for line in lines:
        if line.startswith("ENVIRONMENT="):
            updated_lines.append(f"ENVIRONMENT={env_mode}\n")
        elif line.startswith("API_KEY="):
            updated_lines.append(f"API_KEY={api_key}\n")
        else:
            updated_lines.append(line)
    
    # Write back
    with open(env_path, 'w') as f:
        f.writelines(updated_lines)
    
    print(f"\n✅ Configuration updated!")
    print(f"   - Environment: {env_mode}")
    print(f"   - API Key (for reference): {api_key}")
    
    if env_mode == "production":
        print("\n📝 Next steps for production mode:")
        print("1. Restart the server to apply changes")
        print("2. Generate an API key by calling POST /api/keys")
        print("3. Use the generated key in X-API-Key header")
    else:
        print(f"\n📝 In development mode, use this header:")
        print(f'   X-API-Key: {api_key}')

if __name__ == "__main__":
    setup_custom_api_key()
