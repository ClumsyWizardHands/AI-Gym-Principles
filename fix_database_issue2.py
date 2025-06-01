"""Fix more database method name issues in training_integration.py"""

# Read the file
with open('src/api/training_integration.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all occurrences of wrong method names
content = content.replace('self.db_manager.get_agent_profile(', 'self.db_manager.get_agent(')
content = content.replace('self.db_manager.create_agent_profile(', 'self.db_manager.create_agent(')
content = content.replace('self.db_manager.save_principle(', 'self.db_manager.upsert_principle(')

# Write the fixed content back
with open('src/api/training_integration.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Fixed more database method calls in training_integration.py")
print("   - get_agent_profile -> get_agent")
print("   - create_agent_profile -> create_agent") 
print("   - save_principle -> upsert_principle")
