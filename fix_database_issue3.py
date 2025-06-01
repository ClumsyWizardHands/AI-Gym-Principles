"""Fix database method calls - remove session parameter"""

# Read the file
with open('src/api/training_integration.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix all database method calls to remove the db session parameter
replacements = [
    # get_agent
    ('await self.db_manager.get_agent(\n                            db, agent_id\n                        )', 
     'await self.db_manager.get_agent(agent_id)'),
    
    # create_agent
    ('await self.db_manager.create_agent(\n                                db,\n                                agent_id=agent_id,\n                                framework=framework,\n                                config=agent_config["config"]\n                            )',
     'await self.db_manager.create_agent(\n                                agent_id=agent_id,\n                                framework=framework,\n                                config=agent_config["config"]\n                            )'),
    
    # get_agent_principles
    ('await self.db_manager.get_agent_principles(\n                db,\n                session.agent_id\n            )',
     'await self.db_manager.get_agent_principles(\n                session.agent_id\n            )'),
    
    # upsert_principle
    ('await self.db_manager.upsert_principle(\n                        db,\n                        agent_id=session.agent_id,\n                        principle=principle,\n                        session_id=session_id\n                    )',
     'await self.db_manager.upsert_principle(\n                        agent_id=session.agent_id,\n                        principle=principle,\n                        session_id=session_id\n                    )')
]

# Apply replacements
for old, new in replacements:
    content = content.replace(old, new)

# Write the fixed content back
with open('src/api/training_integration.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Fixed database method calls in training_integration.py")
print("   - Removed 'db' session parameter from all DatabaseManager method calls")
print("   - get_agent, create_agent, get_agent_principles, upsert_principle")
