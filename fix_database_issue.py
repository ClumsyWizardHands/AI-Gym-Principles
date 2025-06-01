"""Fix the database method name issue in training_integration.py"""

import re

# Read the file
with open('src/api/training_integration.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all occurrences of get_session() with session()
content = content.replace('self.db_manager.get_session()', 'self.db_manager.session()')

# Write the fixed content back
with open('src/api/training_integration.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Fixed database method calls in training_integration.py")
print("   Changed self.db_manager.get_session() -> self.db_manager.session()")
