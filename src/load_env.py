"""
Helper module to load environment variables from .env file
Makes it easier to use API keys without hardcoding them
"""

import os
from pathlib import Path

def load_env_file():
    """
    Load environment variables from .env file if it exists
    """
    env_file = Path(__file__).parent.parent / ".env"
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        return True
    else:
        return False


def get_evds_api_key():
    """
    Get EVDS API key from environment variable or .env file
    
    Returns:
        API key string or None if not found
    """
    # Try loading from .env file first
    load_env_file()
    
    # Get from environment variable
    api_key = os.environ.get('EVDS_API_KEY')
    
    if api_key and api_key != 'your_api_key_here':
        return api_key
    else:
        return None


if __name__ == "__main__":
    # Test loading
    load_env_file()
    api_key = get_evds_api_key()
    
    if api_key:
        print(f"✅ Found API key: {api_key[:10]}...{api_key[-5:]}")
    else:
        print("⚠️  No API key found in .env file or environment variables")
        print("   Create a .env file with: EVDS_API_KEY=your_key_here")
