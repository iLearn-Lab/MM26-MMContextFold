"""
api_keys.py

Centralized API key management.

Usage:
1. Set environment variables (recommended for production)
2. Or create a .env file with python-dotenv
"""

import os


def get_api_key(key_name: str, default: str = "") -> str:
    return os.environ.get(key_name, default)


# ========== Search APIs ==========
SERPER_API_KEY = get_api_key("SERPER_API_KEY", "")
SERPAPI_API_KEY = get_api_key("SERPAPI_API_KEY", "")

# ========== Jina (Web Reader) ==========
JINA_API_KEY = get_api_key("JINA_API_KEY", "")

# ========== LLM APIs ==========
OPENROUTER_API_KEY = get_api_key("OPENROUTER_API_KEY", "")


# ========== Setup environment variables ==========
def setup_env():
    pass


setup_env()
