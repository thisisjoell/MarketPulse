# test_env.py

from dotenv import load_dotenv
import os

load_dotenv()

print("✅ Reddit Client ID:", os.getenv("REDDIT_CLIENT_ID"))
print("✅ OpenAI Key (partial):", os.getenv("OPENAI_API_KEY")[:10] + "...")
