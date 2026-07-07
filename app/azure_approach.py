import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

def get_openai_client() -> Optional[object]:
    """Initialize and return OpenAI client using .env"""

    try:
        from openai import OpenAI

        base_url = os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", "gpt-5.2-chat")

        if not base_url or not api_key:
            raise ValueError("Missing Azure OpenAI configuration in .env")

        print("🔧 Initializing OpenAI client from .env...")

        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

        # 🔍 Quick health check
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say OK"}],
        )

        print(" OpenAI client ready!")
        return client

    except Exception as e:
        print(f" OpenAI error: {e}")
        return None
