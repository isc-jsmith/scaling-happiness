import os
from pathlib import Path

from dotenv import load_dotenv


def load_settings(env_path: str | None = None) -> dict:
    """
    Load configuration and API keys.

    Looks for an .env file in the project root by default and
    expects OPENAI_API_KEY to be defined.
    """
    if env_path is None:
        # Assume project root is one level up from this file
        root = Path(__file__).resolve().parents[1]
        env_path = root / ".env"

    load_dotenv(dotenv_path=env_path)

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file in the project root."
        )

    return {
        "openai_api_key": openai_api_key,
        "project_root": str(Path(__file__).resolve().parents[1]),
    }

