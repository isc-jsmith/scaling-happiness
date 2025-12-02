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
        print(f'Resolved .env path to {env_path}')

    load_dotenv(dotenv_path=env_path)

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file in the project root."
        )

    fhir_endpoint = os.environ.get("FHIR_ENDPOINT")
    fhir_auth_user = os.environ.get("FHIR_AUTH_USER")
    fhir_auth_passwd = os.environ.get("FHIR_AUTH_PASSWORD")
    return {
        "openai_api_key": openai_api_key,
        "project_root": str(Path(__file__).resolve().parents[1]),
        "fhir_endpoint": fhir_endpoint,
        "fhir_auth_passwd": fhir_auth_passwd,
        "fhir_auth_user":fhir_auth_user,
    }

