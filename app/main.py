from __future__ import annotations

from pathlib import Path
import json
from requests.auth import HTTPBasicAuth

import requests
from langchain_community.vectorstores import FAISS

from .config import load_settings
from .data_prep import (
    extract_fhir_examples,
    extract_fhir_schema,
    load_and_clean_json_files,
    make_chunks,
)
from .rag_agent import build_rag_chain
from .tools import create_web_tools
from .vector_store import build_vector_store


def initialize_agent(settings:dict) -> FAISS:
    if not settings:
       settings = load_settings()
       
    root = Path(settings["project_root"])
    content_dir = root / "content"

    fhir_schema_dir = root / "fhir_data"
    fhir_examples_dir = root / "example_data"

    extract_fhir_schema(content_dir, fhir_schema_dir)
    extract_fhir_examples(content_dir, fhir_examples_dir)

    fhir_schema_documents = load_and_clean_json_files(fhir_schema_dir)
    fhir_example_documents = load_and_clean_json_files(fhir_examples_dir)

    all_documents = list(fhir_schema_documents) + list(fhir_example_documents)
    chunks = make_chunks(all_documents)

    vector_store = build_vector_store(chunks, settings["openai_api_key"])
    return vector_store


def run_cli():
    print("Loading settings.")
    settings = load_settings()
    print("Initialising agent.")
    vector_store = initialize_agent(settings=settings)
    retriever = vector_store.as_retriever()
    rag_chain = build_rag_chain(retriever, settings["openai_api_key"])

    web_tools = create_web_tools()
    print(
        "Web tools enabled: "
        + ", ".join(sorted(web_tools.keys()))
        + " (results will be summarized into the prompt)."
    )

    while True:
        user_query = input(
            "\nEnter your request for clinical data "
            "(or type 'exit' to quit): "
        )
        if user_query.lower() == "exit":
            print("Exiting interactive session.")
            break

        if not user_query:
            print("Please enter a valid request.")
            continue

        print(f"Processing your request: {user_query}")

        web_snippets: list[str] = []
        for name, tool in web_tools.items():
            try:
                print(f"- Calling web tool: {name}")
                snippet = tool.run(user_query)
                if snippet:
                    web_snippets.append(f"[{name}]\n{snippet}")
            except Exception as exc:
                print(f"  (Tool {name} failed: {exc})")

        web_context = "\n\n".join(web_snippets)
        augmented_query = user_query
        if web_context:
            augmented_query = (
                f"{user_query}\n\n"
                "[WEB_RESULTS]\n"
                f"{web_context}"
            )

        try:
            response = rag_chain.invoke(augmented_query)

            # Attempt to parse structured JSON response separating natural language
            # content from the FHIR bundle. Be tolerant of markdown code fences.
            parsed = None
            raw_text = response.strip()

            try:
                parsed = json.loads(raw_text)
            except Exception:
                if raw_text.startswith("```"):
                    lines = raw_text.splitlines()
                    # Drop first fence line (e.g. ``` or ```json)
                    lines = lines[1:]
                    # Drop trailing fence line(s)
                    while lines and lines[-1].strip().startswith("```"):
                        lines = lines[:-1]
                    cleaned = "\n".join(lines).strip()
                    try:
                        parsed = json.loads(cleaned)
                    except Exception:
                        parsed = None
                else:
                    parsed = None

            fhir_bundle = None
            natural_language = None
            fhir_payload = None

            if isinstance(parsed, dict) and "fhir_bundle" in parsed:
                fhir_bundle = parsed.get("fhir_bundle")
                natural_language = parsed.get("natural_language") or ""

                # Prepare payload to POST (pure FHIR bundle only)
                if isinstance(fhir_bundle, (dict, list)):
                    fhir_payload = json.dumps(fhir_bundle)
                    pretty_fhir = json.dumps(fhir_bundle, indent=2)
                else:
                    fhir_payload = fhir_bundle
                    try:
                        pretty_fhir = json.dumps(json.loads(fhir_bundle), indent=2)
                    except Exception:
                        pretty_fhir = str(fhir_bundle)

                print("\n--- Natural Language Summary ---")
                if natural_language:
                    print(natural_language)
                else:
                    print("(none)")

                print("\n--- FHIR Bundle ---")
                print(pretty_fhir)
            else:
                # Fall back to original raw response if not structured
                print("\n--- Generated Clinical Data (raw) ---")
                print(response)

        except Exception as exc:
            print(f"An error occurred during data generation: {exc}")
            continue

        default_endpoint = settings.get("fhir_endpoint")
        prompt = (
            "\nOptional: enter a FHIR endpoint URL to POST this "
            "bundle to (press Enter to "
        )
        if default_endpoint:
            prompt += f"use default {default_endpoint!r} or "
        prompt += "skip): "

        endpoint_url = input(prompt).strip() or default_endpoint or ""
        if endpoint_url:
            try:
                # If we successfully parsed a FHIR bundle above, prefer that for POST.
                payload_to_send = fhir_payload if 'fhir_payload' in locals() and fhir_payload else response
                fhir_username = settings.get('fhir_auth_user')
                fhir_password = settings.get('fhir_auth_passwd')

                http_response = requests.post(
                    url=endpoint_url,
                    data=payload_to_send,
                    headers={"Content-Type": "application/fhir+json"},
                    timeout=15,
                    auth=HTTPBasicAuth(username=fhir_username, password=fhir_password),
                )
                print(
                    f"POSTed bundle to {endpoint_url} "
                    f"(status {http_response.status_code})."
                )
            except Exception as exc:
                print(f"Failed to POST bundle to endpoint: {exc}")


if __name__ == "__main__":
    run_cli()
