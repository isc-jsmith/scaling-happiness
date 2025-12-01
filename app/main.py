from __future__ import annotations

from pathlib import Path

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


def initialize_agent() -> FAISS:
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
    vector_store = initialize_agent()
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
            print("\n--- Generated Clinical Data ---")
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
                http_response = requests.post(
                    endpoint_url,
                    data=response,
                    headers={"Content-Type": "application/fhir+json"},
                    timeout=15,
                )
                print(
                    f"POSTed bundle to {endpoint_url} "
                    f"(status {http_response.status_code})."
                )
            except Exception as exc:
                print(f"Failed to POST bundle to endpoint: {exc}")


if __name__ == "__main__":
    run_cli()
