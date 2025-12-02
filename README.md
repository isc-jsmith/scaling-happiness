# Synthetic Clinical Data Generator

This project provides an AI-powered agent for generating **synthetic clinical data** in either natural language or FHIR format. It is based on a Jupyter notebook prototype and has been turned into a reusable Python application with both a CLI and a small Flask web UI.

The agent uses:

- OpenAI `gpt-4o` via LangChain
- FHIR schemas and example bundles (unzipped into `fhir_data/` and `example_data/`)
- A FAISS vector store for retrieval-augmented generation (RAG)
- Web tools (DuckDuckGo, Wikipedia, PubMed, Arxiv) for additional context

## Prerequisites

- Python 3.10+ recommended
- Install dependencies:

```bash
pip install -r requirements.txt
```

- Create a `.env` file in the project root with at least:

```bash
OPENAI_API_KEY=your-openai-api-key
FHIR_ENDPOINT=https://your-fhir-endpoint.example.com  # optional default
FHIR_AUTH_USER=<FHIR Server HTTP User> # optional if server supports unauthenticated access
FHIR_AUTH_PASSWORD=<FHIR Server HTTP Password>
```

## Running the CLI

From the project root:

```bash
python3 -m app.main
```

You will be prompted for a clinical scenario (and optionally a FHIR endpoint to POST the generated bundle to). Type `exit` to quit.

## Running the web UI

From the project root:

```bash
python3 -m app.web
```

Then open `http://127.0.0.1:5000` in your browser. Use the form to enter a scenario and optionally a FHIR endpoint URL; the app will show the generated data and, if configured, the POST status.
