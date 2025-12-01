from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template_string, request

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


_HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Synthetic Clinical Data Generator</title>
    <style>
      body { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
             margin: 2rem; max-width: 960px; }
      textarea { width: 100%; min-height: 120px; }
      pre { white-space: pre-wrap; word-wrap: break-word; padding: 1rem;
            background: #f5f5f5; border-radius: 4px; }
      .meta { font-size: 0.9rem; color: #555; margin-bottom: 0.5rem; }
      button { padding: 0.5rem 1.25rem; font-size: 1rem; }
    </style>
  </head>
  <body>
    <h1>Synthetic Clinical Data Generator</h1>
    <p>
      Describe the patient condition and demographics. You can also request
      FHIR output, for example:
    </p>
    <ul>
      <li>
        Natural language:
        <code>Generate natural language data for a 45-year-old male with type 2 diabetes and hypertension.</code>
      </li>
      <li>
        FHIR:
        <code>Generate FHIR data for a 60-year-old female with osteoporosis.</code>
      </li>
    </ul>

    <form method="post" action="/generate">
      <label for="query"><strong>Your prompt</strong></label><br />
      <textarea id="query" name="query" required></textarea><br /><br />
      <button type="submit">Generate Clinical Data</button>
    </form>

    {% if response %}
      <hr />
      <div class="meta">
        Web tools used: {{ tools|join(", ") if tools else "none" }}
      </div>
      <h2>Generated Clinical Data</h2>
      <pre>{{ response }}</pre>
    {% endif %}
  </body>
  </html>
"""


def _initialize_components():
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
    retriever = vector_store.as_retriever()
    rag_chain = build_rag_chain(retriever, settings["openai_api_key"])
    tools = create_web_tools()

    return settings, rag_chain, tools


def create_app() -> Flask:
    app = Flask(__name__)

    settings, rag_chain, web_tools = _initialize_components()
    app.config["APP_SETTINGS"] = settings
    app.config["RAG_CHAIN"] = rag_chain
    app.config["WEB_TOOLS"] = web_tools

    @app.get("/")
    def index():
        return render_template_string(_HTML_TEMPLATE)

    @app.post("/generate")
    def generate():
        query = request.form.get("query") or request.json.get("query")  # type: ignore[union-attr]
        if not query:
            if request.content_type and "application/json" in request.content_type:
                return jsonify({"error": "Missing 'query'"}), 400
            return render_template_string(
                _HTML_TEMPLATE, response=None, tools=[]
            )

        rag = app.config["RAG_CHAIN"]
        tools = app.config["WEB_TOOLS"]

        web_snippets: list[str] = []
        tool_names: list[str] = []
        for name, tool in tools.items():
            try:
                snippet = tool.run(query)
                if snippet:
                    web_snippets.append(f"[{name}]\n{snippet}")
                    tool_names.append(name)
            except Exception:
                continue

        web_context = "\n\n".join(web_snippets)
        augmented_query = query
        if web_context:
            augmented_query = (
                f"{query}\n\n"
                "[WEB_RESULTS]\n"
                f"{web_context}"
            )

        try:
            response = rag.invoke(augmented_query)
        except Exception as exc:  # pragma: no cover - simple error surface
            if request.content_type and "application/json" in request.content_type:
                return jsonify({"error": str(exc)}), 500
            return render_template_string(
                _HTML_TEMPLATE,
                response=f"Error during generation: {exc}",
                tools=tool_names,
            )

        if request.content_type and "application/json" in request.content_type:
            return jsonify(
                {
                    "response": response,
                    "tools_used": tool_names,
                }
            )

        return render_template_string(
            _HTML_TEMPLATE, response=response, tools=tool_names
        )

    return app


def run_web(host: str = "127.0.0.1", port: int = 5000):
    app = create_app()
    app.run(host=host, port=port)


if __name__ == "__main__":
    run_web()

