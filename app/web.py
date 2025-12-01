from __future__ import annotations

from pathlib import Path

import requests
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
      :root {
        color-scheme: light dark;
        --bg: #0f172a;
        --bg-alt: #020617;
        --card-bg: #020617;
        --text: #e5e7eb;
        --muted: #9ca3af;
        --accent: #38bdf8;
        --accent-soft: rgba(56, 189, 248, 0.12);
        --border: #1f2937;
        --danger: #f97373;
        --radius-lg: 12px;
        --radius-md: 8px;
        --shadow-soft: 0 18px 40px rgba(15, 23, 42, 0.75);
      }

      * { box-sizing: border-box; }

      body {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
          sans-serif;
        margin: 0;
        min-height: 100vh;
        display: flex;
        align-items: stretch;
        justify-content: center;
        background: radial-gradient(circle at top, #1e293b 0, #020617 52%);
        color: var(--text);
      }

      .shell {
        width: 100%;
        max-width: 1160px;
        padding: 2.5rem 1.75rem 3rem;
      }

      .layout {
        display: grid;
        grid-template-columns: minmax(0, 3fr) minmax(0, 2.4fr);
        gap: 1.75rem;
        align-items: flex-start;
      }

      @media (max-width: 960px) {
        .shell { padding: 1.75rem 1.25rem 2.25rem; }
        .layout { grid-template-columns: 1fr; }
      }

      .panel {
        background: radial-gradient(circle at top left, #111827 0, #020617 50%);
        border-radius: var(--radius-lg);
        border: 1px solid rgba(148, 163, 184, 0.1);
        box-shadow: var(--shadow-soft);
        overflow: hidden;
      }

      .header {
        padding: 1.25rem 1.5rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.22);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        background: linear-gradient(
          135deg,
          rgba(56, 189, 248, 0.10),
          rgba(15, 23, 42, 0.3)
        );
      }

      .title-block h1 {
        margin: 0 0 0.15rem;
        font-size: 1.35rem;
        letter-spacing: 0.02em;
      }

      .title-block p {
        margin: 0;
        font-size: 0.9rem;
        color: var(--muted);
      }

      .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.3rem;
        justify-content: flex-end;
      }

      .badge {
        border-radius: 999px;
        padding: 0.1rem 0.55rem;
        font-size: 0.75rem;
        border: 1px solid rgba(148, 163, 184, 0.65);
        color: var(--muted);
        background: rgba(15, 23, 42, 0.7);
        white-space: nowrap;
      }

      .badge-accent {
        border-color: rgba(56, 189, 248, 0.65);
        color: var(--accent);
        background: rgba(15, 23, 42, 0.9);
      }

      main {
        padding: 1.5rem 1.5rem 1.6rem;
        display: flex;
        flex-direction: column;
        gap: 1.3rem;
      }

      .field-label {
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 0.25rem;
      }

      textarea {
        width: 100%;
        min-height: 140px;
        resize: vertical;
        padding: 0.75rem 0.85rem;
        border-radius: var(--radius-md);
        border: 1px solid rgba(148, 163, 184, 0.45);
        background: rgba(15, 23, 42, 0.9);
        color: var(--text);
        font-family: inherit;
        font-size: 0.95rem;
      }

      textarea:focus-visible {
        outline: none;
        border-color: var(--accent);
        box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.4);
      }

      .hint {
        font-size: 0.82rem;
        color: var(--muted);
        margin-top: 0.3rem;
      }

      code {
        font-size: 0.82rem;
        background: rgba(15, 23, 42, 0.9);
        padding: 0.2rem 0.35rem;
        border-radius: 4px;
        border: 1px solid rgba(148, 163, 184, 0.4);
      }

      .endpoint-input {
        width: 100%;
        padding: 0.55rem 0.7rem;
        border-radius: var(--radius-md);
        border: 1px solid rgba(148, 163, 184, 0.4);
        background: rgba(15, 23, 42, 0.9);
        color: var(--text);
        font-size: 0.9rem;
      }

      .endpoint-input:focus-visible {
        outline: none;
        border-color: var(--accent);
        box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.35);
      }

      .actions {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        margin-top: 1.15rem;
      }

      button {
        padding: 0.55rem 1.3rem;
        font-size: 0.95rem;
        font-weight: 500;
        border-radius: 999px;
        border: none;
        cursor: pointer;
        background: linear-gradient(135deg, #38bdf8, #0ea5e9);
        color: #0b1120;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        box-shadow: 0 10px 30px rgba(56, 189, 248, 0.35);
      }

      button[disabled] {
        opacity: 0.7;
        cursor: wait;
        box-shadow: none;
      }

      .spinner {
        width: 14px;
        height: 14px;
        border-radius: 999px;
        border: 2px solid rgba(15, 23, 42, 0.1);
        border-top-color: rgba(15, 23, 42, 0.9);
        animation: spin 0.7s linear infinite;
      }

      .meta {
        font-size: 0.85rem;
        color: var(--muted);
        margin-bottom: 0.4rem;
      }

      .meta-strong {
        color: var(--accent);
      }

      .result-card {
        background: var(--card-bg);
        border-radius: var(--radius-lg);
        border: 1px solid rgba(148, 163, 184, 0.25);
        padding: 1.1rem 1.25rem 1.25rem;
      }

      .result-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 0.75rem;
        margin-bottom: 0.45rem;
      }

      .result-header h2 {
        margin: 0;
        font-size: 1.02rem;
      }

      .pill {
        font-size: 0.8rem;
        border-radius: 999px;
        padding: 0.15rem 0.6rem;
        background: var(--accent-soft);
        border: 1px solid rgba(56, 189, 248, 0.6);
        color: var(--accent);
      }

      .pill-danger {
        border-color: rgba(248, 113, 113, 0.7);
        color: var(--danger);
        background: rgba(248, 113, 113, 0.08);
      }

      pre {
        white-space: pre-wrap;
        word-wrap: break-word;
        padding: 0.85rem 0.9rem;
        background: #020617;
        border-radius: var(--radius-md);
        border: 1px solid rgba(15, 23, 42, 0.9);
        font-size: 0.88rem;
        max-height: 480px;
        overflow: auto;
      }

      .examples {
        font-size: 0.8rem;
        padding: 0.95rem 1.1rem;
        border-radius: var(--radius-md);
        background: rgba(15, 23, 42, 0.85);
        border: 1px dashed rgba(148, 163, 184, 0.6);
      }

      .examples ul {
        margin: 0.45rem 0 0;
        padding-left: 1.1rem;
      }

      .examples li {
        margin-bottom: 0.25rem;
      }

      @keyframes spin {
        to { transform: rotate(360deg); }
      }
    </style>
    <script>
      function handleSubmit(form) {
        const button = form.querySelector("button[type=submit]");
        const spinner = document.getElementById("spinner");
        if (!button || !spinner) return true;
        button.disabled = true;
        spinner.style.display = "inline-block";
        return true;
      }
    </script>
  </head>
  <body>
    <div class="shell">
      <div class="panel">
        <header class="header">
          <div class="title-block">
            <h1>Synthetic Clinical Data Generator</h1>
            <p>FHIR-aware RAG agent with web tools and optional bundle POST.</p>
          </div>
          <div class="badge-row">
            <span class="badge badge-accent">FHIR schema + examples</span>
            <span class="badge">DuckDuckGo · Wikipedia · PubMed · Arxiv</span>
          </div>
        </header>

        <main>
          <section>
            <form method="post" action="/generate" onsubmit="return handleSubmit(this);">
              <div>
                <div class="field-label">Patient scenario</div>
                <textarea
                  id="query"
                  name="query"
                  required
                  placeholder='e.g. "Generate FHIR data for a 60-year-old female with osteoporosis."'
                ></textarea>
                <div class="hint">
                  Examples:
                  <code>Generate natural language data for a 45-year-old male with type 2 diabetes and hypertension.</code>
                </div>
              </div>

              <div style="margin-top: 1rem;">
                <div class="field-label">FHIR endpoint URL (optional)</div>
                <input
                  id="endpoint_url"
                  name="endpoint_url"
                  class="endpoint-input"
                  placeholder="https://example.com/fhir"
                  value="{{ default_endpoint or '' }}"
                />
                <div class="hint">
                  If provided, the generated bundle will be POSTed as
                  <code>application/fhir+json</code> to this endpoint.
                </div>
              </div>

              <div class="actions">
                <button type="submit">
                  <span id="spinner" class="spinner" style="display:none;"></span>
                  <span>Generate Clinical Data</span>
                </button>
                <div class="examples">
                  <strong>Tip:</strong> Ask explicitly for <code>FHIR</code> or
                  <code>natural language</code> output and include key conditions,
                  medications, and demographics.
                </div>
              </div>
            </form>
          </section>

          {% if response %}
            <section class="result-card">
              <div class="result-header">
                <h2>Generated Clinical Data</h2>
                {% if post_status %}
                  <span class="pill{% if 'Failed' in post_status %} pill-danger{% endif %}">
                    {{ post_status }}
                  </span>
                {% else %}
                  <span class="pill">Preview only – not POSTed</span>
                {% endif %}
              </div>

              <div class="meta">
                <span class="meta-strong">Web tools used:</span>
                {{ tools|join(", ") if tools else "none" }}
              </div>

              <pre>{{ response }}</pre>
            </section>
          {% endif %}
        </main>
      </div>
    </div>
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
        return render_template_string(
            _HTML_TEMPLATE,
            response=None,
            tools=[],
            post_status=None,
            default_endpoint=settings.get("fhir_endpoint"),
        )

    @app.post("/generate")
    def generate():
        json_body = None
        if request.is_json:
            json_body = request.get_json(silent=True) or {}

        query = request.form.get("query") if not json_body else json_body.get("query")
        endpoint_url = (
            request.form.get("endpoint_url")
            if not json_body
            else json_body.get("endpoint_url")
        )

        if not query:
            if request.is_json:
                return jsonify({"error": "Missing 'query'"}), 400
            return render_template_string(
                _HTML_TEMPLATE,
                response=None,
                tools=[],
                post_status=None,
                default_endpoint=settings.get("fhir_endpoint"),
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

        post_status: str | None = None

        try:
            response = rag.invoke(augmented_query)
        except Exception as exc:  # pragma: no cover - simple error surface
            if request.is_json:
                return jsonify({"error": str(exc)}), 500
            return render_template_string(
                _HTML_TEMPLATE,
                response=f"Error during generation: {exc}",
                tools=tool_names,
                post_status=None,
                default_endpoint=settings.get("fhir_endpoint"),
            )

        if endpoint_url:
            try:
                http_response = requests.post(
                    endpoint_url,
                    data=response,
                    headers={"Content-Type": "application/fhir+json"},
                    timeout=15,
                )
                post_status = (
                    f"POSTed bundle to {endpoint_url} "
                    f"(status {http_response.status_code})."
                )
            except Exception as exc:
                post_status = f"Failed to POST bundle to {endpoint_url}: {exc}"

        if request.is_json:
            return jsonify(
                {
                    "response": response,
                    "tools_used": tool_names,
                    "endpoint_url": endpoint_url,
                    "post_status": post_status,
                }
            )

        return render_template_string(
            _HTML_TEMPLATE,
            response=response,
            tools=tool_names,
            post_status=post_status,
            default_endpoint=settings.get("fhir_endpoint"),
        )

    return app


def run_web(host: str = "127.0.0.1", port: int = 5000):
    app = create_app()
    app.run(host=host, port=port)


if __name__ == "__main__":
    run_web()
