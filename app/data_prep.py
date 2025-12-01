from __future__ import annotations

import json
import os
import re
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_fhir_schema(content_dir: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tgz_file_path = content_dir / "package.tgz"
    with tarfile.open(tgz_file_path, "r:gz") as tar:
        tar.extractall(path=output_dir, filter="data")
    return output_dir


def extract_fhir_examples(content_dir: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_file_path = content_dir / "examples.json.zip"
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    return output_dir


def clean_json_content(data):
    if isinstance(data, dict):
        return {k: clean_json_content(v) for k, v in data.items() if v is not None}
    if isinstance(data, list):
        return [clean_json_content(elem) for elem in data if elem is not None]
    if isinstance(data, str):
        cleaned_str = re.sub(r"\b(null|undefined)\b", "", data, flags=re.IGNORECASE)
        cleaned_str = re.sub(r"\s+", " ", cleaned_str).strip()
        return cleaned_str
    return data


def load_and_clean_json_files(root_dir: Path) -> List[str]:
    documents: List[str] = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.startswith(".") or not file.endswith(".json"):
                continue
            file_path = Path(root) / file
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    json_data = json.load(f)
                cleaned_data = clean_json_content(json_data)
                documents.append(json.dumps(cleaned_data, separators=(",", ":")))
            except json.JSONDecodeError:
                continue
    return documents


def make_chunks(documents: Iterable[str]) -> List[Document]:
    docs = [Document(page_content=doc) for doc in documents]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

