from __future__ import annotations

from typing import Iterable, List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


def build_vector_store(documents: Iterable[Document], openai_api_key: str) -> FAISS:
    embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.from_documents(list(documents), embedding_model)

