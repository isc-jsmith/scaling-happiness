from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from .models import Chunk, SearchFilters, SearchResult


class VectorStore(ABC):
    """Abstract interface for a clinical vector store."""

    @abstractmethod
    def upsert_chunks(self, chunks: List[Chunk]) -> None:
        """Insert or update a batch of chunks."""

    @abstractmethod
    def delete_by_source(self, source_id: str) -> None:
        """Delete all chunks tied to a given source document."""

    @abstractmethod
    def delete_by_patient(self, patient_id: str) -> None:
        """Delete all chunks for a given patient."""

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filters: Optional[SearchFilters] = None,
    ) -> List[SearchResult]:
        """Vector search using a query embedding."""

    @abstractmethod
    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 10,
        filters: Optional[SearchFilters] = None,
    ) -> List[SearchResult]:
        """Vector search using an existing embedding."""


