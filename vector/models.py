from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    id: str
    patient_id: str
    source_id: str
    chunk_index: int
    text: str
    modality: str
    timestamp: Optional[datetime]
    metadata: Dict[str, Any]
    embedding: List[float]


@dataclass
class SearchFilters:
    patient_ids: Optional[List[str]] = None
    modalities: Optional[List[str]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    source_ids: Optional[List[str]] = None


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


