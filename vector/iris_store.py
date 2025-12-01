from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from .base import VectorStore
from .models import Chunk, SearchFilters, SearchResult


class IrisVectorStore(VectorStore):
    """
    VectorStore implementation backed by InterSystems IRIS for Health using
    DB-API (intersystems-irispython).

    This version is designed to use native IRIS vector storage and similarity:
      - Embeddings are stored in a native vector column.
      - Similarity is computed inside IRIS using a configurable SQL expression.

    Because IRIS vector SQL syntax can vary by version, this class exposes a
    configurable `similarity_sql` expression that you can adjust to match your
    deployment (for example, a cosine similarity function).

    It assumes a table like:

        clinical_chunks(
            id            VARCHAR PRIMARY KEY,
            patient_id    VARCHAR NOT NULL,
            source_id     VARCHAR NOT NULL,
            chunk_index   INTEGER NOT NULL,
            text          VARCHAR(MAX) NOT NULL,
            modality      VARCHAR NOT NULL,
            ts            TIMESTAMP NULL,
            metadata_json VARCHAR(MAX) NULL,
            embedding     VECTOR
        )

    You may need to adjust the DDL in `initialise_schema` and the
    `similarity_sql` default to match the exact IRIS for Health version and
    configuration you are using.
    """

    def __init__(
        self,
        connection,
        *,
        table_name: str = "clinical_chunks",
        similarity_sql: Optional[str] = None,
    ) -> None:
        """
        connection: a live DB-API connection from intersystems-irispython,
        e.g. iris.connect(...).

        table_name: name of the table that stores chunks.

        similarity_sql: SQL expression used to compute similarity between
        the stored embedding column and a parameterized query embedding.
        This must be a valid IRIS SQL expression that returns a numeric
        "higher is more similar" score. The expression should contain a
        single positional placeholder (`?`) for the query embedding, for
        example:

            'MY_COSINE_SIMILARITY_FUNC(embedding, ?)'

        If None, a generic placeholder expression is used and will need
        to be customized before use.
        """
        self._conn = connection
        self._table_name = table_name
        # Default expression is a placeholder and likely needs to be
        # replaced with a real IRIS vector similarity function.
        self._similarity_sql = similarity_sql or "/* TODO: set similarity SQL, e.g. COSINE_SIM(embedding, ?) */ 0"

    def initialise_schema(self) -> None:
        """
        Create the clinical_chunks table and basic indexes if they do not exist.

        This is safe to call repeatedly at application startup.
        """
        table_sql = f"""
            CREATE TABLE {self._table_name} (
                id            VARCHAR(255) PRIMARY KEY,
                patient_id    VARCHAR(255) NOT NULL,
                source_id     VARCHAR(255) NOT NULL,
                chunk_index   INTEGER NOT NULL,
                text          VARCHAR(4000) NOT NULL,
                modality      VARCHAR(255) NOT NULL,
                ts            TIMESTAMP NULL,
                metadata_json VARCHAR(4000) NULL,
                embedding     VECTOR
            )
        """

        index_sql_statements = [
            f"CREATE INDEX {self._table_name}_patient_idx ON {self._table_name} (patient_id)",
            f"CREATE INDEX {self._table_name}_source_idx ON {self._table_name} (source_id)",
            f"CREATE INDEX {self._table_name}_modality_idx ON {self._table_name} (modality)",
            f"CREATE INDEX {self._table_name}_ts_idx ON {self._table_name} (ts)",
        ]

        with self._conn.cursor() as cur:
            self._execute_ddl_ignore_exists(cur, table_sql)
            for stmt in index_sql_statements:
                self._execute_ddl_ignore_exists(cur, stmt)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return

        # Simple strategy: delete existing ids, then insert fresh rows.
        ids = [c.id for c in chunks]
        self._delete_by_ids(ids)
        self._insert_chunks(chunks)

    def delete_by_source(self, source_id: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self._table_name} WHERE source_id = ?", (source_id,))
        self._conn.commit()

    def delete_by_patient(self, patient_id: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self._table_name} WHERE patient_id = ?", (patient_id,))
        self._conn.commit()

    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filters: Optional[SearchFilters] = None,
    ) -> List[SearchResult]:
        rows = self._select_rows_with_db_similarity(filters, query_embedding, k)
        return [
            SearchResult(
                chunk=self._row_to_chunk(row, row["embedding"]),
                score=row["score"],
            )
            for row in rows
        ]

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 10,
        filters: Optional[SearchFilters] = None,
    ) -> List[SearchResult]:
        rows = self._select_rows_with_db_similarity(filters, embedding, k)
        return [
            SearchResult(
                chunk=self._row_to_chunk(row, row["embedding"]),
                score=row["score"],
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _execute_ddl_ignore_exists(cur, sql: str) -> None:
        """
        Execute a DDL statement, ignoring "already exists" errors.

        This keeps initialize_schema idempotent without relying on
        dialect-specific IF NOT EXISTS syntax.
        """
        try:
            cur.execute(sql)
        except Exception as exc:  # pragma: no cover - defensive
            message = str(exc).lower()
            if "already exists" in message or "exists" in message:
                return
            raise

    def _delete_by_ids(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        sql = f"DELETE FROM {self._table_name} WHERE id IN ({placeholders})"
        with self._conn.cursor() as cur:
            cur.execute(sql, tuple(ids))
        self._conn.commit()

    def _insert_chunks(self, chunks: List[Chunk]) -> None:
        sql = f"""
            INSERT INTO {self._table_name} (
                id,
                patient_id,
                source_id,
                chunk_index,
                text,
                modality,
                ts,
                metadata_json,
                embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = []
        for c in chunks:
            metadata_json = json.dumps(c.metadata or {})
            ts_value = c.timestamp
            # DB-API adapters usually handle datetime objects directly.
            params.append(
                (
                    c.id,
                    c.patient_id,
                    c.source_id,
                    c.chunk_index,
                    c.text,
                    c.modality,
                    ts_value,
                    metadata_json,
                    c.embedding,
                )
            )

        with self._conn.cursor() as cur:
            cur.executemany(sql, params)
        self._conn.commit()

    def _select_rows_with_db_similarity(
        self,
        filters: Optional[SearchFilters],
        query_embedding: List[float],
        k: int,
    ) -> List[Dict[str, Any]]:
        """
        Select rows from IRIS and compute similarity using a native vector
        function inside SQL.

        The exact similarity behavior is governed by `self._similarity_sql`,
        which should be a valid IRIS SQL expression, for example:

            'MY_COSINE_SIMILARITY_FUNC(embedding, ?)'

        The query embedding is passed in as the parameter for the placeholder.
        """
        base_sql = f"""
            SELECT
                id,
                patient_id,
                source_id,
                chunk_index,
                text,
                modality,
                ts,
                metadata_json,
                embedding,
                {self._similarity_sql} AS score
            FROM {self._table_name}
        """
        where_clauses: List[str] = []
        params: List[Any] = []

        if filters:
            if filters.patient_ids:
                placeholders = ",".join("?" for _ in filters.patient_ids)
                where_clauses.append(f"patient_id IN ({placeholders})")
                params.extend(filters.patient_ids)
            if filters.modalities:
                placeholders = ",".join("?" for _ in filters.modalities)
                where_clauses.append(f"modality IN ({placeholders})")
                params.extend(filters.modalities)
            if filters.source_ids:
                placeholders = ",".join("?" for _ in filters.source_ids)
                where_clauses.append(f"source_id IN ({placeholders})")
                params.extend(filters.source_ids)
            if filters.date_from:
                where_clauses.append("ts >= ?")
                params.append(filters.date_from)
            if filters.date_to:
                where_clauses.append("ts <= ?")
                params.append(filters.date_to)

        if where_clauses:
            base_sql += " WHERE " + " AND ".join(where_clauses)

        # Append ORDER BY and limit; the similarity expression uses the
        # query embedding as a parameter.
        base_sql += " ORDER BY score DESC"
        base_sql += " FETCH FIRST ? ROWS ONLY"

        params.append(query_embedding)
        params.append(k)

        with self._conn.cursor() as cur:
            cur.execute(base_sql, tuple(params))
            columns = [desc[0] for desc in cur.description]
            rows = [dict(zip(columns, row)) for row in cur.fetchall()]

        return rows

    @staticmethod
    def _row_to_chunk(row: Dict[str, Any], embedding: List[float]) -> Chunk:
        metadata_raw = row.get("metadata_json")
        metadata = json.loads(metadata_raw) if metadata_raw else {}
        ts = row.get("ts")
        if isinstance(ts, str):
            # Fallback: attempt simple ISO parsing; adjust if needed.
            try:
                ts = datetime.fromisoformat(ts)
            except ValueError:
                ts = None

        return Chunk(
            id=row["id"],
            patient_id=row["patient_id"],
            source_id=row["source_id"],
            chunk_index=int(row["chunk_index"]),
            text=row["text"],
            modality=row["modality"],
            timestamp=ts,
            metadata=metadata,
            embedding=embedding,
        )

    @staticmethod
    def _l2_norm(vec: List[float]) -> float:
        return math.sqrt(sum(x * x for x in vec))

    @classmethod
    def _cosine_similarity(
        cls,
        a: List[float],
        b: List[float],
        norm_a: Optional[float] = None,
    ) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        norm_a = norm_a if norm_a is not None else cls._l2_norm(a)
        norm_b = cls._l2_norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        return dot / (norm_a * norm_b)
