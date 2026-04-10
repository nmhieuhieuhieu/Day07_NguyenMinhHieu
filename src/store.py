from __future__ import annotations
from typing import Any, Callable
from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb
            client = chromadb.Client()
            self._collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        embedding = self._embedding_fn(doc.content)
        return {
            "id":        str(self._next_index),
            "content":   doc.content,
            "embedding": embedding,
            "metadata":  {
                "doc_id": doc.id,          # ← sửa doc.doc_id → doc.id
                **(doc.metadata or {}),
            },
        }

    def _search_records(
        self,
        query: str,
        records: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not records:
            return []
        query_vec = self._embedding_fn(query)
        scored = [
            (record, _dot(query_vec, record["embedding"]))
            for record in records
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            {**record, "score": round(score, 6)}
            for record, score in scored[:top_k]
        ]

    def add_documents(self, docs: list[Document]) -> None:
        for doc in docs:
            record = self._make_record(doc)
            self._next_index += 1
            if self._use_chroma:
                self._collection.add(
                    ids=[record["id"]],
                    documents=[record["content"]],
                    embeddings=[record["embedding"]],
                    metadatas=[record["metadata"]],
                )
            else:
                self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self._use_chroma:
            query_vec = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_vec],
                n_results=min(top_k, self._collection.count()),
            )
            return [
                {
                    "id":       results["ids"][0][i],
                    "content":  results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score":    round(1 - results["distances"][0][i], 6),
                }
                for i in range(len(results["ids"][0]))
            ]
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        if self._use_chroma:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        if self._use_chroma:
            query_vec = self._embedding_fn(query)
            kwargs: dict[str, Any] = {
                "query_embeddings": [query_vec],
                "n_results": min(top_k, self._collection.count()),
            }
            if metadata_filter:
                kwargs["where"] = metadata_filter
            results = self._collection.query(**kwargs)
            return [
                {
                    "id":       results["ids"][0][i],
                    "content":  results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score":    round(1 - results["distances"][0][i], 6),
                }
                for i in range(len(results["ids"][0]))
            ]

        if metadata_filter:
            filtered = [
                record for record in self._store
                if all(
                    record["metadata"].get(k) == v
                    for k, v in metadata_filter.items()
                )
            ]
        else:
            filtered = self._store

        return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        if self._use_chroma:
            results = self._collection.get(where={"doc_id": doc_id})
            ids_to_delete = results.get("ids", [])
            if ids_to_delete:
                self._collection.delete(ids=ids_to_delete)
                return True
            return False

        original_count = len(self._store)
        self._store = [
            record for record in self._store
            if record["metadata"].get("doc_id") != doc_id
        ]
        return len(self._store) < original_count