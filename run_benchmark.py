"""
Benchmark script: Run 5 queries against the Vietnamese Criminal Code
using different chunking strategies and embedding backends.

Usage:
    # Mock embeddings (no extra install needed):
    python run_benchmark.py

    # Local embeddings (requires: pip install sentence-transformers):
    python run_benchmark.py --embedder local

    # Choose chunking strategy:
    python run_benchmark.py --strategy fixed
    python run_benchmark.py --strategy sentence
    python run_benchmark.py --strategy recursive
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

# Fix Windows console encoding for Vietnamese text
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.chunking import FixedSizeChunker, RecursiveChunker, SentenceChunker
from src.embeddings import LocalEmbedder, MockEmbedder, _mock_embed
from src.models import Document
from src.store import EmbeddingStore

DATA_FILE = Path(__file__).parent / "data" / "Bo-luat-hinh-su.doc.md"

BENCHMARK_QUERIES = [
    {
        "id": "Q1",
        "query": "Người từ đủ 14 tuổi đến dưới 16 tuổi phải chịu trách nhiệm hình sự về tội gì?",
        "gold_article": "Điều 12",
        "gold_answer": (
            "Người từ đủ 14 tuổi đến dưới 16 tuổi phải chịu trách nhiệm hình sự "
            "về tội phạm rất nghiêm trọng, tội phạm đặc biệt nghiêm trọng quy định "
            "tại một trong các điều 123, 134, 141, 142, 143, 144, 150, 151, 168, 169, "
            "170, 171, 173, 178, 248, 249, 250, 251, 252, 265, 266, 286, 287, 289, "
            "290, 299, 303, 304 của Bộ luật này."
        ),
        "type": "Factual",
    },
    {
        "id": "Q2",
        "query": "Hình phạt cho tội trộm cắp tài sản có giá trị từ 500 triệu đồng trở lên là gì?",
        "gold_article": "Điều 173 khoản 4",
        "gold_answer": (
            "Phạm tội thuộc trường hợp chiếm đoạt tài sản trị giá 500.000.000 đồng trở lên: "
            "bị phạt tù từ 12 năm đến 20 năm."
        ),
        "type": "Factual + specific amount",
    },
    {
        "id": "Q3",
        "query": "Các tình tiết giảm nhẹ trách nhiệm hình sự được quy định như thế nào?",
        "gold_article": "Điều 51",
        "gold_answer": (
            "Điều 51 liệt kê 22 tình tiết giảm nhẹ, bao gồm: người phạm tội đã ngăn chặn "
            "hoặc làm giảm bớt tác hại; tự nguyện sửa chữa, bồi thường; phạm tội trong "
            "trường hợp vượt quá giới hạn phòng vệ chính đáng; phạm tội lần đầu và thuộc "
            "trường hợp ít nghiêm trọng; phạm tội vì bị người khác đe dọa hoặc cưỡng bức; "
            "v.v. Ngoài ra Tòa án có thể coi các tình tiết khác là tình tiết giảm nhẹ."
        ),
        "type": "Multi-chunk",
    },
    {
        "id": "Q4",
        "query": "Người gây tai nạn giao thông rồi bỏ chạy bị xử phạt như thế nào?",
        "gold_article": "Điều 260",
        "gold_answer": (
            "Vi phạm quy định về tham gia giao thông đường bộ gây hậu quả rồi bỏ trốn "
            "hoặc không cứu giúp: phạt tù từ 03 năm đến 10 năm (khoản 2). "
            "Nếu gây hậu quả đặc biệt nghiêm trọng: phạt tù từ 07 năm đến 15 năm (khoản 3)."
        ),
        "type": "Reasoning",
    },
    {
        "id": "Q5",
        "query": "Tội phạm được phân loại thành mấy loại? Mỗi loại có mức hình phạt tối đa bao nhiêu?",
        "gold_article": "Điều 9",
        "gold_answer": (
            "4 loại: (1) Tội phạm ít nghiêm trọng - phạt tiền, cải tạo không giam giữ "
            "hoặc phạt tù đến 03 năm; (2) Tội phạm nghiêm trọng - phạt tù từ trên 03 năm "
            "đến 07 năm; (3) Tội phạm rất nghiêm trọng - phạt tù từ trên 07 năm đến 15 năm; "
            "(4) Tội phạm đặc biệt nghiêm trọng - phạt tù từ trên 15 năm đến 20 năm, "
            "tù chung thân hoặc tử hình."
        ),
        "type": "Synthesis",
    },
]

STRATEGIES = {
    "fixed": lambda cs: FixedSizeChunker(chunk_size=cs, overlap=100),
    "sentence": lambda _: SentenceChunker(max_sentences_per_chunk=5),
    "recursive": lambda cs: RecursiveChunker(chunk_size=cs),
}


def load_document() -> str:
    return DATA_FILE.read_text(encoding="utf-8")


def run_benchmark(strategy_name: str, chunk_size: int, embedder_name: str) -> None:
    # --- Select embedder ---
    if embedder_name == "local":
        print("Loading local embedding model (all-MiniLM-L6-v2)... ", end="", flush=True)
        t0 = time.time()
        embed_fn = LocalEmbedder()
        print(f"done in {time.time() - t0:.1f}s")
    else:
        embed_fn = MockEmbedder()
        print(f"Using mock embeddings (64-dim deterministic)")

    # --- Load & chunk ---
    print(f"\nStrategy: {strategy_name} | chunk_size={chunk_size}")
    text = load_document()
    print(f"Document size: {len(text):,} chars")

    chunker = STRATEGIES[strategy_name](chunk_size)
    chunks = chunker.chunk(text)
    print(f"Chunks created: {len(chunks)}")
    avg_len = sum(len(c) for c in chunks) / len(chunks) if chunks else 0
    print(f"Avg chunk length: {avg_len:.0f} chars")

    # --- Build store ---
    docs = [
        Document(
            id=f"blhs_chunk_{i}",
            content=chunk,
            metadata={"source": "Bo-luat-hinh-su", "chunk_index": i},
        )
        for i, chunk in enumerate(chunks)
    ]

    print(f"\nEmbedding {len(docs)} chunks... ", end="", flush=True)
    t0 = time.time()
    store = EmbeddingStore(collection_name="benchmark", embedding_fn=embed_fn)
    store.add_documents(docs)
    embed_time = time.time() - t0
    print(f"done in {embed_time:.1f}s")
    print(f"Store size: {store.get_collection_size()}")

    # --- Run queries ---
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    for bq in BENCHMARK_QUERIES:
        print(f"\n--- {bq['id']}: {bq['query']}")
        print(f"    Gold: {bq['gold_article']} | Type: {bq['type']}")

        t0 = time.time()
        results = store.search(bq["query"], top_k=3)
        search_time = time.time() - t0

        for rank, r in enumerate(results, 1):
            content_preview = r["content"][:200].replace("\n", " ")
            print(f"    Top-{rank} (score={r['score']:.4f}): {content_preview}...")

        gold_found = any(bq["gold_article"].split()[0] in r["content"] for r in results)
        status = "FOUND" if gold_found else "MISSED"
        print(f"    >> Gold article in top-3: {status} | Search time: {search_time*1000:.1f}ms")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Embedder:  {embedder_name}")
    print(f"Strategy:  {strategy_name}")
    print(f"Chunks:    {len(chunks)}")
    print(f"Avg len:   {avg_len:.0f}")
    print(f"Embed time: {embed_time:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark queries on Bo luat Hinh su")
    parser.add_argument(
        "--strategy",
        choices=["fixed", "sentence", "recursive"],
        default="recursive",
        help="Chunking strategy (default: recursive)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size in chars (default: 500, ignored for sentence strategy)",
    )
    parser.add_argument(
        "--embedder",
        choices=["mock", "local"],
        default="mock",
        help="Embedding backend (default: mock)",
    )
    args = parser.parse_args()
    run_benchmark(args.strategy, args.chunk_size, args.embedder)


if __name__ == "__main__":
    main()
