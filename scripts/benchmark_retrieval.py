"""可复现的 Numpy 精确检索与残差 IVF-PQ 合成基准。"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document import Chunk
from src.index.ivfpq_store import IVFPQVectorStore
from src.index.vector_store import NumpyVectorStore


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.ascontiguousarray(
        vectors / np.where(norms == 0, 1.0, norms),
        dtype=np.float32,
    )


def run_benchmark(
    *,
    n_vectors: int,
    dim: int,
    n_queries: int,
    top_k: int,
    n_clusters: int,
    n_probe: int,
    n_subvectors: int,
    n_bits: int,
    n_iters: int,
    seed: int,
) -> list[dict]:
    if min(
        n_vectors, dim, n_queries, top_k, n_clusters, n_probe,
        n_subvectors, n_bits, n_iters,
    ) <= 0:
        raise ValueError("所有基准参数必须为正数")
    if dim % n_subvectors != 0:
        raise ValueError("dim 必须能被 n_subvectors 整除")
    if n_bits > 8:
        raise ValueError("n_bits 必须小于等于 8")
    if top_k > n_vectors:
        raise ValueError("top_k 不能超过 n_vectors")

    rng = np.random.default_rng(seed)
    vectors = _normalize(rng.normal(size=(n_vectors, dim)).astype(np.float32))
    query_ids = rng.integers(0, n_vectors, size=n_queries)
    queries = _normalize(
        vectors[query_ids]
        + rng.normal(0, 0.01, size=(n_queries, dim)).astype(np.float32)
    )
    chunks = [
        Chunk(id=str(i), document_id="benchmark", content=f"vector {i}")
        for i in range(n_vectors)
    ]

    exact = NumpyVectorStore()
    started = perf_counter()
    exact.add(vectors, chunks)
    exact_build_ms = (perf_counter() - started) * 1000.0

    approximate = IVFPQVectorStore(
        n_clusters=n_clusters,
        n_probe=n_probe,
        n_subvectors=n_subvectors,
        n_bits=n_bits,
        n_iters=n_iters,
    )
    started = perf_counter()
    approximate.add(vectors, chunks)
    approximate_build_ms = (perf_counter() - started) * 1000.0

    # 预热，避免首次 Python/C++ 调度影响计时。
    exact.search(queries[0], top_k=top_k)
    approximate.search(queries[0], top_k=top_k)

    exact_ids = []
    started = perf_counter()
    for query in queries:
        exact_ids.append([
            chunk.id for chunk, _ in exact.search(query, top_k=top_k)
        ])
    exact_query_ms = (perf_counter() - started) * 1000.0

    approximate_ids = []
    started = perf_counter()
    for query in queries:
        approximate_ids.append([
            chunk.id for chunk, _ in approximate.search(query, top_k=top_k)
        ])
    approximate_query_ms = (perf_counter() - started) * 1000.0

    recall = float(np.mean([
        len(set(expected) & set(actual)) / top_k
        for expected, actual in zip(exact_ids, approximate_ids)
    ]))

    common = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seed": seed,
        "n_vectors": n_vectors,
        "dim": dim,
        "n_queries": n_queries,
        "top_k": top_k,
        "n_clusters": min(n_clusters, n_vectors),
        "n_probe": min(n_probe, n_clusters, n_vectors),
        "n_subvectors": n_subvectors,
        "n_bits": n_bits,
        "n_iters": n_iters,
    }

    def row(backend, build_ms, query_ms, recall_at_k, memory_bytes):
        return {
            **common,
            "backend": backend,
            "build_ms": round(build_ms, 6),
            "query_total_ms": round(query_ms, 6),
            "query_avg_ms": round(query_ms / n_queries, 6),
            "queries_per_second": round(n_queries / (query_ms / 1000.0), 6),
            "recall_at_k": round(recall_at_k, 6),
            "index_data_bytes": int(memory_bytes),
        }

    return [
        row("numpy_exact", exact_build_ms, exact_query_ms, 1.0, vectors.nbytes),
        row(
            "ivfpq",
            approximate_build_ms,
            approximate_query_ms,
            recall,
            approximate.estimated_memory_bytes,
        ),
    ]


def write_csv(rows: list[dict], output: str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-vectors", type=int, default=5000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--n-queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--n-clusters", type=int, default=64)
    parser.add_argument("--n-probe", type=int, default=8)
    parser.add_argument("--n-subvectors", type=int, default=8)
    parser.add_argument("--n-bits", type=int, default=4)
    parser.add_argument("--n-iters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        rows = run_benchmark(
            n_vectors=args.n_vectors,
            dim=args.dim,
            n_queries=args.n_queries,
            top_k=args.top_k,
            n_clusters=args.n_clusters,
            n_probe=args.n_probe,
            n_subvectors=args.n_subvectors,
            n_bits=args.n_bits,
            n_iters=args.n_iters,
            seed=args.seed,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"benchmark error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(rows, ensure_ascii=False, indent=2))
    if args.output:
        write_csv(rows, args.output)
        print(f"CSV: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
