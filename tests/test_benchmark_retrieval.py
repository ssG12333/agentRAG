"""检索基准脚本的最小可复现测试。"""

import csv

from scripts.benchmark_retrieval import run_benchmark, write_csv


def test_retrieval_benchmark_smoke_and_csv(tmp_path):
    rows = run_benchmark(
        n_vectors=128,
        dim=16,
        n_queries=8,
        top_k=5,
        n_clusters=8,
        n_probe=4,
        n_subvectors=4,
        n_bits=3,
        n_iters=4,
        seed=7,
    )

    assert [row["backend"] for row in rows] == ["numpy_exact", "ivfpq"]
    assert rows[0]["recall_at_k"] == 1.0
    assert 0.0 <= rows[1]["recall_at_k"] <= 1.0
    assert rows[1]["index_data_bytes"] < rows[0]["index_data_bytes"]
    assert rows[0]["query_avg_ms"] >= 0.0
    assert rows[1]["query_avg_ms"] >= 0.0

    output = tmp_path / "benchmark.csv"
    write_csv(rows, str(output))
    with output.open(encoding="utf-8-sig", newline="") as handle:
        restored = list(csv.DictReader(handle))
    assert len(restored) == 2
    assert restored[1]["backend"] == "ivfpq"
