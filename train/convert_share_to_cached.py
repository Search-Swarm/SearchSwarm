#!/usr/bin/env python3
"""Convert the SearchSwarm-SFT bundles into flat ms-swift training data.

SearchSwarm-SFT (https://huggingface.co/datasets/SearchSwarm/SearchSwarm-SFT)
ships a single ``train.parquet`` with one bundle per row — a main-agent
conversation plus the sub-agent conversations it dispatched::

    {"source": "redsearcher" | "openseeker",
     "question": "<main task question>",
     "answer": "<ground-truth answer>",
     "messages": [{"role": "system"|"user"|"assistant", "content": "..."}, ...],
     "subagents": [{"question": "<sub-agent briefing>", "messages": [...]}, ...],
     "num_subagents": <int>}

A bundle holds ``1 + num_subagents`` independent conversations, so it is not
directly trainable. This script unrolls every bundle into flat ms-swift
records — ``{"messages": [...]}``, one line per main trajectory and one per
sub-agent trajectory. Conversations are stored already normalized (system
prompt folded into a leading system message, roles limited to
system/user/assistant, every trajectory ends on an assistant message), so
splitting is the only transformation.

The parquet must be read streaming: it is a single ~2.1 GB row group whose
nested sub-agent content column decompresses to ~5.8 GB, beyond Arrow's 2 GB
per-chunk string limit. Whole-file readers (``pandas.read_parquet``,
``pyarrow.parquet.read_table``, a plain ``datasets.load_dataset``) fail with
"ArrowNotImplementedError: Nested data conversions not supported for chunked
array outputs" or exhaust memory. ``ParquetFile.iter_batches`` decodes the
file incrementally and keeps peak memory at a few hundred MB.

Usage::

    python convert_share_to_cached.py --parquet train.parquet --out data.jsonl

    # Optionally chain ms-swift pre-tokenization (same as running
    # ``swift export --to_cached_dataset true`` on the JSONL yourself):
    python convert_share_to_cached.py --parquet train.parquet --out data.jsonl \
        --cached-dataset-dir ./cached_dataset
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

try:
    import pyarrow.parquet as pq
except ImportError:
    sys.exit("error: pyarrow is required (pip install pyarrow)")

TOKENIZER_MODEL = "Qwen/Qwen3-30B-A3B-Thinking-2507"
MAX_LENGTH = 131072


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unroll SearchSwarm-SFT bundles into flat ms-swift training records."
    )
    parser.add_argument("--parquet", required=True,
                        help="train.parquet from SearchSwarm/SearchSwarm-SFT")
    parser.add_argument("--out", default="data.jsonl",
                        help="output JSONL of flat ms-swift records (default: data.jsonl)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="bundles decoded per batch while streaming (default: 32)")
    parser.add_argument("--cached-dataset-dir", default=None,
                        help="if set, run `swift export --to_cached_dataset true` on the "
                             "JSONL afterwards (requires ms-swift)")
    parser.add_argument("--dataset-num-proc", type=int,
                        default=min(64, os.cpu_count() or 1),
                        help="worker count for swift export (default: min(64, cpus))")
    return parser.parse_args()


def write_record(out, messages) -> None:
    out.write(json.dumps({"messages": messages}, ensure_ascii=False,
                         separators=(",", ":")) + "\n")


def main() -> int:
    args = parse_args()
    pf = pq.ParquetFile(args.parquet)
    missing = {"messages", "subagents"} - set(pf.schema_arrow.names)
    if missing:
        sys.exit(f"error: {args.parquet} is missing column(s) {sorted(missing)}; "
                 "expected the SearchSwarm-SFT train.parquet")

    total = pf.metadata.num_rows
    bundles = mains = subs = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for batch in pf.iter_batches(batch_size=args.batch_size,
                                     columns=["messages", "subagents"]):
            for row in batch.to_pylist():
                write_record(out, row["messages"])
                mains += 1
                for sub in row["subagents"]:
                    write_record(out, sub["messages"])
                    subs += 1
                bundles += 1
                if bundles % 500 == 0:
                    print(f"[convert] {bundles}/{total} bundles", flush=True)
    print(f"[convert] {bundles} bundles -> {mains} main + {subs} sub-agent "
          f"= {mains + subs} training records -> {args.out}")

    if args.cached_dataset_dir:
        cmd = [
            "swift", "export",
            "--model", TOKENIZER_MODEL,
            "--dataset", args.out,
            "--max_length", str(MAX_LENGTH),
            "--to_cached_dataset", "true",
            "--output_dir", args.cached_dataset_dir,
            "--dataset_num_proc", str(args.dataset_num_proc),
        ]
        print(f"[convert] {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            return rc
        print(f"[convert] cached dataset ready -> use "
              f"DATA_PATH={args.cached_dataset_dir}/train")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
