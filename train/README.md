# SearchSwarm Training Scripts

Full-parameter SFT for **SearchSwarm-30B-A3B**, run through ms-swift's Megatron
backend — the `megatron sft` CLI, also called **Megatron-SWIFT**. It combines
tensor / pipeline / expert / sequence parallelism, which makes 30B-MoE
full-parameter training at 128K context practical on a multi-node GPU cluster.

## Prerequisites

- Python ≥ 3.10 (3.12 recommended)
- CUDA ≥ 12.4 (12.8 recommended)
- PyTorch ≥ 2.6 (2.8 recommended)

`setup_env.sh` installs the rest: `ms-swift`, `megatron-core`, `mcore-bridge`
(loads HF weights online, no manual mcore conversion), `transformer-engine`,
`flash-attn`, and optionally `apex`.

## Smoke test (single GPU)

A quick end-to-end check of the environment, data path, and `megatron sft`
launch chain — **not** a real training run. Defaults to Qwen2.5-0.5B-Instruct on
1 GPU with the bundled `debug_data/sft_debug.jsonl`.

```bash
bash setup_env.sh
bash train_megatron.sh

# multi-GPU variant:
NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES=0,1 bash train_megatron.sh
```

> 30B-MoE full-parameter at 128K context is **not** single-node feasible (even on
> 8×H200). Use one of the multi-node paths below.

## SearchSwarm-SFT dataset preparation

[SearchSwarm-SFT](https://huggingface.co/datasets/SearchSwarm/SearchSwarm-SFT)
ships a single `train.parquet` with one **bundle** per row — a main-agent
conversation plus the sub-agent conversations it dispatched:

```json
{
  "source":        "redsearcher | openseeker",
  "question":      "<main task question>",
  "answer":        "<ground-truth answer>",
  "messages":      [{"role": "system|user|assistant", "content": "..."}],
  "subagents":     [{"question": "<sub-agent briefing>", "messages": ["..."]}],
  "num_subagents": 3
}
```

A bundle holds `1 + num_subagents` independent conversations, so it is not
directly trainable. `convert_share_to_cached.py` unrolls every bundle into flat
ms-swift records — `{"messages": [...]}`, one line per main trajectory and one
per sub-agent trajectory. Conversations are stored already normalized (system
prompt folded into a leading system message; roles limited to
`system`/`user`/`assistant`; every trajectory ends on an assistant message), so
the converter only splits — it never rewrites content.

```bash
hf download SearchSwarm/SearchSwarm-SFT --repo-type dataset --local-dir SearchSwarm-SFT

python convert_share_to_cached.py \
    --parquet SearchSwarm-SFT/train.parquet \
    --out data.jsonl
```

> [!IMPORTANT]
> Stream this parquet — never whole-file read it. It is a single ~2.1 GB row
> group whose nested sub-agent content column decompresses to ~5.8 GB, past
> Arrow's 2 GB per-chunk string limit, so `pandas.read_parquet`,
> `pyarrow.parquet.read_table`, and a plain `datasets.load_dataset` fail with
> `ArrowNotImplementedError: Nested data conversions not supported for chunked
> array outputs` (or exhaust memory), and the Hub dataset viewer cannot preview
> the `messages` / `subagents` columns. The converter streams with
> `ParquetFile.iter_batches`, which keeps peak memory at a few hundred MB. The
> same pattern works for any custom reader:
>
> ```python
> import pyarrow.parquet as pq
>
> pf = pq.ParquetFile("train.parquet")
> for batch in pf.iter_batches(batch_size=32):
>     for row in batch.to_pylist():
>         row["messages"], row["subagents"]  # full nested data, decoded incrementally
> ```

## Multi-node training

The 30B config defaults to 8 nodes × 8 GPU = 64 GPUs at 128K context. All three
paths run the same `megatron sft`; choose by your environment:

- **Ray** — nodes can't SSH each other (typical cloud); you bring up a Ray cluster.
- **SSH** — nodes have passwordless SSH between them.
- **Shared filesystem** — a scheduler launches N identical containers that share a
  filesystem but give no rank and no SSH (Kubernetes Job, cloud batch, ...).

All three consume a **pre-tokenized `--cached_dataset`** so every node reads
identical tokenized data (avoids per-node preprocessing drift). Build one from
the `data.jsonl` produced above (or let the converter chain this step via
`--cached-dataset-dir`), then point `DATA_PATH` at the resulting `train/`
subdirectory:

```bash
swift export \
    --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --dataset data.jsonl \
    --max_length 131072 \
    --to_cached_dataset true \
    --output_dir /path/to/cached_dataset \
    --dataset_num_proc 64
# -> use DATA_PATH=/path/to/cached_dataset/train
```

Measured with the Qwen3 tokenizer, SearchSwarm-SFT trajectories top out around
118K tokens, so `--max_length 131072` keeps them intact. Set `USE_HF=1` if the
tokenizer should come from the HF Hub rather than ModelScope.

### Ray (no inter-node SSH; recommended for cloud)

Megatron has no built-in Ray launcher, so `ray_launch_megatron.py` uses Ray as a
"passwordless-SSH" dispatcher: it sends `megatron sft` to every GPU node, pinned
by NodeAffinity so rank 0 lands on the head.

```bash
# On every node:
bash setup_env.sh
# (make the cached_dataset reachable at the same path on every node)

# Bring up the cluster — head node:
bash start_ray.sh head
# ...and each worker node:
bash start_ray.sh worker <HEAD_IP>

# On the HEAD NODE ONLY — this launches training across the whole cluster.
# Edit MODEL_PATH / DATA_PATH at the top of train_megatron_ray.sh first.
bash train_megatron_ray.sh

# When done, on every node:
bash stop_ray.sh
```

`start_ray.sh` auto-detects the NCCL interface and writes `.nccl_env`, which the
launcher injects into every node's task.

### SSH (traditional; nodes have passwordless SSH)

This path uses torchrun, which is a **per-node** launcher — you start it on every
node with that node's rank.

```bash
# 1. Edit nodes.conf: node IPs, SSH user, DATA_PATH (the cached_dataset), MODEL_PATH.

# 2. HEAD NODE ONLY — sets up passwordless SSH, detects the network, writes env.sh:
bash configure.sh

# 3. On every node — install dependencies:
bash setup_env.sh

# 4. Make env.sh, these scripts, and the cached_dataset available on every node
#    (shared filesystem, or copy env.sh + rsync the data).

# 5. On every node, with that node's rank (0 .. N-1):
bash train_megatron_multinode.sh <NODE_RANK>
```

`configure.sh` runs on the head node only; `train_megatron_multinode.sh` reads
`MASTER_ADDR` and the NCCL settings from the generated `env.sh`.

### Shared filesystem (no SSH, no Ray)

For schedulers that launch N identical containers sharing a filesystem but give
each node no rank and no SSH (Kubernetes Job, cloud batch, SLURM with a shared
FS). Run the **same** command on every node at the same time — the nodes
self-organize through a rendezvous directory on the shared FS: each claims a
unique rank (atomic `mkdir` lock), rank 0 publishes the master address, and all
wait at a barrier before launching.

```bash
# On every node — identical command; same NNODES and RDZV_DIR on each:
NNODES=8 \
RDZV_DIR=/shared/rdzv/run-001 \
MODEL_PATH=Qwen/Qwen3-30B-A3B-Thinking-2507 \
DATA_PATH=/shared/cached_dataset/train \
OUTPUT_DIR=/shared/megatron_output/searchswarm-sft \
bash train_megatron_shared_fs.sh
```

`RDZV_DIR` must be on the shared filesystem, the same path on every node, and
fresh per launch (e.g. include a job id) so stale rank locks don't linger. If a
node picks the wrong network interface, set `NODE_IP` and/or `NCCL_SOCKET_IFNAME`.

## Converting mcore weights back to HF

`--save_safetensors true` is on by default, so checkpoints are already HF-format
safetensors. If you ever disable it:

```bash
swift export \
    --mcore_model ./megatron_output/.../checkpoint-xxx \
    --to_hf true --torch_dtype bfloat16 \
    --output_dir ./megatron_output/.../checkpoint-xxx-hf
```

## Files

| File | Purpose |
|---|---|
| `convert_share_to_cached.py` | Unroll SearchSwarm-SFT bundles (`train.parquet`) into flat ms-swift `data.jsonl`; `--cached-dataset-dir` chains `swift export`. |
| `train_megatron.sh` | Single-GPU smoke test (Qwen2.5-0.5B + `debug_data/`). |
| `train_megatron_multinode.sh <RANK>` | SSH/torchrun path; run on every node. Reads `env.sh`. |
| `train_megatron_ray.sh` | Ray path; run once on the head node. |
| `train_megatron_shared_fs.sh` | Shared-FS path; run the same command on every node (self-elects rank/master via `RDZV_DIR`, no SSH/Ray). |
| `ray_launch_megatron.py` | Ray launcher: dispatches `megatron sft` to all GPU nodes. |
| `start_ray.sh` / `stop_ray.sh` | Ray cluster up / down (run on each node). |
| `configure.sh` | Head-node setup: passwordless SSH, network detection, writes `env.sh`. |
| `nodes.conf` | SSH-path cluster config (IPs, SSH user, data/model paths, parallelism). |
| `setup_env.sh` | Install Megatron-SWIFT and dependencies (run on each node). |

## Parallelism defaults

`train_megatron_multinode.sh` and `train_megatron_ray.sh` default to TP=4, PP=2,
EP=4, CP=2 per 8-GPU node. Keep the high-communication dimensions (TP, EP) within
a node (NVLink); PP and DP can span nodes. Override via `nodes.conf` or a command
prefix: `TP=4 PP=2 EP=4 bash train_megatron_*.sh`. On OOM, raise PP or CP, or set
`--recompute_granularity full`.
