# llama2-cu.rs

A Rust port of [llama2.c](https://github.com/karpathy/llama2.c),
with a GPU backend built using
[NVIDIA cuTile Rust](https://github.com/NVIDIA/cuda-rust).

I built this project to study LLM inference and explore CUDA kernel
programming in Rust. It started as a port of llama2.c and grew to
include my own cuTile implementation of the inference operations.

## What's included

- CPU inference in Rust
- cuTile GPU kernels for RMSNorm, RoPE, attention, and feed-forward layers
- CUDA Graph capture and replay
- Tokenization and sampling adapted from llama2.c

## Development

I implemented the port and GPU kernels myself as a learning exercise,
without AI code generation or coding agents.
AI was used only to help find documentation.

## Running

### Prerequisites

- Rust and Cargo
- A CUDA development environment with Clang/libclang
- A cuTile-compatible NVIDIA GPU and driver for GPU inference

The current build includes CUDA dependencies even when running in CPU mode.
See the benchmark environment below for the configuration used here.

### Setup

```bash
git clone https://github.com/SangHyeok-Yoon/llama2-cu.rs.git
cd llama2-cu.rs

wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin

cargo build --release --locked
```

`tokenizer.bin` is included in the repository.
Run the executable from the repository root: the current implementation
loads `stories15M.bin` and `tokenizer.bin` from the working directory.

### CPU inference

```bash
./target/release/llama-cu-rs -m cpu -i "Once upon a time"
```

### GPU inference

```bash
./target/release/llama-cu-rs -m gpu -i "Once upon a time"
```

GPU mode uses CUDA Graph capture and replay, with warm-up before generation.

Omit `-i` to enter an interactive prompt loop:

```bash
./target/release/llama-cu-rs -m gpu
```

CPU is the default mode. Generation uses temperature `1.0`, top-p `0.9`,
and a maximum of 256 forward steps, including prompt processing.
These settings and the model path are currently configured in `src/main.rs`.

## Performance

Measured on November 4, 2026, using a release build of commit `d7db193`.

### Environment

| Component | Configuration |
| --- | --- |
| CPU | Intel Core i7-14700K |
| GPU | NVIDIA GeForce RTX 4090, 24 GB |
| OS | Ubuntu 24.04.4 LTS on WSL2 |
| NVIDIA driver | 610.88 |
| CUDA Toolkit | 13.3 |
| Rust / cuTile | 1.99.1 / 0.4.0 |

### Methodology

For each model and backend, 20 generation runs were executed consecutively
in a single process, keeping the model loaded throughout:

- The first 10 runs were discarded as warm-up.
- The following 10 runs were measured.
- Results are the arithmetic mean of the reported tok/s values.
- GPU clocks were managed automatically, without manual clock locking.

All runs used the prompt `Once upon a time`, temperature `1.0`,
top-p `0.9`, and a maximum of 256 forward steps including prompt processing.

The CPU implementation is single-threaded. The GPU backend uses cuTile
kernels with CUDA Graph capture and replay.

### Results

| Model | CPU mean tok/s | GPU mean tok/s | GPU / CPU |
| --- | ---: | ---: | ---: |
| stories15M | 142.15 | 1,982.35 | 13.95× |
| stories110M | 19.24 | 567.51 | 29.50× |

### Measurement scope

Throughput uses the application's `achieved tok/s` metric. It excludes
model loading, GPU initialization, graph setup, warm-up, and the first
forward step of each run.

The timed region includes subsequent prompt processing, inference,
sampling, decoding, and output writes. GPU execution also includes
copying logits back to the CPU. Output was captured through a pipe.

Sampling uses no fixed seed, so generated sequences and lengths vary.
Generation may stop before reaching the step limit.
