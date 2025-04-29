# 🚀 CUDA Learning Journey & Roadmap

Welcome to my GPU learning journey!  
This repository documents my path from CUDA fundamentals to optimizing AI kernels for production-level performance.

---

## 📘 Primary Learning Resource

**Book:**  
> *Programming Massively Parallel Processors: A Hands-on Approach*  
> Authors: David B. Kirk, Wen-mei W. Hwu

---

## 🧭 Learning Phases

### 🟢 Phase 1: Fundamentals
> Goal: Learn the CUDA execution model and write basic kernels.

- [ ] Set up CUDA dev environment
- [ ] Learn thread/block/grid hierarchy
- [ ] Write first CUDA program (vector addition)
- [ ] Experiment with block sizes
- [ ] Understand memory hierarchy (global, shared, registers)
- [ ] Build CMake-based template

### 🔵 Phase 2: Intermediate Kernels
> Goal: Write more complex and optimized kernels using shared memory.

- [ ] Matrix addition and multiplication
- [ ] Use of shared memory (tiling)
- [ ] Learn about bank conflicts and coalesced access
- [ ] Reduction (sum) using tree method
- [ ] Warp-level intrinsics (shuffle)
- [ ] Prefix sum (scan)

### 🟣 Phase 3: Performance Profiling
> Goal: Profile and optimize your kernels.

- [ ] Learn to use Nsight Compute
- [ ] Interpret key metrics: occupancy, memory throughput, warp efficiency
- [ ] Optimize memory usage (shared, register, global)
- [ ] Time kernels using `cudaEvent_t`
- [ ] Measure speedups vs CPU baseline

### 🔶 Phase 4: Advanced AI Ops
> Goal: Build and optimize real AI operations in CUDA.

- [ ] Softmax, LayerNorm, GELU kernels
- [ ] Fuse multiple ops into one kernel
- [ ] Mixed precision (`float16` / Tensor Cores optional)
- [ ] Workload balancing (row-wise, warp-wise)
- [ ] Use of streams for overlapping compute and memcpy

### 🟥 Phase 5: Systems Integration
> Goal: Build pipelines that combine multiple CUDA ops.

- [ ] Use `cudaStream_t` for concurrency
- [ ] Profile full app with Nsight Systems
- [ ] Measure end-to-end latency
- [ ] Build benchmarking framework
- [ ] Prepare real-time AI op use cases (e.g. inference block)

---

## 🧪 Mini Projects

Chapters 1–4: Basic CUDA Programming

Concepts:
    - Thread hierarchy
    - Kernel launching
    - Device vs Host memory

Mini-Projects:

    Vector Addition: Naive version (each thread adds one element)

    2D Matrix Addition: Threads organized in 2D grids

    Vector Scaling: Multiply vector elements by a scalar

Chapter 5: Memory and Data Locality

Concepts:

    Global vs Shared memory

    Memory coalescing

Mini-Projects:

    Shared Memory Matrix Addition: Rewrite previous matrix addition using shared memory tiles.

    Global Memory Coalescing Checker: Write a program that intentionally breaks coalescing, and then fix it to measure performance difference.

Chapter 6: Parallel Patterns - Reduction

Concepts:

    Reduction (sum, max)

Mini-Projects:

    Sum Reduction Kernel:
    Write a kernel to compute the sum of an array.

    Max Reduction Kernel:
    Find the maximum element in a large array using warp shuffles if you want to go deeper.

Chapter 7: Parallel Scan (Prefix Sum)

Concepts:

    Exclusive and Inclusive scan

    Work-efficient vs naive scan

Mini-Projects:

    Prefix Sum Kernel (Naive):
    Parallel inclusive scan using basic threads.

    Prefix Sum Kernel (Optimized):
    Use work-efficient scan (Blelloch scan algorithm).

Chapter 8–9: Tiling and Memory Optimization

Concepts:

    Tiling for matrix multiplication

    Shared memory optimization

Mini-Projects:

    Naive Matrix Multiplication (GEMM):
    No shared memory, just indexing.

    Tiled GEMM (Shared Memory):
    Use tiling with shared memory for better performance.

    Compare naive vs tiled performance using Nsight Compute.

Chapter 10: Additional Patterns

Concepts:

    Histograms

    Sparse Matrix Operations

Mini-Projects:

    Histogram Kernel:
    Compute a histogram of values (pay attention to atomic operations).

    Sparse Matrix-Vector Multiplication (SpMV):
    If you feel adventurous, it's great for memory-access pattern learning.

Chapters 11+: Advanced Topics

Concepts:

    Streams, concurrency

    Occupancy optimization

    Warp-level primitives

Mini-Projects:

    Multi-Stream Vector Addition:
    Overlap memory copies and kernel execution using streams.

    Warp-Reduction Kernel:
    Do sum reduction within a warp using warp shuffle instructions.

Bonus: AI-specific Mini-Projects

Once you finish these and want to start blending into AI ops:

    Softmax Kernel: Implement Softmax for a batch of vectors.

    LayerNorm Kernel: Implement simple LayerNorm without framework help.

These skills map directly to real AI framework kernels (like PyTorch, TensorFlow).

| # | Project | Status | Notes |
|--|---------|--------|-------|
| 1 | Vector Add (1D) | ☐ | First working kernel |
| 2 | Vector Add (2D grid) | ☐ | Practice grid indexing |
| 3 | Matrix Add | ☐ | Memory layout practice |
| 4 | Matrix Multiply (naive) | ☐ | Row-col inner product |
| 5 | Matrix Multiply (tiled) | ☐ | Shared memory optimization |
| 6 | Sum Reduction | ☐ | Tree + warp shuffle |
| 7 | Prefix Sum (Scan) | ☐ | Used in stream compaction |
| 8 | Histogram | ☐ | Atomics and privatization |
| 9 | LayerNorm | ☐ | AI kernel |
| 10 | Multi-stream Overlap | ☐ | Use `cudaStream_t` |

---

## 🏆 CUDA Challenge List

| Challenge | Description | Done |
|----------|-------------|------|
| 1 | 1D/2D Vector Add + Grid indexing | ☐ |
| 2 | Try multiple block sizes | ☐ |
| 3 | Add error handling macro | ☐ |
| 4 | Rewrite Matrix Add with shared mem | ☐ |
| 5 | Memory coalescing vs strided | ☐ |
| 6 | Use `cudaOccupancyMaxPotentialBlockSize` | ☐ |
| 7 | Tree Reduction | ☐ |
| 8 | Warp Shuffle Reduction | ☐ |
| 9 | Prefix Scan | ☐ |
| 10 | Histogram with privatization | ☐ |
| 11 | Tiled Matrix Multiply | ☐ |
| 12 | Softmax kernel | ☐ |
| 13 | Fused LayerNorm | ☐ |
| 14 | Stream Overlap | ☐ |
| 15 | Benchmark CUDA vs CPU | ☐ |

---

## 📦 Profiling Tools

### Nsight Compute (ncu)
> For analyzing individual CUDA kernels

- Metrics: occupancy, memory throughput, warp efficiency
- Command: `ncu ./your_binary`

### Nsight Systems (nsys)
> For analyzing full program timeline

- Measure: CPU-GPU overlaps, kernel launch delays, memcpy overlaps
- Command: `nsys profile ./your_binary`

---

## 📅 Weekly Progress Log

| Week | Highlights |
|------|------------|
| Week 1 | Set up project, completed 1D vector add |
| Week 2 | _[to be filled]_ |
| Week 3 | _[to be filled]_ |

---
