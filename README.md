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
