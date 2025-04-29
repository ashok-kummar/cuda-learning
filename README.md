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

## 📘 CUDA Mini-Projects Timeline

This section breaks down the CUDA mini-projects aligned with chapters from the book  
**"Programming Massively Parallel Processors: A Hands-on Approach"**  
Each mini-project builds core GPU programming skills for AI kernel optimization.

---

### 📗 Chapters 1–4: Basic CUDA Programming

**Concepts:**
- Thread hierarchy (thread, block, grid)
- Kernel launching
- Host vs Device memory

**Mini-Projects:**
- ✅ **Vector Addition** – Naive version (each thread adds one element)
- ✅ **2D Matrix Addition** – Threads organized in 2D grid
- ✅ **Vector Scaling** – Multiply vector elements by a scalar

---

### 📘 Chapter 5: Memory and Data Locality

**Concepts:**
- Global vs Shared memory
- Memory coalescing

**Mini-Projects:**
- ✅ **Shared Memory Matrix Add** – Rewrite matrix add using shared memory tiling
- ✅ **Global Memory Coalescing Checker** – Write one version that breaks coalescing and compare with an optimized one using Nsight Compute

---

### 📙 Chapter 6: Parallel Patterns – Reduction

**Concepts:**
- Reduction (sum, max, min)

**Mini-Projects:**
- ✅ **Sum Reduction Kernel** – Parallel reduction using grid-stride loops
- ✅ **Max Reduction Kernel** – Find maximum value using warp shuffles (optional advanced version)

---

### 📒 Chapter 7: Parallel Scan (Prefix Sum)

**Concepts:**
- Inclusive vs Exclusive scan
- Work-efficient vs Naive scan

**Mini-Projects:**
- ✅ **Prefix Sum (Naive)** – Simple parallel scan using shared memory
- ✅ **Prefix Sum (Work-Efficient)** – Use Blelloch scan or similar work-efficient strategy

---

### 📕 Chapters 8–9: Tiling and Memory Optimization

**Concepts:**
- Matrix tiling for GEMM
- Shared memory optimizations

**Mini-Projects:**
- ✅ **Naive GEMM (Matrix Multiply)** – Row-by-col dot-product (no shared memory)
- ✅ **Tiled GEMM with Shared Memory** – Use tiling to load shared blocks and improve locality
- ✅ **Compare Naive vs Tiled performance** – Use Nsight Compute metrics

---

### 📓 Chapter 10: Additional Patterns

**Concepts:**
- Histograms
- Sparse matrix ops

**Mini-Projects:**
- ✅ **Histogram Kernel** – Bin values into histogram (use atomics carefully)
- ✅ **Sparse Matrix-Vector Multiply (SpMV)** – Bonus: Practice memory access efficiency

---

### 📔 Chapters 11+: Advanced Topics

**Concepts:**
- Streams and overlapping kernels
- Occupancy optimization
- Warp-level primitives

**Mini-Projects:**
- ✅ **Multi-Stream Vector Add** – Use streams to overlap compute and memory copies
- ✅ **Warp-Level Reduction** – Use warp shuffles (`__shfl_xor_sync`) for faster reductions

---

## 🧠 Bonus: AI-Specific CUDA Mini-Projects

These mirror real-world kernels used in AI/ML frameworks like PyTorch and TensorFlow.

**Mini-Projects:**
- ✅ **Softmax Kernel** – Normalize logits in-place
- ✅ **LayerNorm Kernel** – Normalize and scale inputs across feature dimensions


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
