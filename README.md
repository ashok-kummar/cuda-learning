# 🚀 CUDA Learning Plan & Progress Tracker

Welcome to my GPU learning journey! This README documents my roadmap, challenges, and progress as I learn CUDA for AI optimization and parallel programming.

---

## 📘 Learning Resource

**Main Book:**  
> *Programming Massively Parallel Processors: A Hands-on Approach*  
> Authors: David B. Kirk, Wen-mei W. Hwu

---

## 🧭 Roadmap

### ✅ Phase 1: Fundamentals
- [ ] Set up CUDA development environment
- [ ] Learn CUDA execution model (threads, blocks, grids)
- [ ] Implement vector addition in CUDA
- [ ] Explore different `threadsPerBlock` sizes
- [ ] Learn memory hierarchy (global, shared, registers)
- [ ] Write clean project structure with CMake

### 🔄 Weekly Progress Updates
I'll update this section weekly with notes and completed challenges.

---

## 🧪 Mini Projects

| # | Project | Status | Notes |
|--|---------|--------|-------|
| 1 | Vector Add (1D) | ☐ | First working CUDA kernel |
| 2 | Vector Add (2D grid) | ☐ | Practice grid indexing |
| 3 | Matrix Add | ☐ | Practice memory layout |
| 4 | Matrix Multiplication (naive) | ☐ | |
| 5 | Matrix Multiplication (tiled/shared memory) | ☐ | Optimize for performance |
| 6 | Reduction (sum) | ☐ | Tree-based and warp-shuffle |
| 7 | Prefix Sum (scan) | ☐ | Use for stream compaction |
| 8 | Histogram | ☐ | Practice atomics |
| 9 | LayerNorm | ☐ | Mini AI-op kernel |
| 10 | Multi-stream overlaps | ☐ | Use `cudaStream_t` |

---

## 🏆 CUDA Challenge List

| Challenge | Description | Done |
|----------|-------------|------|
| 1 | 1D & 2D Vector Add + Grid indexing | ☐ |
| 2 | Experiment with different block sizes | ☐ |
| 3 | Error handling macro for all CUDA calls | ☐ |
| 4 | Rewrite matrix add with shared memory | ☐ |
| 5 | Memory coalescing vs strided access | ☐ |
| 6 | Use `cudaOccupancyMaxPotentialBlockSize` | ☐ |
| 7 | Tree-based sum reduction | ☐ |
| 8 | Warp-level shuffle-based reduction | ☐ |
| 9 | Stream compaction with prefix sum | ☐ |
| 10 | Naive matrix multiplication | ☐ |
| 11 | Tiled matrix mult (shared memory) | ☐ |
| 12 | Compare tile sizes: 8x8, 16x16, 32x32 | ☐ |
| 13 | Histogram optimization (privatization) | ☐ |
| 14 | Overlapping memcopy + compute (streams) | ☐ |
| 15 | Softmax kernel | ☐ |
| 16 | Fused LayerNorm (mean/var/scale) | ☐ |
| 17 | Memory optimization of AI op | ☐ |

---

## 🔧 Tools & Profiling

### 🔹 Profiler

- **Start with:** `Nsight Compute (ncu)`
- **Later:** `Nsight Systems (nsys)` for full-app timeline

> Nsight Compute is ideal when optimizing a single kernel  
> Nsight Systems helps when analyzing full workloads, memory copy overlaps, and multiple streams.

### 🔹 Planned Guides

- [ ] Nsight Compute Starter Guide ✅ (queued)
- [ ] Advanced Template with Timing, Testing, CMake, Profiling (coming soon)

---

## 🔁 Next Steps

- [ ] Complete 2–3 mini-projects
- [ ] Switch to advanced CUDA project template
- [ ] Start performance profiling using Nsight Compute
- [ ] Begin automated unit testing of kernel outputs

---

## 🥇 Badge Milestones

| Badge | Requirement | Earned |
|-------|-------------|--------|
| CUDA Beginner | Complete 5 beginner challenges | ☐ |
| Memory Master | Optimize shared/coalesced memory kernels | ☐ |
| Warp Wizard | Use warp intrinsics (shuffle, ballots) | ☐ |
| Performance Profiler | Profile and improve kernel runtime | ☐ |
| AI Kernel Pro | Implement and optimize AI-op kernel | ☐ |

---

## 📅 Weekly Progress Log

| Week | Highlights |
|------|------------|
| Week 1 | Started vector add, set up CMake, began Challenge 1 |
| Week 2 | _[to be filled]_ |
| Week 3 | _[to be filled]_ |

---

## 💬 Notes & Learnings

> “Optimization without measurement is just guesswork.”  
> Profiling is as important as coding — always measure performance and verify correctness!

---

## 🙌 Credits

This roadmap was designed with guidance from an AI mentor powered by ChatGPT.  
Structured like a real-world GPU engineering journey 🚀

---

