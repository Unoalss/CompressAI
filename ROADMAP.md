# 🗺️ VQ2 & LEGION AI Roadmap

This roadmap outlines the milestones, planned features, and research directions for the **VQ2** compression pipeline and the **LEGION** streaming engine.

---

## 📍 Phase 1: Core Foundation & Proof of Concept (Completed ✅)

- [x] **Randomized Hadamard Transform (RHT)**: Incoherence rotation neutralizing activation outliers in transformer layers.
- [x] **Hessian LDLQ Compensation**: Inverse Cholesky 2nd-order curvature compensation for 8-D vector quantization.
- [x] **Rate-Distortion Optimization (RDO)**: Layer-wise sensitivity sweep and non-uniform bit allocation (2.0 vs 3.0 bpw).
- [x] **Continuous Knowledge Distillation (KD)**: PyTorch differentiable student tuning of scales, codebooks, and RMSNorms.
- [x] **ANS Entropy Packaging**: Sub-byte entropy packaging producing `.vq2` and `.vq2r` containers.
- [x] **Qwen3.5-2B Milestone**: 567 MB container beating GGUF Q4_K_M by 53.5% with preserved reasoning.
- [x] **MoE Paged Streaming Engine (`.vq2m`)**: Sub-1.25GB VRAM execution of Qwen3.8-35B-A3B (10,240 experts) with LRU caching.
- [x] **Milestone B2 Production Candidate (`qwen35b_mixed_b2.vq2m`)**: Selective promotion of 16 critical anchor experts (L31–L39), dropping KL divergence by 75.4% (to 0.0176) with 0.9974 cosine similarity.
- [x] **Fused Triton 2D-Grid GEMV**: Parallel expert dispatch executing 8 active experts in parallel.
- [x] **Compatible Server APIs**: Built-in HTTP server supporting OpenAI, Ollama, and llama.cpp endpoints.

---

## 🚀 Phase 2: Ecosystem & Cross-Platform Acceleration (In Progress 🔄)

- [ ] **`llama.cpp` Integration**:
  - [ ] Implement `GGML_TYPE_VQ8_16` / `GGML_TYPE_VQ8_16_8` tensor types in GGML.
  - [ ] Support loading `.vq2` and `.vq2m` directly inside `llama.cpp` CLI and web UI.
  - [ ] CPU SIMD vector lookup with AVX-512, AVX2, and ARM NEON.
- [ ] **Apple Silicon (MPS / MLX)**:
  - [ ] Native Metal Shading Language (MSL) decompression kernel.
  - [ ] Apple MLX integration for unified memory execution on M1/M2/M3/M4 chips.
- [ ] **Kernel Autotuning & Architecture Portability**:
  - [ ] Triton autotuning configs for NVIDIA Ampere, Ada Lovelace, Hopper, and Blackwell.
  - [ ] AMD ROCm / HIP compatibility verification.
- [ ] **Developer Tooling & Packaging**:
  - [ ] GitHub Actions CI workflow (linting, tests, wheel builds).
  - [ ] PyPI package publication (`pip install vq2`).
  - [ ] Hugging Face model hub downloader and exporter.

---

## 🔮 Phase 3: Extreme Scale MoE & Distributed Execution (Planned 📋)

- [ ] **Massive MoE Support (100B–300B Parameters)**:
  - [ ] Adapt VQ2M pipeline to **DeepSeek-V3 / DeepSeek-V4-Flash** (284B parameters, 256 experts).
  - [ ] Target: Run 284B MoE on a single 16GB–24GB consumer GPU with streaming from NVMe SSD.
- [ ] **Speculative Vector Decoding**:
  - [ ] Integrate n-gram prompt lookahead and draft head speculative verification.
  - [ ] Target: Boost streaming generation throughput from ~9.5 tok/s to >20 tok/s.
- [ ] **Dynamic Expert Prefetching**:
  - [ ] Machine-learning predictor for next-layer expert routing.
  - [ ] Prefetch anticipated cold experts into VRAM asynchronously over PCIe.
- [ ] **Edge & Mobile Deployment**:
  - [ ] Android / ONNX Runtime execution via NPU / OpenCL.
  - [ ] Sub-1-watt standby mode for edge embedded devices.

---

## 💡 How to Propose a Feature or RFC

Have an idea for VQ2 or LEGION?
1. Open a GitHub Issue using the `[RFC]` prefix.
2. Outline the problem, mathematical or engineering approach, and expected memory/speed trade-offs.
3. Join the discussion in the PRs and Issues tab!
