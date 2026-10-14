# Contributing to VQ2 & LEGION AI

Thank you for your interest in contributing to **VQ2 & LEGION AI**!  
We are building the future of open-source extreme vector quantization and memory-virtualized LLM execution. We welcome contributions from ML engineers, systems programmers, CUDA/Triton developers, and researchers.

---

## 🎯 Priority Areas (Where We Need Your Help)

If you're wondering where to start, here are our highest-priority tasks (see detailed spec in [`docs/GUIA_PESQUISA_CPU_FIRST.md`](docs/GUIA_PESQUISA_CPU_FIRST.md)):

### 1. 🔥 CPU-01: Projeções Densas sem Materialização Permanente na RAM
* **Goal**: Eliminate the permanent RAM reconstruction of the **328 dense projection matrices** in the CPU runtime. Implement `quant_linear` for CPU directly over the compressed indices with small per-thread transient buffers.
* **Why it matters**: This is the primary bottleneck preventing the 35B model from executing within a 4–8 GB RAM budget on basic PCs and mobile phones.
* **Skills needed**: Python, PyTorch C++ extension, low-memory runtime design.

### 2. ⚡ CPU-02: Kernels SIMD Nativos (AVX2 & ARM NEON)
* **Goal**: Write vector-quantized GEMV kernels in C++ using AVX2 (x86_64) and ARM NEON (ARM64) with a scalar reference implementation for strict bit-exact verification.
* **Why it matters**: Replaces slow Python loops with high-throughput native vector operations on PCs and Android/Snapdragon devices.
* **Skills needed**: C++, SIMD (AVX2, NEON), numerical computing.

### 3. ⚙️ `llama.cpp` / GGML Native VQ Integration
* **Goal**: Implement a custom GGML tensor type (e.g. `GGML_TYPE_VQ8_16`) supporting 8-D vector decompression.
* **Why it matters**: Allows `.vq2` and `.vq2m` models to run out-of-the-box on CPU, macOS Metal (MPS), and Android/iOS via `llama.cpp`.
* **Skills needed**: C++, SIMD (AVX-512 / ARM NEON), GGML internals.

### 2. ⚡ Triton Kernel Performance Tuning
* **Goal**: Autotune and optimize the `_vq_gemv_16_8` and `_parallel_moe_gemv` kernels in [`vq2/kernels.py`](file:///e:/compressAI/vq2/kernels.py).
* **Specific items**:
  - Implement warp-level reductions for non-power-of-2 matrix shapes.
  - Test memory coalescing across Ampere (SM86), Ada Lovelace (SM89), and Hopper (SM90).
  - Add native Triton prefetching for MoE expert weights.

### 3. 🍏 Apple Silicon (Metal / MLX) Backend
* **Goal**: Create a native Apple Silicon runtime using Apple MLX or custom Metal Shading Language (MSL) kernels.
* **Why it matters**: Enables running 35B and 70B MoE models locally on Mac Studio, MacBook Pro, and Mac Mini with unified memory.

### 4. 🧠 Architecture Extensions
* **Goal**: Expand VQ2 quantization scripts to support other popular model families:
  - DeepSeek-V3 / DeepSeek-V4-Flash (MoE with fine-grained routing).
  - Llama-3.1 / 3.3 (8B, 70B).
  - Mistral / Mixtral (8x7B, 8x22B).
  - Gemma-2 / Gemma-3.

### 5. 📦 PyPI Packaging & CI/CD
* **Goal**: Publish `vq2` on PyPI so users can simply run `pip install vq2`.
* **Add GitHub Actions workflows**:
  - Linting (Ruff / Black).
  - Unit tests with CPU fallback (`pytest`).
  - Safe tensor validation.

---

## 🛠️ Development Setup

1. **Fork and clone**:
   ```bash
   git clone https://github.com/Unoalss/CompressAI.git
   cd compressAI
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux / macOS
   # or on Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   pip install pytest black isort ruff
   ```

4. **Verify your setup**:
   ```bash
   python -m vq2.smoke
   ```

---

## 🧪 Testing Guidelines

Before opening a pull request:
1. Ensure all new modules have appropriate unit tests.
2. For numerical kernels (Triton/CUDA/C++), ensure equivalence against the PyTorch unquantized reference:
   ```bash
   python scratch/test_numerical_equivalence.py
   ```
3. Test with both CPU fallback (`HAVE_TRITON = False`) and GPU mode (`HAVE_TRITON = True`).

---

## 📝 Commit & PR Conventions

* Use descriptive commit messages following Conventional Commits:
  - `feat: add metal MPS kernel for 8-D vector decompression`
  - `fix: resolve race condition in LRU expert cache eviction`
  - `perf: optimize Triton 2D-Grid GEMV memory coalescing`
  - `docs: update quickstart guide with Linux pip instructions`
* Keep PRs focused on a single logical change.
* Include benchmark or profiling numbers if submitting a performance improvement.

---

## 💬 Community & Discussion

* Open an Issue to discuss architectural changes or propose new features.
* Check [ROADMAP.md](ROADMAP.md) and [ARCHITECTURE.md](ARCHITECTURE.md) to understand current design decisions.
