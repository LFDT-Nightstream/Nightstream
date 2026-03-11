# GPU Acceleration Plan for Nightstream via Mojo

## Executive Summary

Nightstream's proving pipeline has clear GPU-parallelizable bottlenecks: **Poseidon2 permutations**, **Ring R_q multiplication** (O(D^2)=2916 ops), **SuperNeo transforms** (54x54 mat-vec), **Ajtai commitments**, and **sumcheck Q-polynomial evaluation**. The key architectural challenge is that the sumcheck protocol is inherently sequential *between* rounds (each round depends on the previous Fiat-Shamir challenge), but massively parallel *within* each round. To avoid CPU-GPU context switching, **entire proving stages must live on-GPU**, including the Poseidon2 transcript.

We use [Mojo](https://www.modular.com/mojo) to implement GPU kernels targeting **CUDA** (NVIDIA), **Metal** (Apple Silicon), and **HIP** (AMD) from a single codebase. The Rust layer remains the orchestration layer (VM execution, CCS construction, IVC session management).

---

## Table of Contents

- [Phase 0: Foundation — Goldilocks Field Kernel](#phase-0-foundation--goldilocks-field-kernel-mojo)
- [Phase 1: Poseidon2 GPU Kernel](#phase-1-poseidon2-gpu-kernel)
- [Phase 2: Ring R_q Arithmetic on GPU](#phase-2-ring-r_q-arithmetic-on-gpu)
- [Phase 3: Ajtai Commitment Kernel](#phase-3-ajtai-commitment-kernel)
- [Phase 4: Sumcheck Prover on GPU](#phase-4-sumcheck-prover-on-gpu-the-big-win)
- [Phase 5: Integration Architecture (Rust <-> Mojo)](#phase-5-integration-architecture-rust--mojo)
- [Phase 6: Implementation Roadmap](#phase-6-implementation-roadmap)
- [Architecture Diagram](#architecture-diagram)
- [Risks & Mitigations](#risks--mitigations)
- [Expected Performance Gains](#expected-performance-gains)

---

## Current Hot Paths Analysis

The following components are CPU-bound bottlenecks identified in the proving pipeline:

| Component | Operation | Complexity | Frequency | GPU Suitable |
|-----------|-----------|-----------|-----------|--------------|
| **Poseidon2** | State permutation | O(rounds) ~ 128 ops | Per 4 field absorptions | YES — parallel sponges |
| **Ring Rq** | Multiplication | O(D^2) = 2916 ops | Per sumcheck round | YES — Karatsuba/NTT |
| **SuperNeo** | Transform | O(D^2) = 2916 ops | Per fold step | YES — matrix-vector |
| **FFT** | Butterfly network | O(n log n) | Polynomial evaluation | YES — standard GPU FFT |
| **Field Arithmetic** | Fq multiply/add | ~1 cycle | Billions of times | YES — SIMD/GPU native |
| **Extension K** | Multiply | O(4) Fq ops | DEC/RLC checks | YES — parallel |

### End-to-End Proving Data Flow

```
Program Execution (RV64IM)
    |
Trace Generation (VmTrace)
    |
Execution Table Construction (RiscvExecTable)
    |
Constraint System (CCS / Trace Wiring)
    |
Shard Folding (Pi_CCS Reduction)          <-- GPU accelerated
    |
Aggregation (Pi_RLC)                      <-- GPU accelerated
    |
Decomposition (Pi_DEC)                    <-- GPU accelerated
    |
IVC Embedded Verifier (public-rho)
    |
Final SNARK (Spartan/SuperSpartan + FRI)
```

---

## Phase 0: Foundation — Goldilocks Field Kernel (Mojo)

**Goal:** Implement the entire Goldilocks field arithmetic as a Mojo GPU kernel library.

### Primitives to implement

| Primitive | Details | GPU Strategy |
|-----------|---------|-------------|
| `Fq` addition | `(a + b) mod p`, p = 2^64 - 2^32 + 1 | Per-thread, trivial |
| `Fq` subtraction | `(a - b) mod p` | Per-thread, trivial |
| `Fq` multiplication | `a * b mod p` using Barrett reduction | Per-thread, u64 x u64 -> u128 reduction |
| `Fq` inverse | Fermat's little theorem: `a^(p-2) mod p` | Per-thread, exponentiation chain |
| `Fq` batch inverse | Montgomery's trick (1 inverse + n muls) | Warp-cooperative |
| `K = Fq^2` ops | Extension field `(a+bu)` where `u^2=7` | 2-wide per thread |
| `K` conjugation | `(a+bu) -> (a-bu)` | Trivial |
| `K` inverse | Via norm: `1/(a+bu) = conj/(a^2-7b^2)` | Per-thread |

### Why Barrett over Montgomery for Goldilocks on GPU

- p = 2^64 - 2^32 + 1 has special structure allowing fast reduction via shifts
- No need for Montgomery domain conversion overhead
- Reduction: `r = x - (x >> 64) * p`, then conditional subtract

### Reference implementation

The Rust reference for field operations lives in:
- `crates/neo-math/src/field.rs` — Fq and K = Fq^2
- `crates/neo-math/src/ring.rs` — Ring R_q arithmetic
- `crates/neo-math/src/s_action.rs` — S-action implementation

### Proposed file structure

```
nightstream-gpu/
    mojo/
        field/
            goldilocks.mojo      # Fq arithmetic kernels
            extension.mojo       # K = Fq^2 arithmetic
            batch_ops.mojo       # Vectorized batch operations
        tests/
            test_field.mojo      # GPU field correctness tests
        mojoproject.toml
```

---

## Phase 1: Poseidon2 GPU Kernel

**Goal:** Batch Poseidon2 permutations on GPU — this is the most impactful single optimization.

### Current bottleneck

- Poseidon2 permutation runs once per RATE=4 field elements absorbed
- WIDTH=8 state, ~22 rounds (8 full + 14 partial for Goldilocks)
- Called thousands of times during a single shard proof (transcript + commitment hashing)
- Configuration lives in `crates/neo-params/src/poseidon2_goldilocks.rs`
- Transcript implementation: `crates/neo-transcript/src/poseidon2.rs`

### GPU kernel design

```
+---------------------------------------------+
|  Batch Poseidon2 Kernel                     |
|                                             |
|  Thread block: 1 block per sponge instance  |
|  Threads: 8 per block (1 per state element) |
|                                             |
|  Shared memory: state[8] per block          |
|                                             |
|  Flow:                                      |
|  1. Load state from global -> shared        |
|  2. For each round:                         |
|     a. S-box (x^7) -- parallel per element  |
|     b. MDS matrix multiply -- shared mem    |
|     c. Round constant add -- parallel       |
|  3. Store state from shared -> global       |
|                                             |
|  Batch: 256-4096 independent sponges        |
+---------------------------------------------+
```

### Poseidon2-specific optimizations

- **External rounds** (full S-box): All 8 elements get x^7 — fully parallel
- **Internal rounds** (partial S-box): Only element 0 gets x^7, then diffusion — still parallel for the linear layer
- **MDS matrix**: Poseidon2 uses a structured M4 + diagonal matrix that decomposes into cheap operations

### Critical design decision

The Poseidon2 transcript (`Poseidon2Transcript`) must also run on GPU to avoid round-trip transfers during sumcheck. This means **the entire transcript state lives on GPU memory**.

### Reference files

- `crates/neo-params/src/poseidon2_goldilocks.rs` — Parameters (WIDTH, RATE, CAPACITY, DIGEST_LEN)
- `crates/neo-transcript/src/poseidon2.rs` — Transcript struct and absorb/squeeze
- `crates/neo-ccs/src/crypto/poseidon2_goldilocks.rs` — Hash functions and permutation cache
- `crates/neo-fold/src/memory_sidecar/memory/precompiles/poseidon2/` — Poseidon2 precompile traces

---

## Phase 2: Ring R_q Arithmetic on GPU

**Goal:** Accelerate cyclotomic ring multiplication from O(D^2) schoolbook to GPU-parallel.

### Current state

- Ring R_q: Polynomials mod Phi_81(X) = X^54 + X^27 + 1, coefficients in Fq
- Schoolbook multiply: 54 x 54 = 2916 Fq multiplications + reduction
- Called heavily in: folding, RLC, DEC, Ajtai commitment
- Implementation: `crates/neo-math/src/ring.rs`

### GPU strategies

**Option A — Batched Schoolbook (recommended for D=54):**
- Each thread block handles one ring multiplication
- 54 threads per block, each computes one output coefficient
- Shared memory for both input polynomials
- Reduction mod Phi_81 integrated into accumulation

**Option B — NTT-based convolution (future optimization):**
- Transform to evaluation domain via NTT (need >= 108 points)
- Point-wise multiply (trivially parallel)
- Inverse NTT back + reduce mod Phi_81
- Requires Goldilocks-compatible NTT of size >= 108 (Goldilocks supports 2-adic NTT up to 2^32)

**Recommendation:** Start with Option A — D=54 is small enough that NTT overhead likely doesn't pay off. Revisit if profiling shows ring mul as top bottleneck.

### SuperNeo transform

The `superneo_bar_block()` is a 54x54 dense matrix-vector product — textbook GPU operation:
- Preload matrix into shared/constant memory (once, it's static)
- Each thread computes one output element (54 threads)
- Batch across many blocks
- Implementation: `crates/neo-math/src/ring.rs` (`superneo_bar_matrix()`, `superneo_bar_block()`, `superneo_bar_vec()`)

---

## Phase 3: Ajtai Commitment Kernel

**Goal:** GPU-accelerate the lattice commitment `c = A * z mod q` where A is a large matrix over R_q.

### Current state

- Ajtai matrix A has dimensions kappa x m where kappa=16, m=2^24
- Each entry is a ring element in R_q (54 Fq coefficients)
- Witness z is decomposed into base-b digits
- Commitment: matrix-vector product over the ring
- Implementation: `crates/neo-ajtai/src/commit.rs`

### GPU architecture

```
+----------------------------------------------+
|  Ajtai Commitment Kernel                     |
|                                              |
|  Phase 1: Digit decomposition (parallel)     |
|  - Each thread decomposes one witness column |
|  - Output: digit matrix in GPU memory        |
|                                              |
|  Phase 2: Matrix-vector product              |
|  - Block-row parallel: each block computes   |
|    one output ring element                   |
|  - Within block: parallel ring multiply-add  |
|  - Reduction tree for accumulation           |
|                                              |
|  ALL in GPU memory -- no CPU round-trip      |
+----------------------------------------------+
```

### Memory considerations

- A matrix is large (kappa x m x 54 x 8 bytes)
- For m=2^24: ~16M ring elements x 54 x 8 = ~6.9 GB
- Strategy: **stream A matrix** through GPU in tiles, accumulate partial sums
- Witness z stays fully resident in GPU memory

---

## Phase 4: Sumcheck Prover on GPU (The Big Win)

**Goal:** Run the entire sumcheck protocol on GPU, including transcript updates, to eliminate per-round CPU-GPU transfers.

### Why this is critical

The sumcheck has `l` rounds (l = log2(trace_size), typically 16-20). Each round:

1. Evaluate Q polynomial over half the hypercube — **massively parallel**
2. Absorb evaluations into Poseidon2 transcript — **sequential but fast**
3. Squeeze challenge from transcript — **sequential but fast**
4. Fold witness polynomials with challenge — **massively parallel**

If Poseidon2 runs on CPU, you get 2 x l = 32-40 GPU-CPU round-trips per shard. By running the transcript on GPU too, **zero round-trips during sumcheck**.

### GPU architecture

```
+-----------------------------------------------------+
|                    GPU DEVICE                        |
|                                                     |
|  +---------------------------------------------+   |
|  | Sumcheck Controller (single-thread kernel)   |   |
|  |                                              |   |
|  |  for round in 0..l:                          |   |
|  |    1. Launch Q-eval sub-kernel ----------+   |   |
|  |    2. Reduce partial sums                |   |   |
|  |    3. Poseidon2 absorb (on-GPU)          |   |   |
|  |    4. Poseidon2 squeeze -> challenge      |   |   |
|  |    5. Launch fold sub-kernel ----------+  |   |   |
|  |                                        |  |   |   |
|  +----------------------------------------+--+---+   |
|                                           |  |       |
|  +----------------------------------------+--+---+   |
|  |  Q-eval Kernel (launched per round)    |  |   |   |
|  |  - 2^(l-round-1) threads              |  |   |   |
|  |  - Each evaluates Q at one point       |  |   |   |
|  |  - Parallel reduction for sum          <--+   |   |
|  +-------------------------------------------+   |   |
|                                                   |   |
|  +-------------------------------------------+   |   |
|  |  Fold Kernel (launched per round)         |   |   |
|  |  - Halves the evaluation table            |   |   |
|  |  - Each thread folds one pair of evals    <---+   |
|  +-------------------------------------------+       |
|                                                       |
|  Data: witness[2^l x m], transcript_state[8]          |
|  ALL resident on GPU for entire sumcheck duration     |
+-------------------------------------------------------+

CPU side:
  1. Upload witness + CCS structure (once)
  2. Wait for sumcheck completion
  3. Download: round polynomials + final ME claims + transcript state
```

### Mojo implementation approach

Use Mojo's **dynamic parallelism** (kernel launching kernels) or a single persistent kernel with barrier synchronization between phases. On Metal (Apple), use threadgroup barriers and indirect command buffers.

### Reference files

- `crates/neo-reductions/src/sumcheck.rs` — Sumcheck protocol
- `crates/neo-reductions/src/engines/optimized_engine/oracle.rs` — Optimized Q-polynomial evaluation (SparseCache)
- `crates/neo-fold/src/shard/prover.rs` — Shard prover orchestration
- `crates/neo-fold/src/shard/ccs_only_batched.rs` — Batched sumcheck

---

## Phase 5: Integration Architecture (Rust <-> Mojo)

### Option A: Shared Library (Recommended for hot path)

```
+----------------------+     C ABI      +----------------------+
|   Rust (main binary) | <-----------> |  Mojo (.dylib/.so)   |
|                      |               |                      |
|  - RV64IM execution  |    FFI calls  |  - GPU kernel mgmt   |
|  - CCS construction  | -----------> |  - Field arithmetic  |
|  - IVC orchestration |               |  - Poseidon2 batch   |
|  - Proof serializ.   | <----------- |  - Ring multiply     |
|                      |   Results     |  - Sumcheck prover   |
|                      |               |  - Ajtai commitment  |
+----------------------+               +----------------------+
```

Mojo compiles to a shared library exposing C-ABI functions. Rust calls via `extern "C"` FFI — same pattern as the existing `crates/neo-fold-ffi/src/lib.rs` but reversed.

**Pros:** No serialization overhead, shared memory possible, low latency
**Cons:** Tighter coupling, same process crash domain

### Option B: Subprocess with Shared Memory

```
+----------------------+              +----------------------+
|   Rust (main binary) |              |  Mojo (subprocess)   |
|                      |   mmap'd     |                      |
|  Writes witness to --+-- shared --->|  Reads witness       |
|  shared memory       |   memory     |  Runs GPU kernels    |
|                      |              |  Writes results      |
|  Reads results from -+-- shared <---|  to shared memory    |
|  shared memory       |   memory     |                      |
|                      |              |                      |
|  Control: Unix pipe -+- commands -->|  Listens for cmds    |
|                      |<- status ----|  Sends completion    |
+----------------------+              +----------------------+
```

**Pros:** Process isolation, independent updates, GPU crash doesn't kill Rust
**Cons:** IPC overhead (~1-10us per command), more complex lifecycle management

### Recommendation: Hybrid approach

- Use **shared library** for the hot path (sumcheck, commitment) — FFI calls with pointers to pre-allocated GPU-mapped memory
- Use **subprocess** for initialization and teardown — the Mojo process manages GPU device lifecycle, allocates memory pools, precomputes constants (Poseidon2 round constants, SuperNeo matrix, twiddle factors)

### Rust FFI interface (proposed)

```rust
// In crates/neo-fold/src/gpu/mod.rs (new)

extern "C" {
    fn gpu_session_new(
        ccs_ptr: *const u8, ccs_len: usize,
        params_ptr: *const u8, params_len: usize,
    ) -> *mut GpuSession;

    fn gpu_session_prove_shard(
        session: *mut GpuSession,
        witness_ptr: *const u8, witness_len: usize,
        proof_out: *mut *mut u8, proof_len: *mut usize,
    ) -> i32;

    fn gpu_session_destroy(session: *mut GpuSession);
}
```

---

## Data Transfer Strategy (Avoiding Context Switching)

The critical principle: **transfer data to GPU once per shard, get results once per shard.**

### Per-Shard GPU Session

| Phase | Direction | Data | Size (typical) |
|-------|-----------|------|----------------|
| SETUP | CPU -> GPU | CCS structure (sparse) | ~few MB |
| SETUP | CPU -> GPU | Ajtai matrix A (streamed) | ~GB range (tiled) |
| SETUP | CPU -> GPU | Poseidon2 round constants | ~2 KB |
| SETUP | CPU -> GPU | SuperNeo matrix | ~23 KB |
| PER-STEP | CPU -> GPU | Witness matrix z[2^16 x m] | ~100s MB |
| PER-STEP | GPU only | Digit decomposition | stays on GPU |
| PER-STEP | GPU only | Ajtai commitment | stays on GPU |
| PER-STEP | GPU only | Sumcheck (all rounds) | stays on GPU |
| PER-STEP | GPU only | Transcript updates | stays on GPU |
| PER-STEP | GPU -> CPU | Round polynomials | ~few KB |
| PER-STEP | GPU -> CPU | ME claims | ~few KB |
| PER-STEP | GPU -> CPU | Updated transcript state | 64 bytes |
| TEARDOWN | GPU -> CPU | Final proof transcript | ~few KB |

The witness stays on GPU for the entire sumcheck. No intermediate CPU-GPU transfers for challenges — the Poseidon2 transcript runs on-device.

---

## Phase 6: Implementation Roadmap

### Stage 1: Foundation (~4-6 weeks)

1. Set up `nightstream-gpu/` Mojo project with `mojoproject.toml`
2. Implement Goldilocks Fq kernel (add, mul, inv, batch_inv)
3. Implement K = Fq^2 extension kernel
4. GPU correctness tests against Rust `neo-math` reference
5. Benchmark: single-thread CPU vs GPU batch (expect 10-50x for batches > 1K)

### Stage 2: Poseidon2 (~3-4 weeks)

1. Implement Poseidon2 permutation kernel (full + partial rounds)
2. Implement batch sponge (absorb/squeeze for N independent sponges)
3. Implement `Poseidon2TranscriptGpu` matching Rust `Poseidon2Transcript` API
4. Cross-validate: Rust transcript and GPU transcript must produce identical challenges for identical inputs
5. Benchmark: target 100-500x speedup for batch sizes > 256

### Stage 3: Ring & Linear Algebra (~3-4 weeks)

1. Implement R_q schoolbook multiplication kernel
2. Implement SuperNeo transform kernel (batched 54x54 mat-vec)
3. Implement R_q batch operations (add, sub, monomial mul)
4. Cross-validate against Rust `neo-math::ring`

### Stage 4: Sumcheck on GPU (~6-8 weeks)

1. Implement Q-polynomial evaluation kernel
2. Implement witness folding kernel
3. Implement on-GPU sumcheck controller (round loop with transcript)
4. Implement sparse cache GPU equivalent (from optimized_engine)
5. End-to-end test: GPU sumcheck vs CPU sumcheck produce identical proofs
6. This is the hardest phase — the optimized Q-eval in `crates/neo-reductions/src/engines/optimized_engine/oracle.rs` has complex factored algebra that must be faithfully ported

### Stage 5: Ajtai Commitment (~3-4 weeks)

1. Implement tiled matrix-vector product over R_q
2. Implement digit decomposition kernel
3. Memory management for large A matrices (streaming/tiling)
4. Integration with sumcheck — commitment feeds into transcript

### Stage 6: Integration (~4-6 weeks)

1. Build Rust FFI bindings to Mojo shared library
2. Create `GpuProvingSession` in Rust that mirrors `FoldingSession`
3. Feature-flag: `--features gpu` enables GPU path, falls back to CPU
4. End-to-end test: GPU-proved shard verified by CPU verifier (proofs must be identical/compatible)
5. Metal-specific testing on Apple Silicon

---

## Architecture Diagram

```
+------------------------------------------------------------------+
|                         RUST LAYER                                |
|                                                                   |
|  +-------------+  +--------------+  +-------------------------+  |
|  | RV64IM VM   |  | CCS Builder  |  | IVC / Session Manager   |  |
|  | (neo-vm-    |  | (neo-ccs,    |  | (neo-fold)              |  |
|  |  trace)     |  |  neo-memory) |  |                         |  |
|  |             |  |              |  |  Dispatches to GPU or   |  |
|  | Sequential  |  | One-time     |  |  CPU based on feature   |  |
|  | execution   |  | setup        |  |  flag                   |  |
|  +------+------+  +------+-------+  +-----------+-------------+  |
|         |                |                      |                 |
|         v                v                      v                 |
|  +-----------------------------------------------------------+   |
|  |              GPU Proving Interface (Rust FFI)              |   |
|  |  gpu_session_new(ccs, params) -> handle                    |   |
|  |  gpu_session_prove_shard(handle, witness) -> proof         |   |
|  |  gpu_session_destroy(handle)                               |   |
|  +-----------------------------+-----------------------------+    |
|                                | C ABI                            |
+--------------------------------+----------------------------------+
                                 |
+--------------------------------+----------------------------------+
|                    MOJO GPU LAYER (.dylib/.so)                     |
|                                |                                  |
|  +-----------------------------v-----------------------------+    |
|  |              Session Manager (CPU-side Mojo)              |    |
|  |  - GPU device init, memory pool alloc                     |    |
|  |  - Precompute constants (Poseidon2 RC, SuperNeo matrix)   |    |
|  |  - Dispatch kernels, collect results                      |    |
|  +-----------------------------+-----------------------------+    |
|                                |                                  |
|  +-----------------------------v-----------------------------+    |
|  |                    GPU KERNELS                            |    |
|  |                                                           |    |
|  |  +--------------+  +--------------+  +----------------+  |    |
|  |  |  Goldilocks  |  |  Poseidon2   |  |  Ring R_q      |  |    |
|  |  |  Fq / K ops  |  |  Permutation |  |  Multiply      |  |    |
|  |  |              |  |  + Transcript |  |  + SuperNeo    |  |    |
|  |  +--------------+  +--------------+  +----------------+  |    |
|  |                                                           |    |
|  |  +--------------+  +--------------+  +----------------+  |    |
|  |  |  Sumcheck    |  |  Ajtai       |  |  Witness       |  |    |
|  |  |  Q-eval +    |  |  Commitment  |  |  Decomposition |  |    |
|  |  |  Fold        |  |  (tiled)     |  |  + Fold        |  |    |
|  |  +--------------+  +--------------+  +----------------+  |    |
|  |                                                           |    |
|  |  Target: CUDA (NVIDIA) | Metal (Apple) | HIP (AMD)       |    |
|  +-----------------------------------------------------------+    |
+--------------------------------------------------------------------+
```

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Goldilocks mul on GPU needs u128 intermediate | No native u128 on GPU | Use 2 x u64 schoolbook or PTX `mul.hi.u64` intrinsic; Mojo may provide `UInt128` or inline asm |
| Mojo 1.0 not yet released (targeting H1 2026) | API instability | Pin to specific Modular Platform version; keep kernel code isolated from Rust |
| Metal doesn't support u64 natively | Apple Silicon slower for 64-bit field | Emulate via 2 x u32 limbs on Metal; or use 4 x u32 Montgomery representation |
| Sumcheck GPU controller complexity | Hardest engineering task | Start with CPU-orchestrated rounds (accept 2l transfers), then move controller to GPU incrementally |
| Ajtai matrix too large for GPU memory | Memory-bound | Tile/stream A matrix; keep only active tile in GPU VRAM |
| Proof determinism | GPU floating-point non-determinism | Goldilocks is integer arithmetic — no floating point involved; determinism guaranteed |

---

## Expected Performance Gains

| Component | CPU (current) | GPU (projected) | Speedup |
|-----------|--------------|-----------------|---------|
| Poseidon2 (batch 1K) | ~5ms | ~50us | ~100x |
| Ring R_q mul (batch 1K) | ~3ms | ~30us | ~100x |
| SuperNeo transform (batch) | ~2ms | ~20us | ~100x |
| Sumcheck round (2^16 evals) | ~50ms | ~1ms | ~50x |
| Ajtai commitment | ~500ms | ~10ms | ~50x |
| **Full shard proof** | **~2-5s** | **~50-150ms** | **~20-40x** |

*Estimates assume NVIDIA A100/H100 class GPU. Apple M-series Metal will be ~3-5x slower due to u64 emulation but still significantly faster than CPU.*

---

## Key Reference Files

| Component | Location |
|-----------|----------|
| Poseidon2 parameters | `crates/neo-params/src/poseidon2_goldilocks.rs` |
| Goldilocks field ops | `crates/neo-math/src/field.rs` |
| Ring R_q arithmetic | `crates/neo-math/src/ring.rs` |
| S-action | `crates/neo-math/src/s_action.rs` |
| Poseidon2 transcript | `crates/neo-transcript/src/poseidon2.rs` |
| Poseidon2 hash functions | `crates/neo-ccs/src/crypto/poseidon2_goldilocks.rs` |
| RV64IM prover | `crates/neo-fold/src/rv64_trace_shard.rs` |
| RV64 trace layout | `crates/neo-memory/src/riscv/trace/rv64.rs` |
| Shard prover | `crates/neo-fold/src/shard/prover.rs` |
| Shard verifier | `crates/neo-fold/src/shard/verifier_and_api.rs` |
| Sumcheck protocol | `crates/neo-reductions/src/sumcheck.rs` |
| Optimized Q-eval oracle | `crates/neo-reductions/src/engines/optimized_engine/oracle.rs` |
| Ajtai commitment | `crates/neo-ajtai/src/commit.rs` |
| CCS protocol | `crates/neo-ccs/src/` |
| Folding engines | `crates/neo-reductions/src/engines/` |
| C FFI reference | `crates/neo-fold-ffi/src/lib.rs` |
| Parameter validation | `crates/neo-params/src/lib.rs` |
| Twist/Shout memory | `crates/neo-memory/src/twist.rs`, `crates/neo-memory/src/shout.rs` |
| Batched sumcheck | `crates/neo-fold/src/shard/ccs_only_batched.rs` |

---

## References

- [Mojo GPU Programming](https://www.modular.com/mojo)
- [Mojo GPU Fundamentals](https://docs.modular.com/mojo/manual/gpu/fundamentals/)
- [Structured Mojo Kernels](https://www.modular.com/blog/structured-mojo-kernels-part-1-peak-performance-half-the-code)
- [Modular GitHub](https://github.com/modular/modular)
