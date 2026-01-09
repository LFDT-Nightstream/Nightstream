# Spartan Compression of Neo: Two-Phase Plan

This document is the “build plan” implied by `docs/spartan-compression-must-wants.md`, optimized for:
- a **small shareable artifact** (target: low 100s KB total), and
- **no loss of guarantees** vs the native `fold_shard_verify_and_finalize` boundary.

The guiding architecture is: **one blob to share** that may contain multiple proofs internally.

---

## Goals (end-state)

Produce one byte-string artifact (call it `BridgeProofV2`) that a verifier can check using:
- pinned verifying keys (or pinned protocol parameters), and
- a small public statement,

and be convinced that:
1) the native shard verifier would accept the run (folding + Route‑A + Twist/Shout + output binding + step linking), and
2) all deferred obligations are **closed** (native `verify_and_finalize` semantics).

---

## Phase 1 (already implemented): verifier-equivalent compression up to obligations

**What Phase 1 proves:**
- A Spartan2 SNARK attests that the native shard verifier would accept the run *up to* producing `ShardObligations { main, val }`.
- Challenges are transcript-derived in-circuit (no prover-chosen challenges).
- The public statement is replay-resistant (binds params/CCS + VM/program + outputs + step linking).

**Where to look in code:**
- Public statement: `crates/neo-spartan-bridge/src/statement.rs`
- API entrypoints: `crates/neo-spartan-bridge/src/api.rs`
- Circuit: `crates/neo-spartan-bridge/src/circuit/fold_circuit.rs`
- Smoke test / size printing: `crates/neo-spartan-bridge/tests/starstream_compression_smoke.rs`

**What Phase 1 does *not* prove:**
- It does not prove Ajtai commitment opening/correctness / finalization of obligations end-to-end.

---

## Phase 2: close obligations with a succinct closure proof (PQ-friendly, small)

Phase 2 upgrades the artifact from “`verify` semantics” to “`verify_and_finalize` semantics”.

### 2.1 Define the closure contract (consensus-critical)

Write down a single, executable notion of “finalized obligation” that the bridge must attest to.

At minimum, for each final `MeInstance` in `ShardObligations.main` and `ShardObligations.val`, closure means:

- **Bounded witness exists:** there exists a witness matrix `Z` in the bounded domain required for Ajtai binding (digit bounds / ℓ∞ bounds per the preset).
- **Ajtai opening/correctness:** `me.c == Commit(pp, Z)` (for the canonical PP identity).
- **ME consistency:** the carried instance fields (`X`, and any `y` / `y_scalars` semantics used by the verifier) are consistent with the *same* `Z` under CCS semantics.

Also define:
- a **canonical obligation ordering** (main lane first, then val lane; preserve vector order; no sorting), and
- a binding digest (see below).

**Action item (recommended):** put this contract as a doc-comment near the finalizer hook:
- `crates/neo-fold/src/finalize.rs`

### 2.2 Bind PP identity and context

Closure is meaningless if the prover can choose PP.

For the bridge setting, prefer *seeded PP* and bind:
- `pp_seed` (32 bytes),
- dimensions `(d, m, κ)`,
- any version tags that affect seeded generation/chunking.

Bind closure to the same run context as Phase 1 (params/CCS/VM/output binding/step linking).

**Practical binding suggestion (size-friendly):**
- Reuse existing Phase 1 statement digests as the “obligations binding”, e.g. `acc_final_main_digest` and `acc_final_val_digest` already present in `SpartanShardStatement`.
- Define `obligations_digest_v1 = H(acc_final_main_digest || acc_final_val_digest)` (domain-separated).

This avoids shipping (or publicly exposing) large per-obligation objects.

### 2.3 Implement a native reference finalizer (small-m only)

Before building any succinct closure proof, build a “golden oracle” finalizer for tests:
- recompute `Commit(pp, Z)` from explicit `Z`,
- check bounds,
- check ME consistency against CCS semantics.

This is only feasible for small `m` test presets, but it gives you:
- correctness regression tests,
- adversarial tests demonstrating the current “obligations gap”.

### 2.4 Implement `ClosureProofV1` (transparent, PQ-friendly)

Hard constraint: with today’s Ajtai API, `verify_open(pp,c,Z)` is literally `commit(pp,Z) == c` (see `crates/neo-ajtai/src/commit.rs`), and at `m = 2^24` recomputing that inside Spartan R1CS is intractable.

So Phase 2 introduces a separate **transparent** proof system specialized for streaming/linear work (FRI/STARK or sumcheck+FRI family).

**Closure proof statement (public inputs):**
- `context_digest` (binds program/run context),
- `pp_id` (seed + dims + version tags, or the raw tuple),
- `obligations_digest_v1` (binds to the exact obligations implied by Phase 1).

**Closure proof witness:**
- the per-obligation `MeInstance` objects (or whatever canonical “obligation payload” your closure predicate consumes),
- the corresponding witness matrices `Z` (streamed; not materialized into RAM at production scale),
- any auxiliary values needed for ME consistency/bounds checking.

**Batching is the main size lever:**
- Use transcript-derived random coefficients to aggregate checks across obligations where sound (linearity of Ajtai commitments and many consistency relations).
- Derive batching challenges from `context_digest` + `obligations_digest_v1` so the closure proof cannot be replayed across shards/runs.

**Output:**
- `ClosureProofV1` bytes, sized/tuned to keep the overall blob in the low 100s KB.

### 2.5 Package as one blob: `BridgeProofV2`

Recommended v1 packaging:
- `BridgeProofV2 = { spartan_proof_bytes, closure_proof_bytes, (optional) digests }`

The verifier:
1) verifies the Phase 1 Spartan proof against the pinned VK and statement,
2) checks the closure proof against the same `context_digest`/`pp_id`/`obligations_digest_v1`,
3) accepts iff both pass.

Avoid including raw `ShardObligations`/`MeInstance` lists in the blob until you have measured they do not blow the size budget.

### 2.6 Benchmark + tune

Benchmarks to collect early:
- Spartan proof size for representative runs,
- Closure proof size (with batching),
- Prover runtime and memory profile (streaming Z),
- Verifier runtime.

Tune FRI/STARK parameters and batching strategy to hit the blob size target.

---

## Phase 2 “first tickets” (junior-dev executable)

1) **Spec**
- Write the closure predicate and canonical ordering in `crates/neo-fold/src/finalize.rs` docs.
- Decide what exactly `obligations_digest_v1` commits to (recommended: reuse `acc_final_*_digest`).

2) **PP binding**
- Add a helper to produce `(pp_seed, κ, d, m)` and a stable `pp_id` digest (domain-separated).

3) **Native reference finalizer (tests only)**
- Implement a concrete `ObligationFinalizer` that consumes explicit witnesses and checks:
  - Ajtai opening (recompute commitment),
  - bounds,
  - ME consistency.
- Add tests that show: `verify` can pass while `finalize` fails (the gap), and that the bridge rejects once closure is enforced.

4) **Closure proof crate skeleton**
- Define `ClosureProofV1` types + serialization + statement layout.
- Stub `prove_closure_v1` / `verify_closure_v1` with TODOs and wiring tests (use tiny params).

5) **Bridge blob**
- Define `BridgeProofV2` container type + (de)serialization + verifier entrypoint that checks both proofs.

