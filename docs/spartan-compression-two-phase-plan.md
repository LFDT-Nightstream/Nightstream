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

## Status (docs vs code)

This plan originally described “what to build next”. The codebase has moved: most Phase 2 plumbing
and a correctness harness exist today.

**Implemented (and tested):**
- Phase 1 Spartan compression: `crates/neo-spartan-bridge/` (statement + circuit + prove/verify).
- Phase 2 contract + oracle: `crates/neo-fold/src/finalize.rs` (closure contract + `ReferenceFinalizer`).
- Phase 2 proof container + backends:
  - `crates/neo-closure-proof/src/lib.rs` (statement + `ClosureProofV1` container),
  - `crates/neo-closure-proof/src/whir_p3_backend.rs` (WHIR full-closure backend; dev milestone).
- One-blob wrapper: `crates/neo-spartan-bridge/src/bridge_proof_v2.rs` (`BridgeProofV2 = { spartan, closure }`).

**Implemented but not production-ready (dev milestone):**
- WHIR-backed full-closure backend (backend id `5`):
  - code: `crates/neo-closure-proof/src/whir_p3_backend.rs`,
  - still serializes obligations in the payload and materializes large eval tables (disk-backed `mmap` helps, but this is not “obligations-private” yet).

**Missing for audit-ready Phase 2 (production profile):**
- obligations-private redesign (closure proof must not ship obligations, and must prove digest binding + weights computation): `docs/spartan-compression-phase2-obligations-private.md`.
- pinned/tuned WHIR parameters + explicit security rationale.
- clear resource envelope (RAM/disk/time) and size caps for the proof artifact.

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

## Phase 2: close obligations with a closure proof (succinct backend is WIP)

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
- Define `obligations_digest = H(acc_final_main_digest || acc_final_val_digest || pp_id_digest)` (domain-separated), where `H` is ZK-friendly (e.g. Poseidon2 over Goldilocks).

This avoids shipping (or publicly exposing) large per-obligation objects.

### 2.3 Implement a native reference finalizer (small-m only)

Before building any succinct closure proof, build a “golden oracle” finalizer for tests:
- recompute `Commit(pp, Z)` from explicit `Z`,
- check bounds,
- check ME consistency against CCS semantics.

This is only feasible for small `m` test presets, but it gives you:
- correctness regression tests,
- adversarial tests demonstrating the current “obligations gap”.

### 2.4 Implement `ClosureProofV1` (proof container + backends)

Hard constraint: with today’s Ajtai API, `verify_open(pp,c,Z)` is literally `commit(pp,Z) == c` (see `crates/neo-ajtai/src/commit.rs`), and at `m = 2^24` recomputing that inside Spartan R1CS is intractable.

So Phase 2 introduces a separate **transparent** proof system specialized for streaming/linear work (FRI/STARK or sumcheck+FRI family).

**Closure proof statement (public inputs):**
- `context_digest` (binds program/run context),
- `pp_id` (seed + dims + version tags, or the raw tuple),
- `obligations_digest` (binds to the exact obligations implied by Phase 1).

**Closure proof witness:**
- the per-obligation `MeInstance` objects (or whatever canonical “obligation payload” your closure predicate consumes),
- the corresponding witness matrices `Z` (streamed; not materialized into RAM at production scale),
- any auxiliary values needed for ME consistency/bounds checking.

**Current code reality:**
- `ClosureStatementV1` / `ClosureProofV1`: `crates/neo-closure-proof/src/lib.rs`
- WHIR full-closure backend (dev milestone, backend id `5`): `crates/neo-closure-proof/src/whir_p3_backend.rs`
  - still obligations-public + large-table materialization; see `docs/spartan-compression-phase2-obligations-private.md`

**Batching is the main size lever:**
- Use transcript-derived random coefficients to aggregate checks across obligations where sound (linearity of Ajtai commitments and many consistency relations).
- Derive batching challenges from `context_digest` + `obligations_digest` so the closure proof cannot be replayed across shards/runs.

**Output:**
- `ClosureProofV1` bytes, sized/tuned to keep the overall blob in the low 100s KB.

### 2.5 Package as one blob: `BridgeProofV2`

Recommended v1 packaging:
- `BridgeProofV2 = { spartan_proof_bytes, closure_proof_bytes, (optional) digests }`

The verifier:
1) verifies the Phase 1 Spartan proof against the pinned VK and statement,
2) checks the closure proof against the same `context_digest`/`pp_id`/`obligations_digest`,
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

## Phase 2 “first tickets” (now done)

These “junior-dev executable” steps are already implemented in code:
- Closure contract: `crates/neo-fold/src/finalize.rs`
- PP binding: `pp_id_digest` in `crates/neo-spartan-bridge/src/api.rs` and `crates/neo-ajtai`
- Reference finalizer: `neo_fold::finalize::ReferenceFinalizer`
- Closure proof crate + explicit backend: `crates/neo-closure-proof/`
- One-blob artifact: `crates/neo-spartan-bridge/src/bridge_proof_v2.rs`

## Phase 2 remaining tickets (audit-ready production profile)

1) **Obligations-private WHIR redesign**
- Remove obligations from WHIR payloads while preserving binding to Phase 1 (`obligations_digest`) and verifier determinism.
- Prove digest binding + weight/claim computation inside the Phase‑2 proof.
- Worklist: `docs/spartan-compression-phase2-obligations-private.md`

2) **Pin security parameters + document them**
- WHIR config (security level, folding factors, PoW policy, etc) must be pinned and justified.

3) **Resource envelope + size caps**
- Define verifier size limits and “expected” prover RAM/disk envelopes.

4) **E2E tests for production profile**
- Add e2e tests that verify the obligations-private backend roundtrips and rejects tampering, with size regression guards.

---

## Appendix: Labrador/Condor (LaBRADOR) evaluation

We briefly evaluated `external/condor-rs/labrador` (LaBRADOR: a lattice-based, post-quantum proof
system for R1CS) as a potential replacement for the “Spartan2 + WHIR closure” direction.

### Summary verdict

LaBRADOR is **not a drop-in replacement** for either Phase 1 or Phase 2 in this plan. Adopting it
would be a **major protocol + implementation rework**, and it does not directly address the
production-scale streaming concerns that dominate Phase 2 at `m = 2^24`.

### Why it doesn’t plug into this plan (today)

1) **Algebra mismatch**
- LaBRADOR’s implementation is hard-coded to `R = Z_q[X]/(X^64 + 1)` with `q = 2^32 − 1`
  (composite modulus, not a prime field).
- Neo’s folding/CCS/ME machinery is built around Goldilocks (prime field) and the Φ₈₁ ring
  `F_q[X]/(X^54 + X^27 + 1)`.

Bridging these is not “parameter tuning”: it is a re-derivation of the protocol’s arithmetic layer,
plus extensive refactoring to make the library generic over ring/modulus choices.

2) **Relation interface mismatch (not a “prove my circuit” backend yet)**
- The current `labrador` crate constructs toy `Statement`/`Witness` instances internally
  (random constraints derived from the witness) rather than exposing an adapter for arbitrary R1CS /
  CCS-like relations.

To use LaBRADOR for Neo, we would need to build a full translation from our Phase‑1/Phase‑2
predicates into LaBRADOR’s dot-product constraint system, plus matching serialization and
determinism requirements.

3) **Verifier embedding / recursion mismatch**
This bridge plan is built around “verify inside a small verifier context” (and in Phase 1, inside a
Spartan circuit). Making LaBRADOR usable here would require one of:
- a new circuit that verifies LaBRADOR proofs (in lattice arithmetic), or
- abandoning circuit-verifiable recursion and treating LaBRADOR as a standalone external verifier
  (changing the entire architecture and trust/UX assumptions).

4) **Streaming and memory are still the hard problem**
Even if we had a relation adapter, LaBRADOR does not automatically give “true streaming” for the
`m = 2^24` witness sizes that drive Phase 2. The core issue is avoiding large `2^n` materializations
and/or massive intermediate tables during commitment/sumcheck/PCS-style operations. Any candidate
system still needs an out-of-core design story that avoids blowups.

### When it might be worth revisiting

If the product goal shifts to “post-quantum proofs” as the primary objective (and we accept a larger
integration effort + new verifier constraints), then LaBRADOR could be pursued as a longer-term R&D
track. The minimum gating milestones would be:
- make the algebra layer generic (or otherwise align it with Neo’s field/ring requirements),
- implement a real relation adapter for our predicates (not toy-generated constraints),
- benchmark proof sizes/runtime on Neo-shaped instances,
- decide how verification is embedded (standalone vs recursive/in-circuit).

### Note on current code health

As imported, `external/condor-rs/labrador` had a failing unit test in release mode; we fixed it so
the crate is green under `cargo test --release -p labrador`. This does not change the architectural
conclusion above, but it keeps the dependency in a usable state for further evaluation.
