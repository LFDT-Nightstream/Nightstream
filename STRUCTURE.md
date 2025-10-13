# STRUCTURE.md — Workspace Overview (Goldilocks‑only)

Goal: document the current Nightstream workspace implementing lattice‑based folding with Ajtai commitments (pay‑per‑bit), a single sum‑check over the extension field K = F_{q^2}, reductions Π_CCS → Π_RLC → Π_DEC under one Poseidon2 transcript, strong‑set challenges, and a Spartan2 Hash‑MLE SNARK as the last mile. The base field is fixed to Goldilocks q = 2^64 − 2^32 + 1.

---

## Repository Layout

```
Cargo.toml            # [workspace] members, shared profiles
crates/
  neo/                # Facade API (prove/verify, IVC/NIVC), examples
  neo-params/         # Typed parameter sets + guard/extension policy
  neo-math/           # Goldilocks field, K=F_{q^2}, ring (Rq), S‑action
  neo-ajtai/          # Ajtai matrix commitment + decomp/split + S‑homomorphism
  neo-ccs/            # CCS frontend, relations (MCS/ME), matrices, poly, gadgets
  neo-fold/           # Folding pipeline: Π_CCS, Π_RLC, Π_DEC (one transcript)
  neo-transcript/     # Poseidon2 transcript (Fiat–Shamir) and helpers
  neo-spartan-bridge/ # ME → Spartan2 (Hash‑MLE PCS) + unified Poseidon2 IO
  neo-cli/            # CLI demos (Fibonacci IVC, lean proof packaging)
  neo-tests/          # Integration and facade tests
  neo-redteam-tests/  # Security/attack tests (binding, parity, tamper)
  neo-quickcheck-tests/ # Property‑based tests (proptest)
docs/                 # System architecture docs
```

Simplified dependencies (acyclic):

```
neo-params → neo-math → { neo-ajtai, neo-ccs, neo-spartan-bridge }
neo-transcript → neo-fold (single transcript owner)
neo-fold → neo-spartan-bridge (last‑mile SNARK)
neo (facade) uses { neo-params, neo-ajtai, neo-ccs, neo-fold, neo-spartan-bridge }
neo-cli depends on neo (examples/demos)
tests crates depend across the stack
```

Rationale: mirrors Nightstream’s design—Ajtai with pay‑per‑bit and S‑homomorphism; one small‑field sum‑check over K=F_{q^2}; CCS→ME relations; RLC with short‑norm ρ from a strong set; DEC to keep norms in range; final compression via Spartan2 Hash‑MLE over the same field.

---

## Key Crates

• neo (facade)
- Simple proving/verification API wrapping the end‑to‑end pipeline.
- Exposes prove/verify, IVC, and NIVC drivers; uses a lean proof format bound to a context digest (Poseidon2) and a VK digest registry.
- Examples: `fib_benchmark.rs`, `fib_folding_nivc.rs`, `incrementer_folding.rs`, `nivc_demo.rs`.

• neo-params
- `NeoParams` with validation: enforces (k+1)·T·(b−1) < B where B = b^k.
- Extension policy v1: supports s = 2 only; provides auto‑tuned Goldilocks presets (e.g., `goldilocks_autotuned_s2`).

• neo-math
- Goldilocks base field and extension K = F_{q^2}.
- Cyclotomic ring R_q with η = 81 (d = 54), coefficient maps cf/cf^{-1}, S‑module rotation matrices.

• neo-ajtai
- Ajtai matrix commitment L: F_q^{d×m} → C; S‑homomorphic, (d,m,B)‑binding.
- Pay‑per‑bit embedding; exports `decomp_b`/`split_b` and constant‑time commit paths.

• neo-ccs
- CCS structures, relations (MCS, ME), matrices/polynomials, and gadgets.
- Production Poseidon2 crypto module used for context digests and transcripts.

• neo-transcript
- Poseidon2 transcript over Goldilocks (WIDTH=12, RATE=8, CAP=4); Merlin‑style API.

• neo-fold
- Owns the single Poseidon2 transcript and the only sum‑check over K.
- Implements Π_CCS (sum‑check over Q), Π_RLC (S‑action combination with guard), Π_DEC (verified split/opening).
- Bridge adapter to legacy ME types for Spartan2.

• neo-spartan-bridge
- Converts ME(b,L) to Spartan2 R1CS with Hash‑MLE PCS; unified Poseidon2 public IO encoding.
- Emits and verifies lean proofs (no embedded VK), bound via VK digest and context digest.

• neo-cli
- Demo CLI to generate/verify Fibonacci proofs; supports bundling or separate VK.

---

## Parameters (current presets)

- Field: q = 2^64 − 2^32 + 1 (Goldilocks). Extension: K = F_{q^2} (s = 2).
- Cyclotomic: η = 81, Φ_η = X^{54} + X^{27} + 1, d = 54.
- Typical Ajtai/MSIS: κ = 16; m sized per application.
- Norm schedule: b = 2, k = 12 → B = 4096.
- Strong set C (coeffs in {−2…2}) with empirical expansion T ≈ 216.
- Guard: (k+1)·T·(b−1) = 2808 < B = 4096 ✓ (enforced).
- Presets: `NeoParams::goldilocks_autotuned_s2(..)` and `goldilocks_127()`; v1 allows only s=2 and computes slack bits accordingly.

---

## Pipeline & Dataflow (one fold iteration)

1) Embed & commit: z = x || w → Z = Decomp_b(z) → c = L(Z)  [MCS].
2) Π_CCS (one sum‑check over K): batch Q for CCS F, range/norm polynomials, and eval ties; derive (α′, r) and produce k ME(b,L).
3) Π_RLC: sample ρ_i from strong set C; fold k+1 → 1 to ME(B,L) under guard (k+1)·T·(b−1) < B.
4) Π_DEC: split high‑norm parent back to k ME(b,L) with verified recomposition and range checks.
5) Last mile: Spartan2 Hash‑MLE SNARK proves ME evaluations; proofs are lean and bound to a Poseidon2 context digest of (CCS, public_input).

Transcript: single Poseidon2 transcript across Π_CCS, Π_RLC, Π_DEC in neo‑fold, and a unified Poseidon2 IO format in the bridge.

---

## Entrypoints

- Library: `neo::{prove, verify, ProveInput, CcsStructure, NeoParams, F}`.
- CLI: `cargo run -p neo-cli -- gen …` and `-- verify …` for Fibonacci demos.
- Examples: `cargo run -p neo --example fib_benchmark --release`.

---

## Tests (by crate)

- neo-ajtai: commitment binding, S‑homomorphism, decomp/split, constant‑time paths, PRG; red‑team tests.
- neo-ccs: CCS properties, commitment gadgets, direct sums, Poseidon2 tests, red‑team cases.
- neo-fold: sum‑check, pipeline composition, RLC/DEC range and tamper tests, transcript parity/uniqueness.
- neo-spartan-bridge: bridge smoke/integration, public‑IO parity/tamper, lean proof tests, VK digest binding.
- neo (facade): IVC/NIVC flows, binding parity, context digest parity, end‑to‑end folding and finalization tests.
- neo-quickcheck-tests: properties for DEC/RLC/bridge headers and security red‑team checks.
- neo-redteam-tests: targeted attack scenarios across folding, bridge, and params guards.

Run: `cargo test --workspace` (see README for feature flags used in security/property suites).

---

## Guardrails

- Exactly one sum‑check over K (in neo‑fold); never over rings.
- Ajtai is the only commitment backend; decomposition/range checks are mandatory inside the same transcript as CCS.
- Strong sampler C with invertible differences and bounded expansion T must be used in Π_RLC.
- Spartan2 Hash‑MLE compression only at the end; unified Poseidon2 IO and context binding.

---

## Notes & Docs

- See `docs/system-architecture.md` for IVC with public‑ρ EV and on‑demand emission.
- README contains quick‑start, examples, security posture, and roadmap.
- This is research software; lean proofs use a VK digest registry and Poseidon2 context binding to prevent replay.

