# neo-spartan-bridge

**Neo → Spartan2 bridge** using **Plonky3-FRI** as the polynomial commitment scheme (PCS).
This crate provides a production adapter over `p3_fri::TwoAdicFriPcs` for real hash-based FRI.

## What's in this crate

- `P3FriPCSAdapter` — a production adapter that forwards `commit/open/verify` to p3-FRI.
- `make_p3fri_engine_with_defaults(seed)` — builds (adapter, challenger, mmcs materials).
- Helpers to **observe public IO** and **domain-separate** the transcript.

> We **do not** ship simulated FRI; this bridge uses **real hash-based FRI** as required by the repo policy and the Neo paper (single sum-check over an extension field).

## Status

- ✅ **P3‑FRI PCS adapter ready** (`P3FriPCSAdapter`) with real `p3_fri::TwoAdicFriPcs`.
- 🧩 **Spartan2 glue**: wire the adapter via closures or implement Spartan2's engine trait for the adapter (see "Integrating with Spartan2" below).
- 🧪 **Tests**: comprehensive unit and integration tests using the real adapter.

## Quick start

```rust
use neo_spartan_bridge::{make_p3fri_engine_with_defaults, P3FriParams};
use neo_spartan_bridge::pcs::challenger::{observe_commitment_bytes, DS_BRIDGE_COMMIT};

let (pcs, mut ch, mats) = make_p3fri_engine_with_defaults(42);

// Example: bind public IO to transcript (same for prover & verifier)
let io_bytes = /* encode_bridge_io_header(&me) or equivalent */;
observe_commitment_bytes(&mut ch, DS_BRIDGE_COMMIT, &io_bytes);

// Now use `pcs.commit / pcs.open / pcs.verify` directly, or pass closures
```

## Integrating with Spartan2

You have two options:

1. **Closure route**: if your Spartan2 pin exposes `prove_with_pcs / verify_with_pcs`, pass closures that forward to `P3FriPCSAdapter::commit/open/verify`.
2. **Trait route**: if Spartan2 defines a PCSEngine trait, implement it for `P3FriPCSAdapter` and change the engine type from `HyraxPCS` to `P3FriPCSAdapter`.

**Crucial**: bind the same public IO bytes (e.g., `encode_bridge_io_header`) into the challenger on both sides before any commit/open/verify, so the single FS transcript stays consistent.

## Running tests

All tests use the real P3FriPCSAdapter:

```bash
cargo test -p neo-spartan-bridge -- --nocapture
```

**Security notes**:

* Hash & FS: Poseidon2 (width 16, rate 8).
* MMCS: Poseidon2‑Merkle (arity 8).
* Default FRI params: `log_blowup=1`, `num_queries=100`, `pow_bits=16`.
* **No `unsafe`** — crate is compiled with `#![forbid(unsafe_code)]`.

**This bridge provides the production-ready foundation for compressing Neo's folded claims into succinct Spartan2 proofs using real FRI polynomial commitments.**