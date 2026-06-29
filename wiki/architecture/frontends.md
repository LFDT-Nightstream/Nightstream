# Frontends

`crates/neo-fold-clean/src/frontends/` — turn user computation into the
`paper::relations::CcsInstance`s the IVC core folds. Each frontend owns one
translation; the IVC core never knows which frontend produced an instance.

## The contract

A frontend must expose at minimum:

- a user-facing relation type (e.g. `direct_ccs::R1cs`),
- a `preprocess` entry taking caller-supplied protocol params, reading Ajtai setup
  from the verifier-owned global registry (seeded variants are test/demo-only), and
- a `build_instance` entry that validates user input, packs the witness, and commits
  via Ajtai.

## The soundness boundary (read this first)

The chain proof attests three things (from `frontends/mod.rs`):

1. each `CcsInstance` satisfies its CCS relation (checked inside the Π_CCS sum-check),
2. the claims at each step folded correctly via NIFS, and
3. the chain hash binds consecutive states.

What it does **not** attest, until the PR5 decider lands: that each instance is the
*encoding of "F′ ran"* — there is not yet an in-circuit proof tying a folded instance
to the previous recursive step's computation. The IVC core trusts the frontend to
supply instances matching the expected F′. For self-prover use (you produced the
witnesses) this is sufficient; verifying a third party's computation needs the Spartan
terminal compression that proves the F′ binding. See [Roadmap](../roadmap.md).

## direct_ccs — the minimal frontend

R1CS in, foldable instance out (`frontends/direct_ccs/`):

- `R1cs` — the shape `(A, B, C, m_in)`; rows enforce `Az ∘ Bz = Cz` via the CCS
  polynomial `f(X₀,X₁,X₂) = X₀·X₁ − X₂`.
- `build_instance` validates the assignment row-by-row (friendly error at the
  offending row instead of a late sum-check failure), packs `z = [x | w]` into the
  Ajtai-shaped `Z` matrix, and commits.
- `mixers::{ajtai_rlc_mixer, ajtai_dec_mixer}` — the commitment-action closures
  Π_RLC / Π_DEC need.

## f_prime — the shared F′ shell

Not an app frontend itself: the app-agnostic *encoded F′ image*
(`frontends/f_prime/`) that app frontends build on — boundary, lifecycle state,
optional source-image NIFS payloads, accumulator handles/selectors, Poseidon traces,
recursive-step image plan, and the base shell CCS rows.

Key types: `FPrimeImage` / `FPrimeImageLayout` / `FPrimeImageConfig` (the low-norm
bit-image layout), `encode_f_prime_step` (the encoder), `build_f_prime_structure`
(the mixed-gate CCS structure), `RecursiveStepImagePlan` (which image regions the
recursive step binds).

The relation it encodes lives in `paper/f_prime/`; the dependency is strictly
`frontends::f_prime → paper::f_prime`.

## r1cs_f_prime — the production F′ frontend

`frontends/r1cs_f_prime/` pins one R1CS shape per chain and emits encoded F′ steps
whose CCS structure enforces every R1CS row through the existing mixed product gate.
The F′ shell (boundary / state / Poseidon transitions / accumulator handles / NIFS
payload) is identical to the test fixture's; only the appended structure rows and the
interpretation of the app-private region differ — `app_private` carries the
bit-decomposed R1CS assignment.

Entry points: `start_chain` / `compile_step` / `compile_chunk`
(`R1csCompilerContext`), `R1csChainBuilder` + `prove_encoded_steps` for lifecycle
plumbing. The Fibonacci analogue used throughout tests lives in
`tests/support/fibonacci_f_prime/` — it is a fixture, not shipped API.

## bellpepper — circuit adapter

`frontends/bellpepper.rs` synthesizes a `bellpepper_core::Circuit` (the gadget
ecosystem Nova/Spartan use, over an `ff`-style Goldilocks wrapper) into sparse CCS,
returning the full assignment `z = [x | w]`. Adapter only: preprocessing and folding go
through the normal lifecycle. This is how the SHA-256 system tests build their circuits
(`tests/system/sha256_bellpepper_*.rs`).
