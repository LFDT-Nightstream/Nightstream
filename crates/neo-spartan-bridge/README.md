# neo-spartan-bridge

Experimental integration layer between Neo folding (Π-CCS / `FoldRun`) and the Spartan2 SNARK.

> **Status:** WIP / non-production. The crate now builds a concrete `FoldRun` circuit and produces real Spartan2 proofs, but some cross-step accumulator constraints and commitment-level checks are still TODO (see “Remaining work”).

---

## Goal

Provide a single Spartan2 proof that attests to the correctness of an entire Neo `FoldRun`, with:

- Π-CCS **sumcheck invariants and terminal identity** enforced as R1CS constraints.
- Π-RLC and Π-DEC **linear/base-b equalities** enforced as constraints.
- **Accumulator chaining** across all steps, from an initial public accumulator to a final public claim.

The Spartan proof uses whatever PCS is chosen by the Spartan **engine** (Hash-MLE today). The bridge only defines a bellpepper circuit over Neo's Goldilocks arithmetic; it does not introduce a second PCS.

---

## Architecture

The crate is split into:

1. **`circuit/`** – R1CS circuit for a `FoldRun`:
   - `FoldRunInstance` – public IO container (digests, accumulator claim, Π-CCS challenge data).
   - `FoldRunWitness` – private witness (FoldRun, Π-CCS proofs, Z, ρ, DEC digits).
   - `FoldRunCircuit` – synthesizes bellpepper constraints for all steps and implements `SpartanCircuitTrait<GoldilocksP3MerkleMleEngine>`.

2. **`gadgets/`** – small reusable gadgets:
   - `k_field` – K-field (degree-2 extension) as 2 limbs over the base field.
   - `pi_ccs` – sumcheck-related helpers and legacy base-b recomposition.
   - `common` – experimental helpers (any unsafe/`cs.get()`-based gadgets are gated behind `unsafe-gadgets` and unused in the main circuit).

3. **`api/`** – high-level `prove_fold_run` / `verify_fold_run` API:
   - Builds a `FoldRunCircuit` and uses Spartan2’s `R1CSSNARK` over `GoldilocksP3MerkleMleEngine` to produce and verify proofs.
   - Public IO is the tuple of `(params_digest, ccs_digest, mcs_digest)` encoded as field limbs.

4. **`engine/`** – experimental hooks for Z-polynomial layout. This is **not required** for Spartan2 integration and is gated behind the `experimental-engine` feature.

---

## Current Implementation

### Π‑CCS side

- **Initial sum T (`claimed_initial_sum`)**
  - `FoldRunCircuit::verify_initial_sum_binding` allocates `α` and `γ` as K variables, lifts ME input `y`-tables into K, and calls `claimed_initial_sum_gadget`.
  - `claimed_initial_sum_gadget` mirrors `claimed_initial_sum_from_inputs` in `neo_reductions`:
    - Same Ajtai MLE χ_α construction and bit ordering.
    - Same γ-weight schedule and outer γ^k factor.
  - The circuit enforces `proof.sc_initial_sum == T_gadget` whenever the proof supplies `sc_initial_sum`.

- **Sumcheck rounds**
  - `verify_sumcheck_rounds`:
    - Allocates round polynomials in K from native coefficients.
    - Uses `sumcheck_round_gadget` for each round to enforce `p(0)+p(1) = claimed_sum`.
    - Uses `sumcheck_eval_gadget` to set the next `claimed_sum = p(challenge)`.
    - At the end, enforces the in-circuit running sum equals `proof.sumcheck_final`.

- **Equality polynomials `eq((α′,r′),·)`**
  - `FoldRunCircuit::eq_points` implements the equality polynomial over K:
    - For vectors `p, q`, computes `∏_i [1 - (p_i + q_i) + 2 p_i q_i]`.
    - Uses one K multiplication per coordinate (`p_i * q_i`) and only linear operations otherwise.
    - Anchors the constant `1` via `k_one` and uses native `neo_math::K` hints for all K multiplications.
  - `verify_terminal_identity` uses this gadget to compute:
    - `eq((α′,r′), β) = eq(α′, β_a) * eq(r′, β_r)`,
    - `eq((α′,r′),(α,r)) = eq(α′, α) * eq(r′, r)`, when ME inputs exist.

- **Terminal identity RHS**
  - Implemented directly in `FoldRunCircuit::verify_terminal_identity`:
    - Recomputes `F′` from the first ME output’s Ajtai digits via an in-circuit base‑b recomposition with native K hints.
    - Computes range products `N′_i` over K (Ajtai norm constraints) using a K-valued range gadget.
    - Builds χ_{α′} and evaluates the linearized CCS views to obtain `Eval′`.
    - Assembles
      - `v = eq((α′,r′),β) · (F′ + Σ γ^i N′_i) + γ^k · eq((α′,r′),(α,r)) · Eval′`,
      - and enforces `v == proof.sumcheck_final` in K.
  - The older stubbed `terminal_identity_rhs_gadget` in `gadgets/pi_ccs.rs` has been removed to avoid unsound usage.

### RLC / DEC / chaining

- **RLC / DEC equalities**
  - `verify_rlc` and `verify_dec` enforce:
    - Correct random linear combination of `X`, `y`, and `r` across children.
    - Correct base‑b decomposition of vectors into Ajtai digits, consistent with the native Π‑RLC/Π‑DEC reductions.
  - Commitment consistency (`c` values) is intentionally left to the outer Ajtai verifier for now; the Spartan circuit only enforces algebraic relations on `X, y, r`.

- **Accumulator chaining**
  - `verify_accumulator_chaining` ties:
    - The public final accumulator in `FoldRunInstance` to the last fold step’s outputs.
  - A full cross-step chaining (from initial accumulator through all DEC children to the final accumulator) is still TODO (see below).

### Spartan2 integration

- `api::prove_fold_run`:
  - Enforces host-side degree bounds on Π‑CCS sumcheck polynomials.
  - Builds `FoldRunInstance` + `FoldRunWitness` and extracts per-step challenges.
  - Constructs `FoldRunCircuit` and runs:
    - `R1CSSNARK::setup`,
    - `R1CSSNARK::prep_prove`,
    - `R1CSSNARK::prove`,
  - Serializes `(vk, snark)` into `SpartanProof::proof_data`.

- `api::verify_fold_run`:
  - Recomputes `(params_digest, ccs_digest)` and checks them against the proof’s instance.
  - Reconstructs the expected public IO (digest limbs).
  - Deserializes `(vk, snark)` and runs Spartan verification.
  - Checks Spartan’s returned public IO matches the instance digests.

---

## Remaining Work

Depending on how much of the folding stack we want Spartan to attest to, the main missing pieces are:

- **Stronger accumulator chaining**
  - Extend `verify_accumulator_chaining` so that:
    - Step 0 inputs are tied to the initial accumulator,
    - Each step’s DEC children are tied to the next step’s ME inputs,
    - The final outputs match the public final accumulator (already enforced).

- **Commitment-level consistency (optional)**
  - If we want Spartan to attest to Ajtai commitments as well:
    - Add linear constraints on `c` in `verify_rlc` / `verify_dec` mirroring the native RLC/DEC commitment relations.
    - Optionally expose a commitment to the final accumulator as part of the public IO.

- **Public IO / statement design**
  - Decide whether the Spartan statement should:
    - encode only digests (current approach),
    - also encode original R1CS public IO,
    - or include final accumulator commitments.

- **Dead-code cleanups**
  - `gadgets/pi_ccs.rs::base_b_recompose_k` is now legacy (the circuit recomposes Ajtai rows inline with K hints) and can be removed or clearly marked as such.

---

## Safety and Caveats

- This crate is **experimental** and should not yet be treated as a hardened verification layer.
- Commitment correctness is still delegated to the outer Ajtai verifier; Spartan currently checks the Π‑CCS algebra and the folding of evaluation views.
- Any gadget that uses `cs.get()` remains gated behind the `unsafe-gadgets` feature and must not be used in production circuits.

---

## License

Apache-2.0
