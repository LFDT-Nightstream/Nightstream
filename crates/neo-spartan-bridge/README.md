# neo-spartan-bridge

Experimental integration layer between Neo folding (Π-CCS / `FoldRun`) and the Spartan2 SNARK.

> **Status:** Folding + Route‑A memory terminal checks are implemented. Output binding is implemented behind `statement.output_binding_enabled=true` (tests are `#[ignore]` because they are very slow).

---

## Goal

Provide a single Spartan2 proof that attests to the correctness of an entire Neo `FoldRun`, with:

- Π-CCS **sumcheck invariants and terminal identity** enforced as R1CS constraints.
- Π-RLC and Π-DEC **linear/base‑b equalities** enforced as constraints (including commitment linear relations).
- **Accumulator chaining** across all steps, from an initial public accumulator to a final public claim.

The Spartan proof uses whatever PCS is chosen by the Spartan **engine** (Hash-MLE today). The bridge only defines a bellpepper circuit over Neo's Goldilocks arithmetic; it does not introduce a second PCS.

---

## Architecture

The crate is split into:

1. **`circuit/`** – R1CS circuit for a `FoldRun`:
   - `FoldRunInstance` – public IO container (`SpartanShardStatement` only).
   - `FoldRunWitness` – private witness (`ShardProof`, per‑step `McsInstance`, initial accumulator).
   - `FoldRunCircuit` – synthesizes constraints for all steps and enforces accumulator endpoint digests.

2. **`gadgets/`** – small reusable gadgets:
   - `k_field` – K-field (degree-2 extension) as 2 limbs over the base field.
   - `poseidon2` / `sponge` / `transcript` – Poseidon2 permutation + in-circuit `Poseidon2Transcript` (Fiat–Shamir source of truth).
   - `sumcheck` – transcript-bound sumcheck gadgets (single + batched, DS framed).

3. **`api/`** – high-level `setup_fold_run` / `prove_fold_run` / `verify_fold_run` API:
   - `setup_fold_run` returns a pinned `(pk, vk)` for a circuit shape (verifiers must not accept a prover-supplied `vk`).
   - `prove_fold_run` produces `SpartanProof { proof_data, statement }`.
   - `verify_fold_run` verifies using a pinned `vk` and checks statement digests (params/CCS/MCS) against the verifier’s view.

---

## Current Implementation

To run the slow RV32 compression tests: `cargo test -p neo-spartan-bridge --release -- --ignored`.

### Phase 1 meaning

The current Spartan proof attests: “there exists a `ShardProof` such that the circuit’s in-circuit verifier accepts for every step, with all verifier coins derived via the canonical Neo transcript.”

Today the circuit covers:
- Π‑CCS verification (Route‑A batched time + Ajtai rounds + terminal identity),
- Π‑RLC/Π‑DEC for the main lane,
- Transcript-derived ρ sampling for Π‑RLC,
- **If `mem_enabled=true`**:
  - Route‑A memory verification (Shout/Twist addr-pre, batched time multi-claim, and terminal algebra checks),
  - Twist val-eval batch (derives `r_val`) + val-lane Π‑RLC/Π‑DEC verification,
  - rollover checks (when a previous step exists).
- **If `output_binding_enabled=true`** (last step only):
  - output sumcheck verification and final output equation, including the `output_binding/inc_total` linkage to Twist time-lane openings.

Limitations:
- Shout `table_spec=None` is rejected in the compression profile (only `LutTableSpec::RiscvOpcode` is supported today).
- `program_digest` is currently a placeholder (0) in the API; use `steps_digest` / `program_io_digest` for binding for now.

### Π‑CCS side

- **Initial sum T (`claimed_initial_sum`)**
  - `claimed_initial_sum_gadget` mirrors `claimed_initial_sum_from_inputs` in `neo_reductions`:
    - Same Ajtai MLE χ_α construction and bit ordering.
    - Same γ-weight schedule and outer γ^k factor.
  - The circuit enforces `proof.sc_initial_sum == T_gadget` whenever the proof supplies `sc_initial_sum`.

- **Route‑A batched time + Ajtai rounds**
  - Phase 1 verifies the Route‑A batched time proof for the CCS/time claim, deriving `r_time` and binding it to `ccs_out[0].r`.
  - Ajtai rounds are verified via the transcript-bound sumcheck gadget, and the final running sum is enforced to equal `proof.sumcheck_final`.

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
  - The terminal identity uses the same transcript-derived `(α,β,γ,r′,α′)` variables used everywhere else (no unconstrained duplicates).

### Fiat–Shamir (Phase 1)

- The circuit maintains an in-circuit `Poseidon2TranscriptVar` (Goldilocks, WIDTH=8, RATE=4) matching `neo_transcript::Poseidon2Transcript` framing.
- Π‑CCS challenges `(α,β_a,β_r,γ)`, Route‑A points, and all sumcheck per-round challenges are sampled from this transcript in-circuit and enforced against the proof’s embedded values.
- Π‑RLC ρ matrices are enforced to match transcript-derived sampling.
  - The sampler is the “no rejection” variant (`u16 % 5`) to keep transcript consumption fixed-length; this matches the current `neo-reductions` implementation.

### RLC / DEC / chaining

- **RLC / DEC equalities**
  - `verify_rlc` and `verify_dec` enforce:
    - Correct random linear combination of `X`, `y`, and `r` across children.
    - Correct base‑b decomposition of vectors into Ajtai digits, consistent with the native Π‑RLC/Π‑DEC reductions.
    - Commitment equalities (`c` coordinates) for Π‑RLC and Π‑DEC, mirroring the native linear relations (commitment *correctness/openings* remains external).

- **Accumulator binding / chaining**
- The circuit threads each step’s `dec_children` variables into the next step’s Π‑CCS checks (no “re-allocation drift” across steps).
  - The public statement binds to:
    - `acc_init_digest`: digest of the initial accumulator,
    - `acc_final_main_digest`: digest of the final main-lane accumulator,
    - `acc_final_val_digest`: digest of the final val-lane obligations accumulator (empty for folding-only runs; non-empty when Twist val-lane folding is present).
  - Digests are Poseidon2-based (`acc_digest/v2`, including commitment + X + r/y/y_scalars) and are enforced inside the circuit.

### Spartan2 integration

- `api::setup_fold_run`:
  - Runs Spartan2 `setup` on a circuit shape to produce `(pk, vk)`.
  - The verifier key must be pinned out-of-band.

- `api::prove_fold_run`:
  - Enforces host-side degree bounds on Π‑CCS sumcheck polynomials.
  - Builds `FoldRunInstance` + `FoldRunWitness`.
  - Constructs `FoldRunCircuit` and runs:
    - `R1CSSNARK::prep_prove`,
    - `R1CSSNARK::prove`,
  - Serializes the `snark` into `SpartanProof::proof_data` (verifier key is not bundled).

- `api::verify_fold_run`:
  - Recomputes `(params_digest, ccs_digest, steps_digest)` and checks them against the proof’s statement.
  - Deserializes `snark` and runs Spartan verification with a pinned `vk`.
  - Checks Spartan’s returned public IO matches the statement encoding.

---

## Remaining Work

Depending on how much of the folding stack we want Spartan to attest to, the main missing pieces are:

- **Program/VM binding**
  - Define and enforce a real `program_digest` (ROM / ELF / cfg) in the statement.
  - Optionally bind a step-linking policy digest.

- **Step linking**
  - Wire a pinned step-linking config into the circuit shape and statement profile.

- **Commitment-level consistency (optional)**
  - Commitment equalities are enforced, but Spartan does not prove Ajtai commitment correctness/openings yet; this remains an explicit external contract.

---

## Safety and Caveats

- This crate is **experimental** and should not yet be treated as a hardened verification layer.
- Commitment correctness is still delegated to the outer Ajtai verifier; Spartan enforces the same *linear commitment relations* checked by the native folding verifier.

---

## License

Apache-2.0
