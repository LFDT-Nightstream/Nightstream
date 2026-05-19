# neo-fold-clean

A paper-faithful, audit-first re-implementation of the SuperNeo IVC integrator.

## What this crate is

The integrator on top of the protocol-primitive crates (`neo-reductions`,
`neo-ccs`, `neo-ajtai`, `neo-transcript`, `neo-math`, `neo-params`, `spartan2`).
It owns:

- The three reductions Π_CCS / Π_RLC / Π_DEC, in paper order, as a thin facade
  over the engine in `neo-reductions`.
- Hypernova Construction 2's augmented function F' and the IVC step.
- The Spartan terminal-compression contract.
- One generic `direct_ccs` frontend.

The protocol math itself lives in the sibling crates and is *not* reimplemented
here.

## What this crate is not

- Not a new fold engine. The optimized engine in `neo-reductions` is reused
  unchanged; the paper-exact engine likewise stays where it is.
- Not a frontend playground. There is one frontend (direct CCS). VM frontends
  are out of scope until this crate is the canonical integrator.
- Not a perf surface. Diagnostics, traces, and shape probes are kept out of
  the protocol path.

## Design rules (non-negotiable)

1. **Paper names.** Code identifiers track the paper symbols defined in
   [`paper/mod.rs`](src/paper/mod.rs). When in doubt, the glossary wins.
2. **Step-down.** Every public function reads top-to-bottom as a sequence of
   named operations. Each step decomposes the same way one level down. No
   spaghetti control flow in protocol paths.
3. **Poseidon2 only** in protocol-binding paths. No mixed hash families.
4. **Digests are compression, never authority.** Every carried digest is
   re-derived by the verifier from authoritative inputs.
5. **Files ≤ 1500 lines.** If a file grows past that, the design is wrong, not
   the line count.
6. **Type the gaps.** A protocol step that isn't proof-complete must be
   represented in the type system, not as a runtime string error.

## Milestone status — SuperNeo / HyperNova IVC core

The SuperNeo / HyperNova §6.3 Construction 2 IVC core path is **complete
for arbitrary fixed-shape R1CS circuits** under the canonical fixed-point
F' image plan. The frontend stack is split by ownership:

- [`frontends::f_prime`](src/frontends/f_prime/) — the
  app-agnostic F' shell: image layout, recursive plan, shell CCS rows,
  encoded step type, shared compiler state, prior-fold verification,
  and unified trace assembly.
- [`frontends::r1cs_f_prime`](src/frontends/r1cs_f_prime/) — generic
  fixed-shape R1CS frontend. Takes a verifier-fixed
  `R1cs { a, b, c, m_in }` and a per-step satisfying assignment
  `z = [x | w]`, emits an encoded F' step the lifecycle folds. R1CS
  product rows `(A_i · z) · (B_i · z) = (C_i · z)` are embedded
  algebraically over `app_assignment` lanes (each variable
  reconstructed from its 64 low-norm bits).
- [`frontends::fibonacci_f_prime`](src/frontends/fibonacci_f_prime/) —
  Fibonacci-specific frontend kept as a concrete app frontend and
  regression fixture. Fibonacci-as-R1CS also goes through the generic
  R1CS compiler (`r1cs_compiler_satisfies_fibonacci_relation`).

Concretely:

- **Fixed-`pc` invariant — Fibonacci frontend.** Base and recursive
  Fibonacci compiler paths share one verifier-owned canonical
  structure (`prep.plan`), shape-validated against
  `post_running.parent_authority`. Pinned by
  `compiler_base_and_recursive_steps_share_structure` running in the
  default suite.
- **Fixed-`pc` invariant — R1CS frontend.** The R1CS frontend uses the
  same verifier-owned-plan discipline, and `preprocess` /
  `preprocess_seeded` reject public-input misconfiguration at the plan
  boundary (`PlanMissingStateXOut`, `PlanAppPublicInputMismatch` —
  pinned by three rejection tests in `system_r1cs_compiler`). The
  structure-digest sanity gate
  (`r1cs_compiler_two_different_shapes_have_different_structure_digests`)
  and the full base+recursive lifecycle gate
  (`r1cs_compiler_base_and_recursive_share_structure`) both run by
  default. The lifecycle gate uses a test-only smaller `Params`
  profile (kappa = 4, m = 2^16, lambda = 60) so prove + extend fits
  under the 5-minute cap; the Goldilocks ring, k_rho, T, and B are
  unchanged, so every Π_RLC / Π_DEC algebraic identity holds
  bit-for-bit.
- **Per-step F' transcript.** Both app compilers call
  `f_prime::compiler::verify_prior_fold`, which rebuilds the
  F'-step transcript that `paper::f_prime::native::prove` initialises.
  A real `StepProof::Recursive` from `audit.steps[i]` is accepted; a
  terminal-fold proof from `finish_uncompressed_with_audit` is rejected
  because it lives under a different transcript label. Pinned by
  `compiler_accepts_real_intermediate_fold_proof`.
- **Unified base/recursive accumulator selector.** `is_base` is
  algebraically bound; the four selector product rows force
  `new_acc_digest = is_base ? H(tag, 0) : H(tag, child_count, c_data…)`.
  Red-team tests in `fibonacci_unified_accumulator_selector.rs` confirm
  flipping `is_base` or selecting a third digest is rejected.
- **Non-replay terminal verifier.** `verify_uncompressed` re-runs the
  terminal NIFS fold against the snapshotted `terminal_inputs` and
  binds the derived state to the recorded `proof.state`; it does **not**
  iterate `audit.steps`. The audit-form verifier
  `verify_uncompressed_audit` adds the chain-replay layer on top.
- **End-to-end on compiler-built chains.** `FibonacciChainBuilder` and
  `R1csChainBuilder` own the compile → prove → derive-next-fold →
  compile → extend loop. Default tests cover single-step terminal verify
  and recursive builder append under smaller test params. The canonical
  big-plan Fibonacci two-step test remains `#[ignore]` because it runs
  the full production-shape compile + 2× extend + finalise + audit verify
  + terminal verify path.
- **Generic R1CS support.** The `r1cs_f_prime` frontend has
  default-green coverage for:
  - R1CS row embedding (satisfying / unsatisfying witness
    accept-reject pair, app-assignment bit-flip detection, R1CS-row-
    count threading).
  - Public-input binding into `state_x_out` — different `x = z[..m_in]`
    produces a different `public_output_digest`; same `x` with
    different `w` produces the same digest
    (`r1cs_compiler_public_output_depends_on_public_input` /
    `_independent_of_private_witness`). The verifier-visible
    `state_x_out` digest absorbs `x` via a
    `PoseidonPreimageLaneSource::AppAssignmentLane` extension, gated by
    `StateXOutPlanOptions::app_public_input_var_indices` (empty for
    Fibonacci, `0..m_in` for R1CS).
  - Preprocessing rejection for miswired public-input indices
    (`PlanMissingStateXOut`, `PlanAppPublicInputMismatch`).
  - Shape-digest separation between distinct R1CS shapes.
  - Full base+recursive lifecycle invariant
    (`r1cs_compiler_base_and_recursive_share_structure`): runs the
    lifecycle end-to-end through `R1csChainBuilder` (preprocess →
    append base assignment → prove → derive recursive fold authority →
    append recursive assignment → extend) under a test-only smaller
    params profile, then asserts base and recursive share one
    structure_digest.

### Default-suite vs. manual-run

The canonical F' image plan is a fixed point of the lifecycle's Π_RLC +
Π_DEC at the production SuperNeo Goldilocks `paper_b2` params
(D=54, KAPPA=18, M=2^30, λ=125). Each lifecycle fold or
`verify_uncompressed` call under that plan is heavy enough that the
following tests are `#[ignore]`'d by default and run only with
`--ignored`:

| Binary | Ignored test(s) | Why ignored | How to run |
|---|---|---|---|
| `system_fibonacci_compiler_unified_structure` | `compiler_two_step_chain_builds_from_scratch_and_verify_uncompressed_accepts` | ~500 s end-to-end (compile + 2× extend + finalize + audit verify + terminal verify) under big plan | `cargo test --release -p neo-fold-clean --test system_fibonacci_compiler_unified_structure -- --ignored compiler_two_step_chain_builds_from_scratch_and_verify_uncompressed_accepts` |
| `system_phase_1_7a_non_linear_verifier` | All 14 tests | Each re-runs the terminal NIFS fold (~70 s × 14 ≫ 5-min cap even with shared bootstrap) | `cargo test --release -p neo-fold-clean --test system_phase_1_7a_non_linear_verifier -- --ignored` |
| `system_ivc_invariants` | `decider_r1cs_size_must_be_constant_in_chain_length` | Two big-plan chains + two finalisations + two terminal syntheses | `cargo test --release -p neo-fold-clean --test system_ivc_invariants -- --ignored` |

The 5-minute per-test cap is recorded in `AGENTS.md`.

### What's intentionally not in this milestone

- **Heterogeneous circuits per chain.** One IVC chain pins one R1CS
  shape (one `pc`, one `F'_j` structure). Multi-shape chains require
  real HyperNova `pc` dispatch: multiple `F'_j` structures plus a
  selector over them. That is a separate design slice.
- **Compressed terminal proof.** The uncompressed verifier is
  non-replay and constant in chain length, but it still re-runs the
  terminal NIFS fold. Spartan compression / on-chain verifier
  integration remains the later proof-size and verifier-cost milestone.
- **Canonical big-plan perf.** Some production-shape regression tests
  remain manual `--ignored` runs. Default tests use smaller algebraic
  params where needed to keep correctness gates fast while preserving
  the Goldilocks ring and Π_RLC / Π_DEC identities.

## Open Questions

Unresolved protocol questions live in [`open-questions/`](open-questions/).
