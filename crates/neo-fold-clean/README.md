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

## Milestone status — SuperNeo / HyperNova core for Fibonacci

The SuperNeo / HyperNova §6.3 Construction 2 IVC core path is **complete
for the Fibonacci frontend** under the canonical fixed-point F' image
plan. Concretely:

- **Fixed-`pc` invariant.** Base and recursive Fibonacci F' compiler
  paths share one verifier-owned canonical structure (`prep.plan`),
  shape-validated against `post_running.parent_authority`. Pinned by
  `compiler_base_and_recursive_steps_share_structure`.
- **Per-step F' transcript.** The compiler's `verify_prior_fold`
  rebuilds the F'-step transcript that
  `paper::f_prime::native::prove` initialises, so a real
  `StepProof::Recursive` from `audit.steps[i]` (NOT a terminal-fold
  proof from `finish_uncompressed_with_audit`) is what the compiler
  accepts. Pinned by `compiler_accepts_real_intermediate_fold_proof`.
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
- **End-to-end on a compiler-built chain.**
  `compiler_two_step_chain_builds_from_scratch_and_verify_uncompressed_accepts`
  compiles base + recursive Fibonacci steps, folds them through the
  lifecycle, finalises, and accepts under the audit verifier. The
  terminal-only verifier is covered by the single-step compiler chain
  and lower-level terminal-fold red-team tests.

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
| `system_phase_1_7a_non_linear_verifier` | All 14 tests | Each re-runs the terminal NIFS fold (~70 s × 14 ≫ 320-second cap even with shared bootstrap) | `cargo test --release -p neo-fold-clean --test system_phase_1_7a_non_linear_verifier -- --ignored` |
| `system_ivc_invariants` | `decider_r1cs_size_must_be_constant_in_chain_length` | Two big-plan chains + two finalisations + two terminal syntheses | `cargo test --release -p neo-fold-clean --test system_ivc_invariants -- --ignored` |

The 320-second per-test cap is recorded in `AGENTS.md`.

### What's intentionally not in this milestone

- **Generic app-witness compiler.** Only Fibonacci is wired through
  `compile_fibonacci_step`. The compiler surface is structured to
  generalise (`FibonacciAppStepInput` is a thin app-specific shell over
  the canonical F' image plumbing), but adding a second frontend is a
  separate slice.
- **Caller-side chain helper.** Callers manage `start_fibonacci_chain`
  + `compile_fibonacci_step` + lifecycle `prove`/`extend` themselves.
  A `FibonacciChainBuilder`-style wrapper is the next ergonomics slice.
- **Smaller test param profile.** Promoting the ignored full-stack tests back
  into the default sweep requires either a soundness-preserving
  smaller-params test profile (`neo-params` addition) or the
  Spartan-compressed terminal proof landing. Both are out of scope here
  and need council sign-off.

## Open Questions

Unresolved protocol questions live in [`open-questions/`](open-questions/).
