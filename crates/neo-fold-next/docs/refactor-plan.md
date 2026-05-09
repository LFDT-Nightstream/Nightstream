# neo-fold-next Refactor Plan

This plan is the working contract for refactoring `crates/neo-fold-next/src`
so protocol ownership is visible from folder paths and important flows are
auditable top-down.

The filesystem should show both:

1. The SuperNeo protocol spine.
2. The frontend/runtime orchestration around that spine.

Do not let frontend folders become accidental owners of SuperNeo math.

## Global Rules

- Work in closed chunks.
- Do not mix unrelated folders except for required call-site/import updates.
- Separate mechanical moves from behavior cleanup.
- Do not add backwards-compatibility shims.
- Do not add new features, env vars, flags, traits, or speculative abstractions.
- Keep every file under 1,500 lines.
- Preserve public behavior unless a public surface is clearly accidental/internal.
- Do not weaken protocol checks.
- Digests are compression, not authority.
- Use Poseidon2-only hashing in proof/public digest paths unless explicitly
  approved otherwise.
- Before moving code into `neo-fold-next/core/superneo`, check whether the
  concept already belongs to `neo_math`, `neo_params`, `neo_ccs`,
  `neo_reductions`, or `neo_ajtai`. If it does, `neo-fold-next` may wrap or
  call that crate, but must not redefine the concept locally.

## Existing Crate Ownership

`neo-fold-next` is an integrator/orchestrator crate for the next proof surface.
It should not duplicate protocol semantics already owned elsewhere.

| Crate | Should Own |
|---|---|
| `neo_math` | Fields, rings, coefficient/ring math. |
| `neo_params` | Parameter sets, dimensions, bounds, parameter validation. |
| `neo_ccs` | CCS/CE relation data shapes, witnesses, matrices. |
| `neo_reductions` | Pi_CCS, Pi_RLC, Pi_DEC native reduction logic, folding engines. |
| `neo_ajtai` | Ajtai commitments and module commitment behavior. |
| `neo-fold-next` | Composition glue, frontend lowering, public proof boundaries, recursive/circuit integration, Spartan/decider handoff. |

## Protocol Ownership Buckets

Every file should be classified with exactly one primary owner.

| Bucket | Owns | Must Not Own |
|---|---|---|
| `embedding` | Field-vector to ring-vector coefficient embedding. | Commitments, transcripts, frontend lowering. |
| `transform` | Inner-product/matrix transform, constant-term evaluation. | Frontend matrix normalization. |
| `relation` | CCS/CE relation shapes, instances, witnesses, openings. | Prover flow, circuit allocation. |
| `pi_ccs` | SuperNeo Pi_CCS reduction: CCS/eval/norm checks, sum-check plumbing. | RLC or DEC logic. |
| `pi_rlc` | Random linear combination, challenge use, norm-growth accounting. | Sum-check polynomial construction or DEC split. |
| `pi_dec` | Base-b decomposition, recomposition, norm reset. | CCS checks or RLC accumulation. |
| `compose` | Mathematical composition: Pi_DEC after Pi_RLC after Pi_CCS. | Internal proof math of any individual reduction. |
| `orchestrator` | Runtime flow: validate, build relation input, call composed protocol, update state/public image, hand off proof. | Protocol internals, commitment internals, circuit constraints. |
| `transcript` | Domain-separated challenge/public digest layout. | Relation math or witness mutation. |
| `commitment/opening` | Ajtai/module commitment interfaces, binding assumptions, low-norm openings. | Proof serialization or frontend-specific state. |
| `public_io` | Public accumulator serialization, public digest inputs, verifier-facing encodings. | Private witnesses/openings. |
| `circuit` | Bellpepper/Spartan variables, constraints, recursive verifier gadgets. | Native prover-only helpers. |
| `frontend_adapter` | Direct CCS/RV32IM lowering into SuperNeo-compatible inputs. | SuperNeo core semantics. |
| `proof_boundary` | Published proof structs, verifier entrypoints, per-frontend adapters. | Internal witnesses/openings/perf data. |
| `measurement/perf` | Timing, shape, diagnostic counters. | Protocol decisions. |
| `mechanical_helper` | Small non-protocol helpers. | Anything that encodes an invariant. |

Important distinction:

- `compose` owns protocol order.
- `pi_ccs`, `pi_rlc`, and `pi_dec` own reduction math.
- `orchestrator` owns runtime/session flow around the protocol.
- Frontends may orchestrate frontend-specific flow, but must delegate SuperNeo
  semantics to the owning modules.

## Step-Down Flow Rule

Important public functions should read top-down:

```rust
validate_inputs()?;
build_relation_input()?;
run_superneo_composition()?;
update_state()?;
prove_or_verify_boundary()?;
```

Each helper may decompose one level further, but the top-level function should
tell the story without forcing the reader through local plumbing first.

## Folder Ownership Rules

- Folder paths must encode ownership.
- Avoid needing a map to explain structure.
- Avoid vague `utils`/`helpers` unless the code is truly mechanical.
- Avoid file/folder twins such as `recursive.rs` plus `recursive/`; prefer
  `recursive/mod.rs`.
- Avoid wrapper theater: delete pass-through functions/modules that add no
  semantic meaning.
- Avoid giant flat structs; group by domain.
- Pure initialization belongs on the initialized type:
  - `Type::new`
  - `Type::from_parts`
  - `Type::from_advice`
  - `Type::from_default_boundary`
- Free `build_*` functions are acceptable only for protocol operations
  spanning multiple owners.
- Do not expose a public function that only calls `*_with_perf` and discards
  perf.
- Perf structs should only exist if consumed by diagnostics/probes or returned
  by an explicit measurement API.
- Avoid giant perf/type bags. A struct with many same-kind timing/count fields
  must be decomposed by protocol phase or ownership domain. Prefer nested
  summaries such as `BuildPerf { accepted_artifact, final_statement, seam,
  total_ms }`, where `final_statement` and `seam` have their own phase-owned
  structs.
- As a rule of thumb, a flat struct with more than 8-12 public scalar fields is
  a refactor smell unless it is a stable external serialization format. Split
  it by domain before adding more fields.
- Perf field names should describe local phase ownership, not encode an entire
  call stack in one identifier. Avoid names like
  `final_statement_recursive_ccs_sample_challenges_ms`; use nested structs:
  `perf.final_statement.recursive.ccs.sample_challenges_ms`.
- Public proof/perf structs must not expose internal phase timing unless that
  timing is intentionally part of a diagnostics API. Diagnostic timing types
  should live under measurement/perf ownership, not protocol-flow modules.

## SuperNeo Invariants

- Native witness is a field vector of length `n_F = d * n_R`.
- Committed object is a ring vector of length `n_R`.
- Coefficient embedding is the only path from field witness chunks into ring
  openings.
- Norm checking is over underlying field coefficients.
- Sum-check and norm-check logic remain field-native over `F` or extension
  field `K`.
- Pi_CCS owns CCS, norm, and prior-evaluation checks.
- Pi_RLC owns random linear combination and norm-growth accounting.
- Pi_DEC owns base-b decomposition from `CE(B)` back to `CE(b)^k`.
- Ajtai commitments are treated as module homomorphisms.
- Binding and sampling assumptions are explicit.
- Public digests never become authority.
- No frontend may redefine SuperNeo accumulator semantics.

## Naming Rules

Use protocol names for protocol operations:

- `pi_ccs`, not generic `fold_step`.
- `pi_rlc`, not generic `combine` or `merge`.
- `pi_dec`, not generic `decompose_claim` outside DEC helpers.
- `ce`, not `eval_claim` where it refers to the formal CE relation.
- `ccs`, not vague `constraint_claim`.
- `embedding`, not `packing`, when it is coefficient embedding.
- `transform`, not `encoding`, when it is the matrix/inner-product transform.
- `opening`, not `witness`, for committed ring openings.
- `witness`, not `opening`, for field witnesses.
- `compose` for mathematical reduction composition.
- `orchestrator`, `state`, or `session` for runtime flow managers.

## Red Flags

- File owns both frontend lowering and SuperNeo relation logic.
- File owns both witness/opening data and public proof serialization.
- File derives transcript challenges outside transcript ownership.
- File duplicates parameter constants.
- File performs ring operations inside field-native sum-check logic.
- File treats digest/hash as authority.
- File has no explicit norm-bound check where one is required.
- File combines Pi_CCS, Pi_RLC, and Pi_DEC internals in one function.
- File returns perf data ignored by all callers.
- File adds a giant flat perf/count struct instead of nested phase summaries.
- Field names encode a whole call stack because the data model lacks phase
  structure.
- Public function only forwards to another function.

## Process

### Persistent Checklist

Maintain `crates/neo-fold-next/docs/refactor-checklist.md` throughout the
refactor.

- Create the checklist before starting Pass 0 if it does not exist.
- Update it at the end of every closed chunk.
- The checklist is the source of truth for progress; do not rely on chat
  history.
- Keep entries concise and factual. Do not include aspirational progress
  language.
- Each row should include ownership bucket, status, remaining risk, and the
  latest verification command.

Suggested table:

| Area | Bucket | Status | Done | Remaining | Risk | Latest Verification |
|---|---|---|---|---|---|---|
| `frontends/direct_ccs/state` | `orchestrator` | `in progress` | Split init/append/compress. | Move pure constructors into owned types. | Medium | `cargo check -p neo-fold-next` |

Allowed statuses:

- `not started`
- `mapped`
- `in progress`
- `blocked`
- `done`

Required checklist structure:

```markdown
# neo-fold-next Refactor Checklist

## Current Focus

| Field | Value |
|---|---|
| Active area | `frontends/direct_ccs` |
| Active chunk | `recursive module structure` |
| Status | `in progress` |
| Latest verification | `cargo check -p neo-fold-next` |
| Next action | `Finish recursive/mod.rs split and rerun direct-CCS tests.` |

## Progress By Area

| Area | Bucket | Status | Done | Remaining | Risk | Latest Verification |
|---|---|---|---|---|---|---|
| `frontends/direct_ccs/state` | `orchestrator` | `in progress` | Split init/append/compress. | Move pure constructors into owned types. | Medium | `cargo check -p neo-fold-next` |

## Protocol Ownership Map

| File/Folder | Bucket | Current Responsibility | Problem | Destination/Owner | Risk | Red Flags |
|---|---|---|---|---|---|---|
| `frontends/direct_ccs/f_prime` | `frontend_adapter` | Direct F' source and verifier-body surfaces. | Needs clearer source/R1CS/verifier split. | `frontends/direct_ccs/f_prime/*` | Medium | None |

## Verification Log

| Date | Command | Result | Notes |
|---|---|---|---|
| `YYYY-MM-DD` | `cargo check -p neo-fold-next` | Pass | After direct-CCS state split. |

## Open Risks

| Risk | Area | Impact | Next Action |
|---|---|---|---|
| Dirty worktree spans several folders. | repo | Harder to review/commit. | Commit closed chunks before broad edits. |

## Decisions

| Decision | Reason | Date |
|---|---|---|
| Treat `neo-fold-next` as integrator, not SuperNeo math owner. | Avoid duplicating `neo_reductions`, `neo_ccs`, `neo_math`, `neo_params`, and `neo_ajtai`. | `YYYY-MM-DD` |
```

Rules for the checklist:

- The `Current Focus` section must always describe the exact active chunk.
- `Progress By Area` tracks implementation/refactor progress.
- `Protocol Ownership Map` tracks Pass 0 classification and red flags.
- `Verification Log` records only commands that were actually run.
- `Open Risks` should be concrete and actionable.
- `Decisions` records architecture choices that would otherwise be lost in chat.

### Pass 0: Protocol Ownership Map

Do not move files during Pass 0.

- Identify files that define or mutate SuperNeo concepts.
- Classify each file by bucket.
- Mark frontend-owned vs core/protocol-owned vs circuit-owned vs
  proof-boundary-owned.
- Identify accidental public APIs and wrappers.

Before moving files in a folder, answer:

1. Which SuperNeo bucket does this code belong to?
2. Is this native protocol, recursive circuit, public proof boundary, or
   frontend adapter?
3. Does it touch witness/opening data?
4. Does it derive transcript challenges?
5. Does it enforce a norm bound?
6. Does it assume a parameter relation such as `n_F = d * n_R` or `B = b^k`?
7. Does it implement one of Pi_CCS, Pi_RLC, Pi_DEC, or composition?
8. Could this code be reviewed independently by comparing it to the
   corresponding reduction?

### Pass 1: Highest-Value Folder Cleanup

Recommended order:

1. `frontends/direct_ccs`
2. `core`
3. `frontends/rv32im`
4. `circuit`
5. `public_proof`
6. `decider`
7. `bin`
8. `vm`

For each folder:

1. Inspect files.
2. Produce an audit table:
   - file/folder,
   - current responsibility,
   - bucket,
   - problem,
   - proposed destination/name,
   - risk,
   - red flags.
3. Pick one closed chunk.
4. State assumptions and success criteria.
5. Do mechanical moves first.
6. Compile.
7. Clean step-down flow and constructors.
8. Run format, compile, and focused tests.
9. Report results.

## Target Ownership Model

```text
src/
  lib.rs
    Small crate surface. No giant umbrella exports.

  core/
    Shared protocol and proof plumbing.
    Should not contain frontend-specific lowering.

    superneo/ or adapters to existing SuperNeo-owning crates
      Only if this crate truly owns missing SuperNeo protocol surfaces.
      Do not duplicate neo_math, neo_ccs, neo_reductions, or neo_params.

      relation/
      reductions/
        pi_ccs/
        pi_rlc/
        pi_dec/
        compose.rs
      transcript/
      public_io/
      commitments/

    construction2/
      Construction-2 state, public images, terminal helpers.

    finalize/
      Finalization/public image digest logic.

    opening/
      Opening/time-opening/convergence surfaces.

    proof/
      Shared proof data shapes.

    prover.rs
    verifier.rs
    session/

  circuit/
    superneo/
      Circuit verifier gadgets mirroring protocol reduction ownership.

    claim/
    nifs/
    transcript/
    witness/

  frontends/
    direct_ccs/
      First-class arbitrary CCS/R1CS path.
      Owns lowering/orchestration, not SuperNeo core semantics.

      adapter/
      relation_builders/
      state/
      f_prime/
      terminal/
      recursive/

    rv32im/
      First-class RV32IM path.
      Owns machine lowering/orchestration, not SuperNeo core semantics.

      chunk/
      kernel/
      main_relation/
      f_prime/
      recursion/
      audit/

  public_proof/
    Published proof boundary.
    Must not contain witnesses/openings or duplicate protocol checks.

    direct_ccs/
    rv32im/

  decider/
    Spartan/decider wrappers and verifier/prover keys.
    No SuperNeo relation definitions.

  vm/
    Generic VM traits/specs only.

  bin/
    Thin diagnostics/probe entrypoints only.
```

## Verification

Always run:

```bash
cargo fmt --all
cargo check -p neo-fold-next
```

For Direct CCS chunks:

```bash
cargo test -p neo-fold-next --release --test direct_ccs_ivc --test direct_ccs_r1cs_low_norm --test direct_ccs_r1cs_export --test direct_ccs_redteam -- --nocapture
```

For RV32IM chunks:

- Run focused RV32IM tests that cover changed surfaces.
- Do not run long perf tests unless explicitly approved.

## Stop Condition Per Folder

- Every file has one primary bucket.
- Every public type/function has a reason to be public.
- Folder paths encode ownership.
- Important flows read top-down.
- No obvious wrappers/shims remain.
- Pure initialization is type-owned.
- Perf APIs are explicit measurement surfaces.
- Perf/count data is grouped by phase; no giant flat bags of timing fields.
- No frontend defines SuperNeo core protocol semantics.
- No public proof folder contains witness/opening data.
- No circuit folder duplicates native prover-only logic.
- Parameter constants are not duplicated.
- No file exceeds 1,500 lines.
- Focused tests pass.

## Completion Criterion

A reviewer can audit SuperNeo protocol ownership before frontend behavior:

1. Relation inputs/outputs.
2. Pi_CCS.
3. Pi_RLC.
4. Pi_DEC.
5. Composition/orchestrator.
6. Transcript/public IO.
7. Proof boundary.
8. Frontend adapters.

## Reporting

After each folder, report:

| File/Folder | Change | Why | Verification |
|---|---|---|---|

After all folders, report:

| Top-level area | Before | After | Remaining risk |
|---|---|---|---|
