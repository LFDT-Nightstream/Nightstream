# Direct CCS / SuperNeo Step Refactor Plan

This plan is the working contract for refactoring the Direct CCS path and the
shared SuperNeo chunk step that Direct CCS depends on in `crates/neo-fold-prototype`.

The current priority is not another global crate reshuffle. The mandatory next
milestone is a real, canonical paper-order step owner. The code must make the
Direct CCS proof flow readable from the public entry point down to the SuperNeo
reductions, including the missing pre-`Pi_CCS` construction phase:

```text
raw/direct CCS step
  -> validate direct CCS shape and witness
  -> embed field witness into the SuperNeo committed object
  -> derive public/input projection X
  -> commit embedded witness
  -> build fresh CcsClaim + CcsWitness
  -> Pi_CCS
  -> Pi_RLC
  -> Pi_DEC
  -> reuse exact private DEC children
  -> advance Construction-2 / F' authority
```

The auditor should not have to infer where embedding, commitment, or fresh CCS
claim construction happened before `Pi_CCS`.

The auditor also should not have to chase a long chain of wrappers to see one
SuperNeo step. There must be one canonical function whose body reads, in order:

```text
prepare fresh + carried claims
Pi_CCS
Pi_RLC
Pi_DEC
derive next Construction-2/public image material
advance Direct CCS state
```

This requirement is not satisfied by adding a wrapper around the current long
functions. If the current file/function structure prevents this shape, refactor
the deeper owner until this shape is the actual code path.

## Scope

In scope:

- `crates/neo-fold-prototype/src/frontends/direct_ccs/**`
- `crates/neo-fold-prototype/src/core/chunk_folding/**`, but only to make the
  canonical paper-order SuperNeo step real.
- Direct CCS tests under `crates/neo-fold-prototype/tests/direct_ccs*`
- Direct CCS docs/checklists under `crates/neo-fold-prototype/docs`

Out of scope:

- RV32IM.
- Broad `core` reorganization outside `core/chunk_folding`.
- Public API redesign in `crates/neo-fold-prototype/src/lib.rs`.
- Lifecycle trait/API redesign in `crates/neo-fold-prototype/src/lifecycle`.
- New features, environment variables, flags, or speculative abstractions.

Treat the current `lib.rs` and `src/lifecycle` work as fixed. If the Direct CCS
cleanup truly requires changing them, stop and explain why before editing.

## Non-Negotiable Rules

- Keep the API simple.
- Build one canonical paper-order step path; do not keep multiple wrapper
  variants for the same proof step.
- Do not add code that only calls another function and contributes no meaning.
- Do not add backwards-compatibility shims.
- Do not move complexity into new names.
- Do not create traits, wrappers, builders, or config objects for one use case.
- Keep every file under 1,500 lines.
- Preserve behavior unless a surface is clearly accidental/test-facing.
- Do not weaken protocol checks.
- Digests are compression, not authority.
- Use Poseidon2-only hashing in proof/public digest paths unless explicitly
  approved otherwise.
- Do not expose test-facing Direct CCS internals as public API.

## Rust Quality Bar

Before each meaningful code chunk, check the relevant guidance in
`external/rust-patterns`.

Use these parts especially:

- `src/idioms/ctor.md`: pure initialization belongs on the type when it is
  really a constructor.
- `src/idioms/coercion-arguments.md`: prefer borrowed arguments when ownership
  is not required.
- `src/patterns/structural/compose-structs.md`: split giant flat structs into
  domain-owned substructures.
- `src/patterns/structural/trait-for-bounds.md`: avoid custom traits just to
  hide ugly bounds.
- `src/patterns/behavioural/newtype.md`: use newtypes only when they remove
  real ambiguity.
- `src/anti_patterns/borrow_clone.md`: do not clone just to silence ownership
  problems.

Legacy code in `tmp/removed-deprecated-neo-fold/src` may be used only as a
source of ideas for flow shape or missing concepts. Do not copy its structure,
naming, wrappers, broad modules, or public surfaces. It was deprecated because
it was not good enough.

## Protocol Grounding

Before changing Direct CCS embedding, fresh claim construction, or folding flow,
re-read the relevant SuperNeo paper sections:

- `docs/superneo-paper/05_5_Embedding_products_with_evaluation_homomorphism.md`
- `docs/superneo-paper/06_6_Strong_and_weak_interactive_reductions.md`
- `docs/superneo-paper/07_7_Neo_s_folding_scheme_for_CCS.md`

Use them to keep the implementation aligned with the actual protocol flow:

- Section 5 grounds the embedding/evaluation-homomorphism layer before `Pi_CCS`.
- Section 6 grounds why `Pi_CCS` is strong and `Pi_RLC` is weak.
- Section 7 grounds the composed CCS folding path:
  `Pi_CCS -> Pi_RLC -> Pi_DEC`.

If a planned change makes the code easier to read but obscures one of those
protocol boundaries, do not make that change.

## Preferred Flow Style

Important functions should read as a protocol story:

```rust
pub fn prove_direct_ccs(...) -> Result<DirectCcsProof, Error> {
    let mut proof = start_direct_ccs_proof(...)?;

    for step in steps {
        proof = prove_one_direct_ccs_step(...)?;
    }

    Ok(proof)
}

fn prove_one_direct_ccs_step(...) -> Result<DirectCcsProof, Error> {
    let fresh_step = build_fresh_superneo_step(...)?;
    let folded = fold_fresh_step_with_superneo(...)?;
    let proof = advance_direct_ccs_authority(...)?;

    Ok(proof)
}
```

Each helper may decompose one level further. The top-level reader should see the
important protocol sequence before seeing local plumbing.

Avoid this shape:

```rust
pub fn important_flow(...) -> Result<_, _> {
    helper_that_only_calls_another_helper(...)
}
```

If a helper exists, it must name a real protocol step, ownership boundary, or
data transformation.

## Mandatory Paper-Order Step Shape

The next implementation target is a real SuperNeo/Construction-2 step shape,
not another local cleanup pass.

The shared SuperNeo chunk owner should expose one canonical function under
`core/chunk_folding` with this shape:

```rust
pub(crate) fn prove_superneo_chunk_step(...) -> Result<SuperNeoChunkStep, Error> {
    let prepared = prepare_fresh_and_carried_claims(...)?;
    let pi_ccs = pi_ccs::prove(...)?;
    let pi_rlc = pi_rlc::prove(...)?;
    let pi_dec = pi_dec::prove(...)?;

    Ok(SuperNeoChunkStep::from_parts(prepared, pi_ccs, pi_rlc, pi_dec))
}
```

The Direct CCS append owner should then expose one Construction-2-facing step:

```rust
fn append_construction2_step(...) -> Result<DirectCcsIvcState, Error> {
    let superneo = prove_superneo_chunk_step(...)?;
    let terminal_replay = build_terminal_replay_surface(&superneo)?;
    let next_u_i = derive_next_construction2_u_i(&superneo)?;
    let next_x_i = compute_construction2_public_image(..., &next_u_i)?;

    self.advance_with(superneo, terminal_replay, next_u_i, next_x_i)
}
```

The exact names may change, but the shape may not. The implementation must
make these phases visible in the body of the real hot path:

| Phase | Required ownership |
|---|---|
| Prepare fresh + carried claims | Shared chunk folding owner. |
| `Pi_CCS` | Existing `neo_reductions`/shared folding owner; Direct CCS must not redefine the math. |
| `Pi_RLC` | Existing reduction owner, called from the paper-order step. |
| `Pi_DEC` | Existing reduction owner, called from the paper-order step. |
| Terminal replay surface | Direct CCS state/terminal owner. |
| Construction-2 next image | Direct CCS Construction-2 state owner. |
| State advance | Direct CCS state owner. |

The refactor must delete or collapse `_with_perf`, `_with_trace`,
`_with_instance_digest`, and `_with_handle` variants when they only thread
parameters through the same step. Perf and trace data should be returned as
fields on the canonical result when they are needed.

Do not create `neo-fold-prototype/src/paper/pi_ccs.rs`, `pi_rlc.rs`, or `pi_dec.rs`
that reimplement or shadow the reduction math. `neo-fold-prototype` is the
orchestrator. The reduction math remains owned by `neo_reductions` and the
shared SuperNeo crates.

## Direct CCS Ownership Model

Target layout:

```text
frontends/direct_ccs/
  mod.rs
    Small Direct CCS module surface.

  program/
    User-facing Direct CCS program and preprocessing data.
    Owns shape validation and stable program metadata.

  step/
    Raw Direct CCS step input and fresh SuperNeo step construction.
    Owns the pre-Pi_CCS path:
      validate -> embed -> project X -> commit -> build CcsClaim/CcsWitness.

  state/
    Direct CCS IVC state evolution.
    Owns carry state, accumulator digests, and append-state transitions.
    Does not own raw step embedding.

  superneo/
    Direct CCS orchestration around existing SuperNeo reductions.
    Owns the call order:
      Pi_CCS -> Pi_RLC -> Pi_DEC.
    Does not redefine reduction math from neo_reductions.

  f_prime/
    Direct CCS F' source, compact image, verifier body, and prior authority.
    Owns Construction-2/F' authority surfaces for Direct CCS only.

  spartan/
    Direct CCS finish/compression handoff.
    Owns Spartan-facing proof completion.

  verify/
    Direct CCS verifier-facing checks.
    Owns no-Spartan and finished-with-Spartan verification paths.

  audit/
    Explicit audit/red-team helpers only.
    Nothing here should be needed by normal proof flow.
```

This layout is a target. Move one closed chunk at a time; do not scatter files
just to match the tree.

## Required Direct CCS Public Flow

The Direct CCS path should support these client-level operations through the
existing public API/lifecycle surface:

| Operation | Meaning |
|---|---|
| `prove` | Build an uncompressed/incremental Direct CCS proof over steps. |
| `extend` | Add one more Direct CCS step to an existing proof. |
| `finish_with_spartan` | Compress/finish an existing Direct CCS proof with Spartan. |
| `prove_and_finish_with_spartan` | Prove the steps and finish with Spartan in one call. |
| `verify` | Verify the uncompressed/incremental Direct CCS proof. |
| `verify_finished_with_spartan` | Verify the finished Spartan proof. |

Do not rename the existing `lib.rs` or `lifecycle` entry points while carrying
out this Direct CCS internal cleanup.

## Fresh SuperNeo Step Construction

This is the missing pre-`Pi_CCS` phase and remains a required ownership
boundary. It is already mostly mapped; until the canonical paper-order step is
real, that step has priority over additional fresh-step cleanup.

The code should make this phase explicit:

```rust
fn build_fresh_superneo_step(...) -> Result<FreshSuperNeoStep, Error> {
    validate_direct_ccs_step(...)?;
    let embedded_witness = embed_direct_ccs_witness(...)?;
    let public_input = derive_public_input_projection(...)?;
    let commitment = commit_embedded_witness(...)?;
    Ok(build_ccs_claim_and_witness(...)?)
}
```

Responsibilities:

| Step | Owns | Must Not Own |
|---|---|---|
| `validate_direct_ccs_step` | Direct CCS shape, public input length, witness dimensions, parameter compatibility. | Pi_CCS/RLC/DEC reduction logic. |
| `embed_direct_ccs_witness` | Field-vector to SuperNeo committed-object representation. | Commitment or transcript challenge derivation. |
| `derive_public_input_projection` | Computing/checking the `X`/public projection from the witness layout. | Folding or DEC child reuse. |
| `commit_embedded_witness` | Calling the Ajtai/module commitment surface. | Defining Ajtai semantics locally. |
| `build_ccs_claim_and_witness` | Producing the fresh `CcsClaim` and `CcsWitness` consumed by folding. | Running SuperNeo reductions. |

If `DirectCcsStep` already contains a prepared `CcsClaim` and `CcsWitness`, the
constructor must make that fact obvious. A reader should be able to follow how
those values were produced and why they match the raw Direct CCS step.

## SuperNeo Folding Boundary

After fresh step construction, the real folding code path must read in this
order:

```text
prepare fresh + carried claims
run Pi_CCS
run Pi_RLC
run Pi_DEC
check DEC recomposition and CE membership
reuse exact private DEC children as the next carry
update Direct CCS state/public image
record F' authority material
```

The Direct CCS frontend may orchestrate Construction-2 state and terminal replay
surface work, but it must not duplicate or redefine the math owned by
`neo_reductions`, `neo_ccs`, `neo_math`, `neo_params`, or `neo_ajtai`.

## Red Flags

- `DirectCcsStep` appears without a visible construction path from raw witness
  to embedded SuperNeo claim.
- A public function only forwards to another function.
- A helper exists only because the caller wanted fewer lines.
- A type name hides whether the value is raw, embedded, committed, carried, or
  Spartan-finished.
- A file owns both raw Direct CCS lowering and Pi_DEC/F' authority logic.
- A file returns perf data ignored by all callers.
- A flat struct has more than 8-12 public scalar fields and is not a stable
  external serialization format.
- A digest is treated as proof authority.
- The flow reaches `Pi_CCS` without showing where the fresh CCS claim and
  witness came from.
- The code claims to follow the paper-order step but the actual hot path still
  jumps through wrapper variants before reaching `Pi_CCS`, `Pi_RLC`, or
  `Pi_DEC`.

## Process

For each closed chunk:

1. State the Direct CCS ownership problem being fixed.
2. State assumptions and success criteria.
3. Inspect the current files before editing.
4. Make the smallest coherent change.
5. Prefer moving code into ownership-aligned files over adding wrappers.
6. Prefer constructors on types for pure initialization.
7. Keep protocol operations as clear free functions when they span owners.
8. Run formatting and focused verification.
9. Update the checklist.

## Goal-Loop Execution

This plan is intended to be safe for a Codex goal loop, but only if the loop
works in small closed chunks and records state in the checklist.

The goal prompt should be short and point here:

```text
Follow crates/neo-fold-prototype/docs/refactor-plan.md and keep
crates/neo-fold-prototype/docs/refactor-checklist.md updated after every closed
chunk.
```

Do not depend on chat memory. The checklist is the durable state for the loop.
If the checklist and the current code disagree, trust the code and repair the
checklist before continuing.

### Resume Contract

Every resumed goal iteration must start by establishing local state before
editing:

1. Read the plan.
2. Read the checklist.
3. Check the worktree status.
4. Identify which dirty files are inside the active Direct CCS chunk and which
   are unrelated existing work.

Do not revert, reformat, or reorganize unrelated dirty files. If the checklist
shows an `in progress` chunk, either finish that exact chunk, mark it `blocked`
with a concrete reason, or repair the checklist if the code shows it was
already finished. Do not start a new area while an older chunk is ambiguous.

The loop is allowed to continue without asking only when the next action is
specific enough to begin from disk state alone. If the next action is broad,
stale, or contradicts the code, the first chunk is checklist repair, not code
movement.

Each loop iteration must do exactly one coherent Direct CCS or
`core/chunk_folding` chunk:

1. Read `crates/neo-fold-prototype/docs/refactor-plan.md`.
2. Read `crates/neo-fold-prototype/docs/refactor-checklist.md`.
3. Check `git status --short` and keep unrelated dirty files untouched.
4. Re-read the SuperNeo paper section relevant to the active chunk:
   - embedding/fresh claim work: section 5,
   - reduction-boundary work: sections 6 and 7,
   - DEC child reuse/F' authority work: section 7.
5. Inspect the files for the active Direct CCS or `core/chunk_folding` area
   before editing.
6. Make one ownership-aligned implementation/refactor change.
7. Avoid `src/lib.rs` and `src/lifecycle` unless the user explicitly approves
   changing them.
8. Run `cargo fmt --all`.
9. Run `cargo check -p neo-fold-prototype`.
10. Run the smallest focused Direct CCS test that covers the changed behavior,
   unless it would exceed the approved time limit.
11. Check that touched files remain below 1,500 lines.
12. Update the checklist with factual progress, verification, remaining risk,
    and the next action.

### Picking The Next Chunk

Until the mandatory paper-order step exists, choose that work before any lower
priority. If the checklist does not identify a concrete next action, inspect
only `frontends/direct_ccs` and `core/chunk_folding`, then choose the first
unresolved item in this order:

1. Canonical paper-order step:
   shared `core/chunk_folding` function for `prepare -> Pi_CCS -> Pi_RLC ->
   Pi_DEC`, plus the Direct CCS Construction-2 append function that consumes it.
2. Fresh step construction:
   raw step -> embedding -> public projection -> commitment -> `CcsClaim` /
   `CcsWitness`.
3. Direct CCS append/state flow:
   carried children, accumulator digest, and exact private child reuse.
4. SuperNeo folding boundary:
   paper-facing order `Pi_CCS -> Pi_RLC -> Pi_DEC` without local math
   duplication.
5. Recursive/F' authority:
   prior authority, Construction-2 reachability, and non-digest-only verifier
   evidence.
6. Spartan finish/compression:
   terminal handoff and finished proof packaging.
7. Verification surfaces:
   no-Spartan verify and finished-with-Spartan verify.
8. Direct CCS tests/red-teams:
   tests that exercise the exact boundary changed by the previous chunks.
9. Documentation/checklist cleanup:
   only when it records actual implementation state or removes stale guidance.

Skip any item that is already `done` in the checklist and still matches the
code. Do not start a broad folder reshuffle just because the target layout says
a folder could eventually exist.

### Autonomous Work Allowed

The loop may do these without asking:

- Rename private/internal Direct CCS helpers when the new name is shorter and
  more precise.
- Move Direct CCS code into ownership-aligned files under
  `frontends/direct_ccs`.
- Move `DirectCcsStep`-adjacent construction logic into the Direct CCS step
  owner.
- Extract named packages from tuple-heavy internal Direct CCS code when the
  package names a real protocol/result boundary.
- Add or update Direct CCS tests that cover the changed behavior.
- Refactor shared `core/chunk_folding` when needed to make the canonical
  paper-order step the real code path.

The loop must ask before doing these:

- Editing `src/lib.rs` or `src/lifecycle`.
- Changing public lifecycle names or user-facing API shape.
- Moving RV32IM code.
- Broad shared `core` outside `core/chunk_folding`, or broad `circuit`
  reorganization.
- Adding traits, feature flags, environment variables, builders, config
  objects, compatibility modules, or deprecated aliases.
- Changing protocol-binding transcript/digest behavior.
- Replacing existing SuperNeo reduction owners with frontend-local math.

The loop must stop and ask for direction if:

- The next useful step requires changing `src/lib.rs` or `src/lifecycle`.
- The change would touch RV32IM or shared `core` outside the
  `core/chunk_folding` paper-order step.
- The implementation requires a new feature flag, environment variable, or
  speculative abstraction.
- A protocol boundary is unclear after re-reading the paper sections.
- A focused verification command fails for a non-obvious reason.
- A file would exceed 1,500 lines.
- The loop would add a public wrapper, helper, trait, or type whose only job is
  to call or rename another thing.

### Advancing The Checklist

Do not mark a chunk `done` until all of this is true:

- The changed code compiles.
- `cargo fmt --all` ran after the last Rust edit.
- At least one focused Direct CCS check or test is logged, unless the chunk is
  documentation-only.
- The checklist names remaining risk honestly.
- The next action is concrete enough for another goal iteration to start
  without reading chat history.

Use `blocked`, not `done`, when the next meaningful step requires a user
decision or an out-of-scope edit.

The loop must not claim progress from code movement alone. Progress only counts
when one of these improves:

- The raw Direct CCS step to fresh SuperNeo claim path is clearer.
- The embedding/projection/commitment boundary before `Pi_CCS` is clearer.
- The `Pi_CCS -> Pi_RLC -> Pi_DEC` folding boundary is clearer.
- The canonical paper-order step becomes the real hot path rather than a
  wrapper or comment.
- Exact private DEC child reuse is clearer.
- F' authority is clearer and remains non-digest-only.
- The public Direct CCS flow is easier to read top-down without wrappers.

If an iteration only renames or moves code, the checklist must explain which
reader-visible ownership problem was removed. If it cannot explain that, the
change should not be made.

### Per-Iteration Completion Message

At the end of each goal iteration, report only the facts needed to resume:

- files changed in this chunk,
- behavior/protocol boundary made clearer,
- verification command and result,
- remaining risk,
- next concrete action from the checklist.

Do not report broad percentages unless the user asks for them. Do not claim the
Direct CCS implementation is fixed unless the stop condition below is actually
satisfied.

## Checklist

Maintain `crates/neo-fold-prototype/docs/refactor-checklist.md`.

Required structure:

```markdown
# Direct CCS Refactor Checklist

## Current Focus

| Field | Value |
|---|---|
| Active area | `core/chunk_folding` + `frontends/direct_ccs/state` |
| Active chunk | `canonical paper-order SuperNeo/Construction-2 step` |
| Status | `mapped` |
| Latest verification | `documentation-only update` |
| Next action | `Introduce the real paper-order step path and make Direct CCS append consume it.` |

## Progress By Area

| Area | Owner | Status | Done | Remaining | Risk | Latest Verification |
|---|---|---|---|---|---|---|
| `core/chunk_folding` + `frontends/direct_ccs/state` | Canonical paper-order step | `mapped` | Required shape is specified. | Make it the real hot path: prepare -> Pi_CCS -> Pi_RLC -> Pi_DEC -> Construction-2 advance. | High | `documentation-only update` |

## Direct CCS Flow Map

| Phase | File/Folder | Status | Notes |
|---|---|---|---|
| Public entry | `src/lib.rs`, `src/lifecycle` | frozen | Do not edit without approval. |
| Raw step validation | `frontends/direct_ccs/step` | not started | Must be explicit. |
| Embedding | `frontends/direct_ccs/step` | not started | Must happen before Pi_CCS. |
| Commitment | `frontends/direct_ccs/step` | not started | Calls Ajtai owner. |
| Fresh CCS claim | `frontends/direct_ccs/step` | not started | Produces `CcsClaim`/`CcsWitness`. |
| Pi_CCS/Pi_RLC/Pi_DEC | `core/chunk_folding` | mapped | Must be one canonical paper-order step, not wrapper variants. |
| Construction-2 step advance | `frontends/direct_ccs/state` | mapped | Consumes the canonical SuperNeo step and advances public image/state. |
| F' authority | `frontends/direct_ccs/f_prime` | mapped | Must not rely on digest-only authority. |
| Spartan finish | `frontends/direct_ccs/spartan` or current owner | mapped | Compression handoff only. |

## Verification Log

| Date | Command | Result | Notes |
|---|---|---|---|

## Open Risks

| Risk | Area | Impact | Next Action |
|---|---|---|---|

## Decisions

| Decision | Reason | Date |
|---|---|---|
```

Allowed statuses:

- `not started`
- `mapped`
- `in progress`
- `blocked`
- `done`

## Verification

After Rust changes:

```bash
cargo fmt --all
cargo check -p neo-fold-prototype
```

For Direct CCS behavior:

```bash
cargo test -p neo-fold-prototype --release --test direct_ccs_ivc -- --nocapture
cargo test -p neo-fold-prototype --release --test direct_ccs_r1cs_low_norm -- --nocapture
cargo test -p neo-fold-prototype --release --test direct_ccs_r1cs_export -- --nocapture
cargo test -p neo-fold-prototype --release --test direct_ccs_redteam -- --nocapture
```

Do not run long tests past the configured time limit unless explicitly
approved.

## Stop Condition

This Direct CCS refactor is complete when:

- The public Direct CCS proof flow reads top-down.
- `lib.rs` and `src/lifecycle` remain stable unless the user explicitly
  approves changes.
- The raw step to fresh SuperNeo claim path is visible and auditable.
- Embedding, public projection, commitment, and claim construction are explicit
  before `Pi_CCS`.
- Pi_CCS, Pi_RLC, and Pi_DEC are called through their existing protocol owners.
- Private DEC children are shown to be exact Pi_DEC outputs before reuse.
- Construction-2/F' authority remains proof/reachability authority, not
  digest-only structure.
- No public wrapper or helper exists without semantic value.
- No Direct CCS file exceeds 1,500 lines.
- Focused Direct CCS tests pass.
