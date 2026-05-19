# Chip8AuthenticatedTrace Spec

## Purpose

- **What it is**: The theorem-facing chunk-level closure from exact per-row
  authenticated CHIP-8 evidence to an authenticated execution-trace surface.
- **Key property**: `authenticatedChunkTraceBound_of_exactTrace`: if every row
  of a chunk carries exact staged evidence and state well-formedness at the
  correct trace index, and the chunk satisfies the simple-kernel input
  contract, then the resulting trace satisfies the exact chunk-local execution,
  continuity, and boundary laws required by the kernel design; strong temporal
  closure then consumes one explicit authenticated temporal-support bundle.
- **Key corollary**: `executionCorrect_of_exactTrace_and_support` shows that
  exact authenticated trace evidence plus the chunk-input contract and one
  authenticated temporal-support bundle satisfy `ExecutionCorrect` with no
  additional whole-trace seam left external to this owner.
- **Protocol role**: This is the owner between row-local
  `Chip8EvidenceCoverage` and the generic chunk-trace semantics in
  `Chip8ExecutionSemantics`. It consumes chunk-input facts from
  `Chip8ChunkInput`, first derives the explicit temporal trace surface,
  then consumes the protocol-native kernel temporal-closure bundle through
  `Chip8PcContinuityBridge`, `Chip8TemporalConsistency`, and
  `Chip8TraceLinkBoundary`, and does not hide stronger trace semantics behind a
  weaker boundary-only surface.

## Target Formulas

### Exact frame evidence

For one authenticated row-backed execution frame

$$
\mathrm{frame} = (\mathrm{dec}, \mathrm{pre}, \mathrm{post}, z),
$$

define:

$$
\mathrm{ExactFrameEvidence}(stepIdx, frame)
$$

to package:

- `ExactSemanticEvidenceCovered(...)` for that exact `(stepIdx, pre, post, dec, z)`
- `StateWellFormed(pre)`
- `StateWellFormed(post)`

This owner is intentionally explicit about state well-formedness because it is
required by the generic semantic closure and is not exported directly by
`Chip8EvidenceCoverage`.

The fixed `ONE` coordinate is not an extra hypothesis here. It is derived from
the exact authenticated Stage-3 row-binding surface:

$$
\mathrm{ExactFrameEvidence}(stepIdx, frame)
\Longrightarrow
\mathrm{wf}(frame.row).
$$

### Frame-level semantic closure

The first theorem target is:

$$
\mathrm{ExactFrameEvidence}(stepIdx, frame)
\Longrightarrow
\mathrm{ExecutionFrameBound}(rom,\sigma,frame).
$$

This closes the row-local gap from authenticated staged evidence to the generic
row-backed execution predicate from `Chip8ExecutionSemantics`.

### Exact trace evidence

Define:

$$
\mathrm{ExactTraceEvidenceFrom}(k, frames)
$$

to mean:

- the head frame has `stepIdx = k`
- if a next frame exists, the adjacent pair exports one exact Stage-2 support
  object proving register adjacency and RAM adjacency between the two frames
- if a next frame exists, the adjacent pair exports one exact Stage-3 support
  object proving the current continuity witness is tied both to the current
  row's `pcNext` and to the next row's `pc`
- the tail satisfies `ExactTraceEvidenceFrom(k+1, tail)`

and define:

$$
\mathrm{ExactTraceEvidence}(frames)
:=
\mathrm{ExactTraceEvidenceFrom}(0, frames).
$$

Let:

$$
\mathrm{traceOf}(frames)
$$

be the list of underlying execution frames.

### Trace-level closure

The trace owner proves:

$$
\mathrm{ExactTraceEvidence}(frames)
\Longrightarrow
\mathrm{Forall}(\mathrm{ExecutionFrameBound})(\mathrm{traceOf}(frames))
$$

and:

$$
\mathrm{ExactTraceEvidence}(frames)
\Longrightarrow
\mathrm{ContinuityTraceBound}(0,\mathrm{traceOf}(frames)).
$$

The continuity theorem uses the row-local Stage-3 continuity witness exported by
`Chip8EvidenceCoverage`, together with the exact row index tracked by
`ExactTraceEvidenceFrom`.

The strengthened exact-trace owner also proves the adjacent-frame support
contracts required by the stronger temporal closure:

$$
\mathrm{ExactTraceEvidence}(frames)
\Longrightarrow
\mathrm{RegisterAdjacentTraceBound}(\mathrm{traceOf}(frames))
$$

$$
\mathrm{ExactTraceEvidence}(frames)
\Longrightarrow
\mathrm{RamAdjacentTraceBound}(\mathrm{traceOf}(frames))
$$

$$
\mathrm{ExactTraceEvidence}(frames)
\Longrightarrow
\mathrm{PcAdjacentBridge}(\mathrm{traceOf}(frames)).
$$

This owner also proves the exact Stage-3 boundary-frame transfer lemmas that
are available from exact evidence plus trace indices:

$$
\mathrm{ExactTraceEvidence}(frame :: rest)
\Longrightarrow
\mathrm{StartBoundaryFrame}(frame)
$$

and

$$
\mathrm{ExactTraceEvidence}(prefix ++ [last])
\land
|prefix ++ [last]| = meta.semanticRows
\Longrightarrow
\mathrm{FinalBoundaryFrame}(last).
$$

The remaining head initial-state agreement and exact semantic-row count are
consumed from `Chip8ChunkInput`.

### Authenticated chunk-trace bundle

Define `AuthenticatedChunkTraceBound` to package exactly the chunk-local surface
required by the strong kernel design:

$$
\mathrm{Forall}(\mathrm{ExecutionFrameBound})(\mathrm{traceOf}(frames))
\land
\mathrm{ContinuityTraceBound}(0,\mathrm{traceOf}(frames))
\land
\mathrm{BoundaryTraceBound}(init,\mathrm{traceOf}(frames)).
$$

The main theorem target is then:

$$
\mathrm{ExactTraceEvidence}(frames)
\land
\mathrm{SimpleKernelChunkInput}(init, meta.semanticRows, \mathrm{traceOf}(frames))
\Longrightarrow
\mathrm{AuthenticatedChunkTraceBound}(frames).
$$

This is the exact chunk-level execution / continuity / boundary closure owned
directly by authenticated staged evidence.

Define `AuthenticatedTemporalSupportBound(frames)` to package the exact
protocol-shaped temporal support:

$$
\mathrm{Stage2TemporalContextBound}(\mathrm{traceOf}(frames))
\land
\mathrm{PcAdjacentBridge}(\mathrm{traceOf}(frames)).
$$

This owner also packages the chunk-local surface through an auxiliary
`AuthenticatedExecutionTraceBound(frames)` object that contains:

$$
\mathrm{AuthenticatedChunkTraceBound}(frames)
\land
\mathrm{stage2TemporalSeedSummary}(frames)
\land
\mathrm{AuthenticatedTemporalSupportBound}(frames)
\land
\mathrm{TemporalInstantiationBound}(\mathrm{traceOf}(frames)).
$$

This auxiliary bundle exists only to keep the concrete register timeline, RAM
timeline, and `pc`-continuity witnesses internal while still exposing both the
exact per-row Stage-2 temporal seed surface and one strong execution-trace
surface to downstream consumers.

`AuthenticatedTemporalSupportBound` is the protocol-shaped theorem-level kernel
closure bundle. It is not an opening claim, not a digest summary, and not a
replacement for exact row-local evidence.

The strengthened closure path is explicit:

1. exact row evidence recovers the per-row Stage-2 temporal seed summary;
2. exact adjacent-pair Stage-2 support yields
   `RegisterAdjacentTraceBound(traceOf(frames))` and
   `RamAdjacentTraceBound(traceOf(frames))`;
3. those adjacent bounds yield one chunk-global
   `Stage2TemporalContextBound(traceOf(frames))`;
4. exact adjacent-pair Stage-3 support yields
   `PcAdjacentBridge(traceOf(frames))`;
5. together they yield `TemporalInstantiationBound(traceOf(frames))` and then
   the exact adjacent-state link theorem.

### Temporal consistency closure

This owner must expose both the direct exact-trace-to-support bridge and the
bridge from that support bundle to the generic temporal-instantiation witness:

$$
\mathrm{ExactTraceEvidence}(frames)
\Longrightarrow
\mathrm{AuthenticatedTemporalSupportBound}(frames)
$$

and

$$
\mathrm{ExactTraceEvidence}(frames)
\land
\mathrm{SimpleKernelChunkInput}(init, meta.semanticRows, \mathrm{traceOf}(frames))
\land
\mathrm{AuthenticatedTemporalSupportBound}(frames)
\Longrightarrow
\mathrm{TemporalInstantiationBound}(\mathrm{traceOf}(frames)).
$$

This theorem packages the exact Jolt-style temporal ingredients needed for
strong execution closure while keeping the concrete witness bundle named and
auditable.

### Whole-trace link closure

This owner must also expose the named adjacent-frame link contract directly:

$$
\mathrm{ExactTraceEvidence}(frames)
\land
\mathrm{SimpleKernelChunkInput}(init, meta.semanticRows, \mathrm{traceOf}(frames))
\Longrightarrow
\mathrm{TraceLinkBound}(\mathrm{traceOf}(frames)).
$$

This theorem may normalize through `Chip8TraceLinkBoundary`, but it must derive
the contract from exact trace evidence itself via the exact temporal-support
bundle instead of consuming a raw whole-trace link hypothesis as an external
assumption.

### Strong execution theorem

With the named link contract discharged from exact trace evidence plus the
chunk-input contract, this owner exposes:

$$
\mathrm{ExactTraceEvidence}(frames)
\land
\mathrm{SimpleKernelChunkInput}(init, meta.semanticRows, \mathrm{traceOf}(frames))
\Longrightarrow
\mathrm{ExecutionCorrect}(rom,\sigma,init,\mathrm{traceOf}(frames)).
$$

### Prepared-step export

Prepared-step export follows from the authenticated trace surface together with
the simple-kernel row-count contract.

The bridge corollary is:

$$
\mathrm{ExactTraceEvidence}(frames)
\land
\mathrm{SimpleKernelChunkInput}(init, meta.semanticRows, \mathrm{traceOf}(frames))
\Longrightarrow
\mathrm{PreparedStepTraceBound}(\mathrm{traceOf}(frames), preparedSteps)
$$

together with the exact prepared-step count required by the simple-kernel
output surface:

$$
preparedSteps.length = meta.semanticRows.
$$

## Paper Anchors

- **Sources**:
  - `./crates/neo-fold-prototype/specs/chip8-kernel.md`
  - `./docs/soundness-specs/twist-and-shout-requirements.md`
  - `./docs/assurance-strategy.md`
- Anchors:
  - Stage-1 / Stage-2 / Stage-3 semantic ownership
  - chunk-local continuity plus start-boundary / final-boundary discipline
  - exact adjacent-state linking across the semantic prefix
  - authenticated chunk-trace closure from exact staged evidence
  - exact prepared-step export from authenticated semantic rows
  - explicit ownership split between chunk input facts, row-local staged
    evidence, and the named whole-trace link contract that must be derived here

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/Trace/AuthenticatedTrace.lean` | Exact trace-evidence closure from authenticated staged rows to chunk-local trace semantics |
| `Nightstream/Chip8/Trace/AuthenticatedTraceInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Frames | `ExactFrameEvidence` | def | Definitional | One row carries exact staged evidence plus the state-well-formedness facts required by the semantic closure |
| Traces | `traceOf` | def | Definitional | Forgets exact-evidence wrappers and recovers the execution-frame trace |
| Traces | `ExactTraceEvidenceFrom` | def | Definitional | Tracks that frame-local step indices agree with trace positions |
| Traces | `ExactTraceEvidence` | def | Definitional | Exact trace-evidence predicate starting at row `0` |
| Bundle | `AuthenticatedChunkTraceBound` | def/structure | Definitional | Packages row-backed execution, continuity, and boundary for one authenticated chunk |
| Summary | `Stage2TemporalSeedSummaryEntry` | def | Definitional | One exact per-row Stage-2 temporal seed summary entry |
| Bundle | `AuthenticatedTemporalSupportBound` | def/structure | Definitional | Packages the chunk-global Stage-2 temporal context and exact Stage-3 `pc` adjacency bridge |
| Theorem | `wf_of_exactFrameEvidence` | theorem | Theorem-Target | Exact authenticated Stage-3 row binding implies the fixed `ONE` coordinate `wf(z)` |
| Theorem | `stage2TemporalSeedSummary_of_frames` | theorem | Theorem-Target | Exact frame evidence recovers the exact per-row Stage-2 temporal seed summary across the frame list |
| Theorem | `headInitialStateMatch_of_chunkInput` | theorem | Theorem-Target | The simple-kernel chunk input contract discharges the head initial-state match for `traceOf frames` |
| Theorem | `traceLength_eq_semanticRows_of_chunkInput` | theorem | Theorem-Target | The simple-kernel chunk input contract discharges exact semantic-row completeness for `traceOf frames` |
| Theorem | `executionFrameBound_of_exactFrameEvidence` | theorem | Theorem-Target | Exact row evidence closes directly to `ExecutionFrameBound` by deriving row-local routing from authenticated staged evidence |
| Theorem | `executionFramesBound_of_exactTrace` | theorem | Theorem-Target | Exact trace evidence yields row-backed execution bounds for every frame |
| Theorem | `continuityTraceBound_of_exactTrace` | theorem | Theorem-Target | Exact trace evidence yields the chunk-local Stage-3 continuity trace |
| Theorem | `temporalTraceBound_of_exactTrace` | theorem | Theorem-Target | Exact trace evidence plus the simple-kernel chunk input contract yield the component-wise temporal trace surface |
| Theorem | `startBoundaryFrame_of_exactHead` | theorem | Theorem-Target | The head row of an exact trace satisfies the semantic start-boundary law |
| Theorem | `lastStepIdx_of_exactTraceFrom_appendLast` | theorem | Theorem-Target | Exact trace indices determine the last row index of a trace suffix |
| Theorem | `finalBoundaryFrame_of_exactTail` | theorem | Theorem-Target | If the exact trace length equals `meta.semanticRows`, the last row satisfies the semantic final-boundary law |
| Theorem | `traceLength_le_publishedLength_of_exactTrace` | theorem | Theorem-Target | Exact trace evidence implies the trace length is bounded by the authenticated published schedule length |
| Theorem | `traceLinkBound_of_exactTrace` | theorem | Theorem-Target | Exact trace evidence plus the simple-kernel chunk input contract derive the named adjacent-state link contract over the semantic prefix |
| Theorem | `authenticatedChunkTraceBound_of_exactTrace` | theorem | Theorem-Target | Exact trace evidence plus the simple-kernel input contract yield the exact authenticated chunk-trace surface |
| Theorem | `executionCorrect_of_authenticatedChunkTraceBound` | theorem | Theorem-Target | The authenticated chunk-trace surface implies `ExecutionCorrect` by first deriving `TraceLinkBound` from its temporal component |
| Theorem | `executionCorrect_of_exactTrace` | theorem | Theorem-Target | Exact trace evidence plus the simple-kernel chunk input contract yield `ExecutionCorrect` |
| Theorem | `preparedStepTraceBound_of_exactTrace` | theorem | Theorem-Target | Exact trace evidence plus the simple-kernel chunk input contract yield exact prepared-step export through continuity alone |
| Theorem | `preparedStepExport_of_exactTrace` | theorem | Theorem-Target | Exact trace evidence yields both exact prepared-step binding and exact prepared-step count `= meta.semanticRows` |

## Proof Obligations

- This owner must not pretend that `Chip8EvidenceCoverage` alone already proves
  `ExecutionCorrect`.
- Row-local routing must be discharged from exact authenticated staged evidence,
  not left as a whole-trace external assumption.
- The named whole-trace link contract must be discharged here from the explicit
  temporal trace surface derived from exact authenticated staged evidence; it
  must not remain an external assumption above `AuthenticatedChunkTraceBound`.
- This owner must consume a chunk-global Stage-2 temporal context, not merely a
  bag of row-local seeds.
- The fixed `ONE` coordinate should be discharged from exact Stage-3 row
  binding, not carried as an extra trace-level assumption.
- The trace theorem must use exact row indices, not an ad hoc existential
  continuity list.
- This owner should discharge the Stage-3 start-boundary and final-boundary
  facts as far as the exact row indices genuinely allow.
- The strongest direct export theorem at this layer should match the kernel
  soundness target and therefore include exact adjacent-state linking rather
  than stopping at a weaker continuity/boundary-only chunk surface.
- The head-frame initial-state match and exact semantic-row count must be
  consumed from the theorem-facing simple-kernel chunk input owner rather than
  being smuggled in as unrelated hypotheses.

## Assumption Ledger

- This module does not re-prove transcript ordering, PCS soundness, or
  Fiat-Shamir soundness.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Nightstream/Chip8/Stage2/EvidenceCoverage.lean`
  - `Nightstream/Chip8/Trace/ChunkInput.lean`
  - `Nightstream/Chip8/Execution/StepComposition.lean`
  - `Nightstream/Chip8/Execution/ExecutionSemantics.lean`
- **Downstream consumers**:
  - top-level kernel soundness
  - trace-level digest normalization
  - release-time artifact audit over whole traces
  - later Rust-refinement theorems for authenticated chunk proofs

## Acceptance Criteria

1. `lake build Nightstream.Chip8.AuthenticatedTrace` succeeds.
2. Exact per-row evidence closes directly to the authenticated chunk-trace
   bundle with only the chunk-boundary hypotheses it truly needs.
3. Exact per-row evidence closes to `ExecutionCorrect` with no additional
   whole-trace-link hypothesis left outside this owner.
4. The prepared-step export theorem is available at the honest continuity-based
   boundary.
5. No `sorry`.
