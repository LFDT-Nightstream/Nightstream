# Chip8StagedBridge Spec

## Purpose

- **What it is**: The theorem-facing staged bridge artifact for the CHIP-8
  release path.
- **What it is not**: It is not the final packaged proof and it does not
  restate the full kernel soundness theorem.
- **Protocol role**: It replaces the current compatibility-only bridge shape
  with one exact public bridge view plus the exact prepared-step export and the
  exact extension-family bundles that the backend is allowed to consume.

## Target Formulas

Define the canonical stage view for one release stage `s`:

$$
\mathrm{ReleaseStageView}(s) := (s,\ \mathrm{stageFamilies}(s)).
$$

Define the canonical bridge-stage list:

$$
\mathrm{canonicalStageViews}
=
[
\mathrm{ReleaseStageView}(\mathrm{ReadonlyBatch}),
\mathrm{ReleaseStageView}(\mathrm{RegisterHistory}),
\mathrm{ReleaseStageView}(\mathrm{RamHistory})
].
$$

Define the public bridge view:

$$
\mathrm{ReleaseBridgePublicView}
=
(\mathrm{foldSchedule},\ \mathrm{chunkCount},\ \mathrm{preparedStepCount},\ \mathrm{stages}).
$$

Its exact realization predicate is:

$$
\mathrm{ReleaseBridgePublicViewBound}(view, n)
\iff
\mathrm{Valid}(view.\mathrm{foldSchedule})
\land
view.\mathrm{chunkCount} =
\mathrm{chunkCount}(view.\mathrm{foldSchedule}, n)
\land
view.\mathrm{preparedStepCount} = n
\land
view.\mathrm{stages} = \mathrm{canonicalStageViews}.
$$

The schedule-parameterized constructor target is:

$$
\mathrm{Valid}(schedule)
\Longrightarrow
\mathrm{ReleaseBridgePublicViewBound}
(\mathrm{releaseBridgePublicView\_of\_schedule}(schedule, n), n).
$$

The canonical whole-trace constructor target is:

$$
\mathrm{ReleaseBridgePublicViewBound}
(\mathrm{releaseBridgePublicView\_of\_preparedStepCount}(n), n).
$$

For one exact authenticated frame list `frames`, define the exact prepared-step
export carried by the staged bridge:

$$
\mathrm{bridgePreparedSteps}(frames)
:=
\mathrm{map}
(\lambda frame.\ \mathrm{mkPreparedStep}(frame.row))
(\mathrm{traceOf}(frames)).
$$

For each authenticated frame, define the exact readonly-batch bundle carried by
the staged bridge:

$$
\mathrm{FrameReadonlyBundle}(frame)
 :=
\mathrm{ReadonlyBatchBundle}
(
rom,\ frame.pre.pc,\ frame.dec,\ frame.pre.V_x,\ frame.pre.V_y,\ xIdx(frame)
).
$$

The staged bridge must preserve one ordered readonly-batch bundle per exact
frame, not merely an existential set-style coverage predicate. This is the
purpose of `ReadonlyBatchTraceBundle(frames)`.

For one exact authenticated trace, define the staged bridge artifact:

$$
\mathrm{StagedBridgeArtifact}(frames),
$$

whose fields are:

- one exact public bridge view,
- one exact prepared-step count,
- one exact prepared-step trace theorem,
- one ordered per-frame readonly-batch bundle,
- one exact Stage-2 `HistoryBundle(frames)`.

The theorem-facing constructor target is:

$$
\mathrm{ExactTraceEvidence}(frames)
\land
\mathrm{SimpleKernelChunkInput}(init, N, \mathrm{traceOf}(frames))
\Longrightarrow
\mathrm{StagedBridgeArtifact}(frames).
$$

This artifact must preserve:

$$
(\mathrm{bridgePreparedSteps}(frames)).\mathrm{length} = N,
$$

$$
\mathrm{PreparedStepTraceBound}
(\mathrm{traceOf}(frames), \mathrm{bridgePreparedSteps}(frames)),
$$

$$
\mathrm{readonlyBatch.length} = (\mathrm{traceOf}(frames)).\mathrm{length},
$$

and

$$
\mathrm{publicView.preparedStepCount}
=
(\mathrm{bridgePreparedSteps}(frames)).\mathrm{length}.
$$

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Stage view | `ReleaseStageView` | structure | Definitional | Packages one release stage together with its exact family inventory |
| Canonical stages | `canonicalStageViews` | def | Definitional | Fixes the exact release-stage order for the staged bridge |
| Public view | `ReleaseBridgePublicView` | structure | Definitional | Packages the exact public bridge counts and stage inventory |
| Predicate | `ReleaseBridgePublicViewBound` | def | Definitional | States the exact canonical public-bridge view contract |
| Constructor | `releaseBridgePublicView_of_schedule` | def | Definitional | Canonical public-bridge view for one explicit fold schedule |
| Constructor | `releaseBridgePublicView_of_preparedStepCount` | def | Definitional | Canonical public-bridge view for one prepared-step count |
| Theorem | `releaseBridgePublicViewBound_of_schedule` | theorem | Theorem-Target | Any admissible explicit fold schedule yields the exact public-view contract |
| Theorem | `releaseBridgePublicViewBound_of_preparedStepCount` | theorem | Theorem-Target | The canonical constructor satisfies the exact public-view contract |
| Theorem | `foldSchedule_eq_wholeTrace_of_preparedStepCount` | theorem | Theorem-Target | The canonical helper is explicitly whole-trace folding |
| Prepared steps | `bridgePreparedSteps` | def | Definitional | Fixes the exact ordered prepared-step list exported by the staged bridge |
| Bundle | `ReadonlyBatchTraceBundle` | inductive | Definitional | Carries one exact readonly-batch bundle per authenticated frame in order |
| Constructor | `readonlyBatchTraceBundle_of_frames` | def | Theorem-Target | Every authenticated frame yields its exact readonly-batch bundle |
| Theorem | `preparedStepTrace_of_exactTrace` | theorem | Theorem-Target | Exact trace closure yields the exact prepared-step trace theorem |
| Theorem | `preparedStepCount_of_exactTrace` | theorem | Theorem-Target | Exact trace closure yields the exact prepared-step count |
| Artifact | `StagedBridgeArtifact` | structure | Definitional | Packages the exact public view, prepared-step export, readonly-batch trace, and Stage-2 history bundle |
| Constructor | `stagedBridgeArtifact_of_exactTrace` | def | Theorem-Target | Exact authenticated trace plus chunk input yield the staged bridge artifact |
| Theorem | `readonlyBatchLength_of_artifact` | theorem | Theorem-Target | The readonly-batch trace stays aligned with the exact trace length |
| Theorem | `preparedStepCount_matches_publicView` | theorem | Theorem-Target | The public bridge view reports the exact prepared-step count |

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Nightstream/Chip8/ReleaseBridge.lean`
  - `Nightstream/Chip8/Trace/AuthenticatedTrace.lean`
  - `Nightstream/Chip8/Stage2/EvidenceCoverageBounds.lean`
- **Downstream consumers**:
  - the later Rust replacement for `bridge/mod.rs`
  - the later release-proof packaging and checker owners

## Proof Obligations

- The staged bridge may trust only exact trace closure and exact chunk input.
- The staged bridge must preserve the exact order of authenticated frames when
  exporting readonly-batch bundles and prepared steps.
- The public bridge view must be canonical and schedule-bearing; no compatibility flag or
  implementation-local mode bit belongs here.
- The history payload must be the exact Stage-2 `HistoryBundle`, not a weaker
  digest-only approximation.

## Paper Anchors

- **Source**:
  - `./docs/assurance-strategy.md`
  - `./docs/new-fold-plan.md`
  - `./crates/neo-fold-prototype/src/bridge/mod.rs`
  - `./crates/neo-fold-prototype/src/pipeline/mod.rs`
  - `./formal/nightstream-lean/specs/chip8/Chip8ReleaseBridge.spec.md`
  - `./formal/nightstream-lean/specs/chip8/Chip8AuthenticatedTrace.spec.md`

## Out of Scope

- final packaged proof digest
- final compressed opening artifact
- backend CCS/RLC/DEC verification
