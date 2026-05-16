# Chip8ExecutionSemantics Spec

## Purpose

- **What it is**: The shared semantic-state and authenticated-trace contract for
  the exact supported CHIP-8 kernel.
- **Key property**: it owns the definitions of row semantics,
  whole-instruction semantics, authenticated execution frames, and
  chunk-local execution traces that are shared by `Chip8StepComposition` and
  `Chip8BurstSession`.
- **Protocol role**: This module is the stable semantic owner below the
  composition theorem and below the decomposed burst theorem. It does not prove
  fetch/decode or memory soundness; it defines the semantic targets those
  modules discharge.

## Target Formulas

### Row semantics

Define:

$$
\mathrm{MicrostepCorrect}(rom,\sigma,dec,pre,post)
$$

for the exact supported 9-family kernel:

- `LdImm`
- `AddImm`
- `Mov`
- `AddRegNoCarry`
- `SkipEqImm`
- `Jump`
- `LdI`
- `StoreRegs`
- `LoadRegs`

This is the exact semantic target later proved from authenticated row-local
bounds by `Chip8StepComposition`.

The lowering / visibility convention owned by this semantic layer is:

- each authenticated row is one CHIP-8 microstep;
- row reads observe the semantic `pre` state of that microstep;
- row writes determine only the semantic `post` state of that microstep;
- therefore Twist's "latest prior write" theorem is consumed at row granularity,
  never by assuming a same-row read may observe a same-row write that has not
  yet been exported into `post`;
- ROM / public lookup tables are separate Stage-1 Shout surfaces, while the
  register file and RAM are separate Stage-2 Twist surfaces rather than one
  tagged shared memory.

### Whole-instruction semantics

Define:

$$
\mathrm{InstructionCorrect}(rom,\sigma,dec,pre,post)
$$

so that:

- non-burst instructions are definitionally the same as `MicrostepCorrect`
- `StoreRegs` / `LoadRegs` are macro-instruction semantics with final
  `PC_NEXT = PC + 1`

### Authenticated execution traces

Define an authenticated execution frame:

$$
\mathrm{ExecutionFrame} := (dec, pre, post, z)
$$

and the row-backed predicate:

$$
\mathrm{ExecutionFrameBound}(rom,\sigma,frame)
$$

meaning:

- the 24-coordinate row is well-formed
- the row binds to the semantic pre/post state and decoded row
- the row is semantically microstep-correct

Define:

$$
\mathrm{ExecutionCorrect}(rom,\sigma,init,trace)
$$

to mean:

- frames satisfy the exact adjacent-state link contract
- every frame satisfies `ExecutionFrameBound`
- the trace satisfies the authenticated Stage-3 continuity relation
- the first frame matches the authenticated initial state and start-boundary law
- the last frame satisfies the authenticated Stage-3 final-boundary law

On the simple kernel boundary, `BoundaryTraceBound([]) = \mathrm{False}`. The
initial-state, start-boundary, and final-boundary obligations are therefore
never discharged vacuously on the empty trace, and the exported
`FirstRowPubliclyBound` object always refers to a real head frame.

Define the explicit head-row bundle:

$$
\mathrm{FirstRowPubliclyBound}(init, frame)
$$

meaning:

- `InitialStateMatches(init, frame.pre)`
- `StartBoundaryFrame(frame)`

This is the theorem-facing object that packages the public first-row ownership
split: `PC(0)` comes from the authenticated initial state carried by the chunk
input, while the burst-start side condition comes from the Stage-3 start
boundary.

### Prepared-step export

Define:

$$
\mathrm{PreparedStepTraceBound}(trace,preparedSteps)
$$

to mean the exported prepared steps are exactly the Stage-3 bridge images of
the authenticated row trace.

The primary generic export theorem at this layer is continuity-based:

$$
\mathrm{ContinuityTraceBound}(stepIdx, trace)
\Longrightarrow
\mathrm{PreparedStepTraceBound}(trace, preparedSteps).
$$

The stronger execution-based corollary is then a downstream convenience:

$$
\mathrm{ExecutionCorrect}(rom,\sigma,init,trace)
\Longrightarrow
\mathrm{PreparedStepTraceBound}(trace, preparedSteps).
$$

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/chip8-kernel.md`
- Anchors:
  - exact supported 9-family kernel semantics
  - chunk-local continuity / prepared-step export
  - macro-instruction semantics for `Fx55` / `Fx65`

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/ExecutionSemantics.lean` | Shared semantic objects and trace relations for the supported CHIP-8 kernel |
| `Nightstream/Chip8/ExecutionSemanticsInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Semantics | `MicrostepCorrect` | def | Definitional | Exact row semantics for the supported kernel |
| Semantics | `InstructionCorrect` | def | Definitional | Exact whole-instruction semantics |
| Trace | `ExecutionFrame` | def | Definitional | Row-backed semantic trace element |
| Trace | `ExecutionFrameBound` | def | Definitional | Authenticated row-backed microstep witness |
| Trace | `FirstRowPubliclyBound` | def | Definitional | Exact head-row public/input plus start-boundary bundle |
| Trace | `BoundaryTraceBound` | def | Definitional | The chunk-level initial-state, start-boundary, and final-boundary bundle |
| Trace | `ExecutionCorrect` | def | Definitional | Authenticated chunk-local semantic execution trace |
| Bridge | `PreparedStepTraceBound` | def | Definitional | Prepared steps are exactly the Stage-3 images of the row trace |
| Theorem | `instructionCorrect_of_nonBurstMicrostep` | theorem | Theorem-Target | Non-burst microstep correctness implies whole-instruction correctness |
| Theorem | `firstRowPubliclyBound_of_boundaryTrace` | theorem | Theorem-Target | The chunk boundary bundle exports the exact head-row public/start-boundary object |
| Theorem | `executionCorrect_of_trace` | theorem | Theorem-Target | Chaining plus frame bounds plus continuity plus the chunk-boundary bundle imply execution correctness |
| Theorem | `preparedStepTraceBound_of_continuity` | theorem | Theorem-Target | Continuity alone determines the exact prepared-step export for a row trace |
| Theorem | `preparedStepTraceBound_of_execution` | theorem | Theorem-Target | Correct execution yields exact prepared-step export |

## Proof Obligations

- `MicrostepCorrect` must stay exact to the current supported kernel.
- `InstructionCorrect` must preserve the distinction between row-local memory
  prefix rows and whole `StoreRegs` / `LoadRegs` instructions.
- `ExecutionCorrect` must remain explicitly continuity-aware.
- `ExecutionCorrect` must use the exact adjacent-state link contract; Stage-3
  continuity alone is not sufficient to discharge local chaining.
- `ExecutionCorrect` must include the final Stage-3 boundary on the last active
  row, not only the start boundary on the head row.
- On the simple kernel boundary, the boundary bundle must exclude the empty
  trace rather than letting the boundary facts disappear vacuously.

## Assumption Ledger

- Fetch/decode, lookup, and Stage-2 memory correctness are not proved here.
- Those semantic facts are proved in `Chip8StepComposition`.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Nightstream/Chip8/WitnessMemoryBinding.lean`
  - `Nightstream/Chip8/ContinuityBridge.lean`
- **Downstream consumers**:
  - `Nightstream/Chip8/StepComposition.lean`
  - `Nightstream/Chip8/BurstSession.lean`

## Implementation Plan

1. Define the shared semantic objects.
2. Define the authenticated execution-trace predicates.
3. Prove the generic non-burst, trace, and prepared-step export lemmas.

## Quality Expectations

- Keep this module semantic and shared.
- Do not move fetch/decode or memory-proof ownership here.
- Keep burst whole-instruction closure in `Chip8BurstSession`.

## Acceptance Criteria

1. `lake build Nightstream.Chip8.ExecutionSemantics` succeeds.
2. `Chip8StepComposition` and `Chip8BurstSession` consume the same semantic owner.
3. No `sorry`.

## Out of Scope

- proving row-local semantics from authenticated bounds
- proving burst-session schedule correctness
- proving claim coverage or PCS binding
