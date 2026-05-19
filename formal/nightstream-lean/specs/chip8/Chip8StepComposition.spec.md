# Chip8StepComposition Spec

## Purpose

- **What it is**: The theorem-facing semantic composition contract for the
  final supported CHIP-8 kernel.
- **Key property**: `microstepCorrect_of_bounds`: if the local row relation,
  Stage-1 fetch/decode/lookup facts, Stage-2 memory facts, and authenticated
  lane-row bindings all hold, then the current supported CHIP-8 row is
  semantically correct.
- **Protocol role**: This is the layer where Nightstream proves the
  CHIP-8-specific glue from the final kernel's local and staged theorem
  surfaces to machine-level execution semantics.

This module is intentionally semantic. It consumes already-extracted semantic
facts. It does **not** derive those facts from opening manifests or proof
objects; that bridge belongs to `Chip8EvidenceCoverage`.

## Target Formulas

### Semantic objects

The composition layer reasons over:

- `DecodedRow`: the authenticated per-row supported-kernel descriptor
- `MachineState`: the semantic state for the supported kernel subset
  (`pc_word`, `i`, register file, RAM)
- `InitialState`: the authenticated chunk-initial state

The current supported subset is exactly:

- `LdImm`
- `AddImm`
- `Mov`
- `AddRegNoCarry`
- `SkipEqImm`
- `Jump`
- `LdI`
- `StoreRegs`
- `LoadRegs`

### Imported boundary relations

The composition theorem consumes the following imported or previously proved
relations.

It also requires the following row-local semantic hypotheses, which are not
part of the staged-evidence extraction boundary itself:

$$
\mathrm{StateWellFormed}(pre)
\land
\mathrm{StateWellFormed}(post)
$$

and:

$$
\mathrm{wf}(z)
\land
\mathrm{chip8RoutingSound}(z).
$$

$$
\mathrm{WitnessBinds}(pre, post, dec, z)
$$

This binds the 24-coordinate lane row to the semantic state objects and the
decoded row.

$$
\mathrm{FetchDecodeBound}(romTable, pre.PC, dec)
$$

This means the committed ROM row at the current absolute word address decodes
to the exact authenticated row descriptor.

$$
\mathrm{LookupBound}(dec, pre, z)
$$

This means `LOOKUP_OUTPUT` is the exact semantic helper result required by the
current row.

$$
\mathrm{MemoryBound}(pre, post, init, dec, z)
$$

This means the memory-derived lane values and Stage-2 register/RAM objects are
semantically correct for the current row.

$$
\mathrm{chip8RowLocalSound}(z)
$$

This is the local row-local consequence imported from `Chip8Routing`.

### Microstep correctness

The primary theorem target is:

$$
\mathrm{StateWellFormed}(pre)
\land
\mathrm{StateWellFormed}(post)
\land
\mathrm{wf}(z)
\land
\mathrm{chip8RoutingSound}(z)
\land
\mathrm{WitnessBinds}(pre, post, dec, z)
\land
\mathrm{FetchDecodeBound}(romTable, pre.PC, dec)
\land
\mathrm{LookupBound}(dec, pre, z)
\land
\mathrm{MemoryBound}(pre, post, init, dec, z)
\land
\mathrm{chip8RowLocalSound}(z)
\Longrightarrow
\mathrm{MicrostepCorrect}(romTable, init, dec, pre, post).
$$

For the supported exact families, `MicrostepCorrect` must express the exact
final-kernel row semantics:

- `LdImm`: `REG_X_NEXT = KK`, `PC_NEXT = PC + 1`
- `AddImm`: low-byte add to `V[x]`, `PC_NEXT = PC + 1`
- `Mov`: `REG_X_NEXT = REG_Y`, `PC_NEXT = PC + 1`
- `AddRegNoCarry`: low-byte add of `REG_X` and `REG_Y`, no `VF` side effect,
  `PC_NEXT = PC + 1`
- `SkipEqImm`: `PC_NEXT = PC + 1 + LOOKUP_OUTPUT`
- `Jump`: `PC_NEXT = NNN_WORD`
- `LdI`: `I_NEXT = NNN_ADDR`, `PC_NEXT = PC + 1`
- `StoreRegs` row: `RAM_ADDR = I_REG + X_IDX`, `REG_X_NEXT = REG_X`,
  `PC_NEXT = PC + BURST_LAST`
- `LoadRegs` row: `RAM_ADDR = I_REG + X_IDX`, `REG_X_NEXT = MEM_VALUE`,
  `PC_NEXT = PC + BURST_LAST`

### Whole-instruction correctness for decomposed instructions

For `Fx55` and `Fx65`, whole-instruction correctness is owned by
`Chip8BurstSession`, which consumes the shared execution semantics from
`Chip8ExecutionSemantics` together with authenticated frame/session bounds.
This module imports that result; it does not re-own the burst schedule
predicate locally.

### Chunk execution correctness

The final kernel is chunk-local. Stage 3 owns continuity support, while the
strong adjacent-state link theorem is discharged above this layer through the
explicit temporal owner. Define:

$$
\mathrm{ExecutionCorrect}(romTable, init, trace)
$$

to mean:

- every row in the trace satisfies `MicrostepCorrect`
- consecutive rows satisfy the exact adjacent-state link contract
- the first row agrees with the authenticated public initial state
- the chunk begins on an instruction boundary with no in-flight burst state
- the last active row satisfies the authenticated Stage-3 final-boundary rule

This is the supported-kernel execution theorem surface exported by the current
final kernel.

### Prepared-step correctness

Let `PreparedStepTraceBound(trace, preparedSteps)` mean that the exported
prepared steps are exactly the Stage-3 bridge images of the semantic rows in the
trace.

The final execution-to-bridge theorem target is:

$$
\mathrm{ExecutionCorrect}(romTable, init, trace)
\land
\mathrm{PreparedStepTraceBound}(trace, preparedSteps)
\Longrightarrow
\text{the root main lane receives the exact authenticated semantic rows}.
$$

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/chip8-kernel.md`
- Anchors:
  - supported opcode coverage
  - row-local routing relation
  - Stage-2 memory semantics
  - Stage-2 value-over-time semantics for virtual `RegVal` / `RamVal`
  - Stage-3 continuity and bridge binding

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/StepComposition.lean` | Composition theorems from final-kernel bounds to supported-subset CHIP-8 semantics |
| `Nightstream/Chip8/StepCompositionInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Semantic objects | `DecodedRow` | def | Definitional | Authenticated per-row supported-kernel descriptor |
| Semantic objects | `MachineState` | def | Definitional | Semantic state space for the supported CHIP-8 subset |
| Semantic objects | `InitialState` | def | Definitional | Authenticated chunk-initial state |
| Bindings | `WitnessBinds` | def | Definitional | Authenticated lane-row binding |
| Bindings | `FetchDecodeBound` | def | Definitional | Authenticated Stage-1 fetch/decode binding |
| Bindings | `LookupBound` | def | Definitional | Authenticated Stage-1 helper-lookup semantics |
| Bindings | `MemoryBound` | def | Definitional | Authenticated Stage-2 semantic memory binding |
| Routing | `wf` | def | Imported-Definitional | Public `ONE` coordinate is fixed to `1` |
| Routing | `chip8RoutingSound` | def | Imported-Definitional | Row-local routing equations force the exact `VX_NEXT` / `I_NEXT` / `PC_NEXT` / `RAM_ADDR` consequences |
| Well-formedness | `StateWellFormed` | def | Definitional | Semantic machine states stay in the exact byte/address ranges used by the kernel |
| Semantics | `MicrostepCorrect` | def | Definitional | Semantic correctness of one supported-kernel row |
| Semantics | `InstructionCorrect` | def | Definitional | Semantic correctness of one whole supported instruction |
| Semantics | `BurstScheduleCorrect` | def | Imported-Definitional | Authenticated burst-session schedule/binding predicate imported from `Chip8BurstSession` |
| Semantics | `ExecutionCorrect` | def | Definitional | Semantic correctness of one chunk-local execution trace |
| Bridge | `PreparedStepTraceBound` | def | Definitional | Exported prepared steps are exactly the Stage-3 images of the semantic rows |
| Theorem | `microstepCorrect_of_bounds` | theorem | Theorem-Target | Final-kernel row bounds imply row semantics |
| Theorem | `instructionCorrect_of_burst` | theorem | Imported-Theorem | Authenticated burst-session bounds imply whole-instruction correctness for `StoreRegs` / `LoadRegs` |
| Theorem | `executionCorrect_of_trace` | theorem | Theorem-Target | Correct rows plus continuity and the chunk-boundary bundle imply chunk execution correctness |
| Theorem | `preparedStepTraceBound_of_execution` | theorem | Theorem-Target | Correct execution plus Stage-3 binding yields exact prepared-step export |

## Proof Obligations

- `MicrostepCorrect` must match the final supported-kernel semantics exactly,
  including `AddRegNoCarry`, `NNN_ADDR` vs `NNN_WORD`, and `PC_NEXT = PC + BURST_LAST`
  on memory-prefix rows.
- This module must target the final 24-coordinate row.
- The theorem surface must keep `wf(z)`, `chip8RoutingSound(z)`, and the
  required machine-state well-formedness hypotheses explicit until a theorem-
  facing root/main-lane owner discharges them.
- `ExecutionCorrect` must use authenticated continuity and initial-state facts,
  not an ad hoc list-linking predicate alone.
- `ExecutionCorrect` must not be justified by Stage-3 continuity alone; the
  stronger adjacent-state link theorem has to come from the higher temporal
  owner.
- `ExecutionCorrect` must include the final Stage-3 boundary on the last active
  row, not only the start-of-chunk burst boundary.
- The composition theorem must remain exact to the currently supported subset; a
  future `DRW`/keypad/timer kernel is a separate extension.

## Assumption Ledger

- The semantic facts consumed here are expected to be supplied by
  `Chip8EvidenceCoverage`.
- The root main-lane CCS proof remains external to this module.
- Full-game semantics for unsupported opcodes are outside the current kernel.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Nightstream/Chip8/Routing.lean`
  - `Nightstream/Chip8/FetchDecodeBinding.lean`
  - `Nightstream/Chip8/WitnessMemoryBinding.lean`
  - `Nightstream/Chip8/ExecutionSemantics.lean`
  - `Nightstream/Chip8/BurstSession.lean`
  - `Nightstream/Chip8/ContinuityBridge.lean`
  - `Nightstream/Chip8/EvidenceCoverage.lean`
- **Downstream consumers**:
  - `Nightstream/Chip8/StagedExecutionDigest.lean`
  - `Nightstream/Chip8/ArtifactAudit.lean`
  - later Rust-refinement theorems for the final kernel proof object
  - later larger CHIP-8 kernels that add draw/input/timer families

## Implementation Plan

1. Define the exact row semantics for the supported subset.
2. Prove the row-level composition theorem.
3. Import and re-export the burst whole-instruction theorem from `Chip8BurstSession`.
4. Prove chunk execution correctness and prepared-step export correctness.

## Quality Expectations

- Keep the module exact to the final kernel.
- Keep unsupported game-oriented families out of the local theorem surface.
- Keep continuity and bridge facts imported explicitly rather than hidden.

## Acceptance Criteria

1. `lake build Nightstream.Chip8.StepComposition` succeeds.
2. The theorem surface matches the final supported kernel exactly.
3. The execution theorem is continuity-aware and chunk-local.
4. No `sorry`.

## Out of Scope

- full CHIP-8 ISA correctness
- `DRW`, keypad, timer, stack, and sound semantics
- the root main-lane CCS proof itself
