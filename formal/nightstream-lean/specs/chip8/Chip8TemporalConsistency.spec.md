# Chip8TemporalConsistency Spec

## Purpose

- **What it is**: The theorem-facing owner for component-wise temporal
  consistency across one authenticated CHIP-8 chunk trace.
- **Key property**: it derives exact adjacent-state linking compositionally from
  Stage-2 register / `I` value-over-time, Stage-2 RAM value-over-time, and the
  theorem-facing Stage-3 `pc` bridge from `Chip8PcContinuityBridge`, rather
  than pretending that control-lane continuity alone proves whole-state trace
  linking.
- **Protocol role**: This module sits between row-local staged evidence and the
  named whole-trace link contract in `Chip8TraceLinkBoundary`.

## Target Formulas

Let:

$$
\mathrm{traceOf}(frames)
$$

be the authenticated execution trace induced by the row-backed frame list.

### Register / `I` temporal consistency

Define:

$$
\mathrm{RegisterTemporalBound}(frames)
$$

to mean:

$$
\forall j < frames.length,\; \forall idx \in \{0,\dots,15\},\;
\mathrm{preState}(frames[j]).v(idx) = \mathrm{RegVal}(idx, j)
$$

$$
\forall j < frames.length,\;
\mathrm{preState}(frames[j]).i = \mathrm{RegVal}(16, j)
$$

and, for every adjacent pair:

$$
\forall j+1 < frames.length,\; \forall idx \in \{0,\dots,15\},\;
\mathrm{postState}(frames[j]).v(idx) = \mathrm{RegVal}(idx, j+1)
$$

$$
\forall j+1 < frames.length,\;
\mathrm{postState}(frames[j]).i = \mathrm{RegVal}(16, j+1).
$$

### RAM temporal consistency

Define:

$$
\mathrm{RamTemporalBound}(frames)
$$

to mean:

$$
\forall j < frames.length,\; \forall addr \in \{0,\dots,4095\},\;
\mathrm{preState}(frames[j]).ram(addr) = \mathrm{RamVal}(addr, j)
$$

and, for every adjacent pair:

$$
\forall j+1 < frames.length,\; \forall addr \in \{0,\dots,4095\},\;
\mathrm{postState}(frames[j]).ram(addr) = \mathrm{RamVal}(addr, j+1).
$$

### `pc` temporal consistency

Define:

$$
\mathrm{PcTemporalBound}(trace)
$$

to mean:

$$
\forall j+1 < trace.length,\;
\mathrm{postState}(trace[j]).pc = \mathrm{preState}(trace[j+1]).pc.
$$

This is the whole-trace form of the authenticated Stage-3 `pc` consequence
exported by `Chip8PcContinuityBridge`.

### Combined temporal surface

Define:

$$
\mathrm{TemporalTraceBound}(frames)
$$

to mean:

$$
\mathrm{RegisterTemporalBound}(frames)
\land
\mathrm{RamTemporalBound}(frames)
\land
\mathrm{PcTemporalBound}(\mathrm{traceOf}(frames)).
$$

For downstream theorem surfaces, this component-wise package may also be carried
as a named witness object:

$$
\mathrm{TemporalInstantiationBound}(frames),
$$

whose fields are:

- one register timeline witness,
- one RAM timeline witness,
- one `pc`-continuity witness.

### Whole-trace link derivation

The core theorem target is:

$$
\mathrm{TemporalTraceBound}(frames)
\Longrightarrow
\mathrm{TraceLinkBound}(\mathrm{traceOf}(frames)).
$$

Equivalently, for each adjacent pair:

$$
\mathrm{TemporalTraceBound}(frames)
\Longrightarrow
\mathrm{AdjacentStateLink}(trace[j], trace[j+1]).
$$

The proof is extensional over the CHIP-8 machine state:

- `pc` from `PcTemporalBound`
- `i` from `RegisterTemporalBound`
- `V[0..15]` from `RegisterTemporalBound`
- `RAM[0..4095]` from `RamTemporalBound`

## Paper Anchors

- **Sources**:
  - `./crates/neo-fold-prototype/specs/chip8-kernel.md`
  - `./docs/soundness-specs/twist-and-shout-requirements.md`
  - `./docs/assurance-strategy.md`
- Anchors:
  - strong adjacent-state linking is theorem-level and compositional
  - Stage-3 continuity alone is not enough
  - Stage-2 virtual `RegVal` / `RamVal` surfaces must remain theorem-facing
    inputs to the trace-link proof

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/TemporalConsistency.lean` | Component-wise temporal trace surface for one authenticated chunk |
| `Nightstream/Chip8/TemporalConsistencyInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Temporal | `RegisterTemporalBound` | def | Definitional | Authenticated `V` / `I` values agree with the Stage-2 register timeline |
| Temporal | `RamTemporalBound` | def | Definitional | Authenticated RAM states agree with the Stage-2 RAM timeline |
| Temporal | `PcTemporalBound` | imported def | Dependency | Authenticated adjacent `pc` values satisfy the Stage-3 `pc` bridge |
| Temporal | `TemporalTraceBound` | def/structure | Definitional | Packages the full component-wise temporal surface |
| Temporal | `TemporalInstantiation` | structure | Definitional | Concrete witness carrying one register timeline, one RAM timeline, and one `pc` continuity proof for the trace |
| Temporal | `TemporalInstantiationBound` | def | Definitional | Named proposition that such a concrete temporal instantiation exists |
| Theorem | `temporalTraceBound_of_instantiation` | theorem | Theorem-Target | One concrete temporal instantiation yields the combined temporal trace surface |
| Theorem | `adjacentStateLink_of_temporalTraceBound` | theorem | Theorem-Target | Component-wise temporal consistency yields one adjacent-state link |
| Theorem | `traceLinkBound_of_temporalTraceBound` | theorem | Theorem-Target | Component-wise temporal consistency yields the whole-trace link contract |
| Theorem | `traceLinkBound_of_temporalInstantiation` | theorem | Theorem-Target | The named witness bundle yields the whole-trace link contract |

## Proof Obligations

- This owner must stay compositional; it must not re-own direct row opening,
  transcript, or digest/audit semantics.
- It must not weaken strong trace linking into Stage-3 continuity alone.
- It must use the exact CHIP-8 machine-state coordinates: `pc`, `i`,
  `V[0..15]`, and `RAM[0..4095]`.

## Assumption Ledger

- This owner assumes the upstream row-local staged evidence has already exposed
  the necessary concrete register timeline, RAM timeline, and theorem-facing
  `pc` bridge witness.
- This owner does not prove cryptographic authentication of those surfaces; it
  proves only the semantic extensional consequence.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Chip8RegisterTimeline`
  - `Chip8RamTimeline`
  - `Chip8PcContinuityBridge`
  - `Chip8TraceLinkBoundary`
- **Downstream consumers**:
  - `Chip8AuthenticatedTrace`
  - `Chip8KernelSoundness`
  - kernel digest/audit owners that need a strong trace theorem
