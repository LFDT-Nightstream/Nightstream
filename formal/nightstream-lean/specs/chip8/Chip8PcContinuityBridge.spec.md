# Chip8PcContinuityBridge Spec

## Purpose

- **What it is**: The theorem-facing Stage-3 bridge from checked lane-shift
  continuity data to real adjacent-row `pc` continuity.
- **Key property**: `pcTemporalBound_of_adjacentBridge`:
  if each adjacent row pair carries the exact current-row continuity witness,
  one theorem-facing fact that the checked shifted `PC` value equals the
  current row's authenticated `PC_NEXT`, and one theorem-facing fact that the
  same shifted value equals the next row's authenticated `PC`, then the trace
  satisfies `PcTemporalBound`.
- **Protocol role**: This is the owner that makes the remaining Stage-3
  semantic seam explicit. It does not authenticate the shift proof itself; it
  packages the exact semantic support facts still needed to turn that
  authenticated Stage-3 surface into a real-machine-state `pc` theorem. This is
  the theorem-level Stage-3 closure object consumed directly by strong kernel
  soundness.

## Target Formulas

### Whole-trace `pc` continuity

Define:

$$
\mathrm{PcTemporalBound}(trace)
$$

to mean:

$$
\forall j+1 < trace.length,\;
\mathrm{postState}(trace[j]).pc = \mathrm{preState}(trace[j+1]).pc.
$$

### Live-row Stage-3 support facts

The checked Stage-3 batch identity is not itself a semantic theorem about
machine states. To derive real-row `pc` continuity, this owner makes the
missing semantic support facts explicit.

For one checked shift proof and one authenticated current row, define:

$$
\mathrm{ShiftPcMatchesCurrentPcNext}(shiftProof, currentRow)
$$

to mean:

$$
shiftProof.shiftPc = currentRow.PC\_NEXT.
$$

For the same checked shift proof and one authenticated next row, define:

$$
\mathrm{ShiftPcMatchesNextRow}(shiftProof, nextFrame)
$$

to mean:

$$
shiftProof.shiftPc = nextFrame.row.PC.
$$

These are theorem-facing semantic consequences of the Stage-3 checked objects.
They are not new direct opening claims.

Here `shiftProof.shiftPc` is the refined Stage-3 shift value above
`Chip8PaddedContinuityCheck`, not the raw padded-domain `Shift[PC](r_shift)`
before excluded-tail correction.

Classification:

- `PcAdjacentBridgeFrom` / `PcAdjacentBridge` are theorem-level kernel closure
  objects;
- they are not opening claims, not reduction summaries, and not audit-only
  carrier objects.

### Adjacent-row bridge

For one adjacent pair, define:

$$
\mathrm{PcAdjacentBridgeFrom}(j, current, next)
$$

to package:

- the exact current-row `ContinuityRowBound` witness,
- `ShiftPcMatchesCurrentPcNext`,
- `ShiftPcMatchesNextRow`.

The core adjacent-row theorem target is:

$$
\mathrm{ContinuityRowBound}(j,current.row)
\land
\mathrm{ShiftPcMatchesCurrentPcNext}
\land
\mathrm{ShiftPcMatchesNextRow}
\land
\mathrm{WitnessBinds}(current)
\land
\mathrm{WitnessBinds}(next)
\Longrightarrow
current.post.pc = next.pre.pc.
$$

### Whole-trace bridge

Define:

$$
\mathrm{PcAdjacentBridge}(trace)
$$

to mean one `PcAdjacentBridgeFrom` witness exists for every adjacent pair in
the trace.

The whole-trace theorem target is:

$$
\mathrm{ExecutionFrameBound}(trace)
\land
\mathrm{StateWellFormed}(trace)
\land
\mathrm{PcAdjacentBridge}(trace)
\Longrightarrow
\mathrm{PcTemporalBound}(trace).
$$

## Paper Anchors

- **Sources**:
  - `./crates/neo-fold-prototype/specs/chip8-kernel.md`
  - `./docs/assurance-strategy.md`
- Anchors:
  - Stage 3 contributes the `pc` component of adjacent-state linking
  - the checked shift/batch identity is support data, not the final semantic
    theorem by itself
  - strong trace linking must be justified on real authenticated rows

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/PcContinuityBridge.lean` | Theorem-facing Stage-3 `pc` bridge for one trace |
| `Nightstream/Chip8/PcContinuityBridgeInterface.lean` | Public re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Temporal | `PcTemporalBound` | def | Definitional | Adjacent authenticated frames agree on `pc` |
| Support | `ShiftPcMatchesCurrentPcNext` | def | Definitional | The checked shifted `PC` equals the current row's authenticated `PC_NEXT` |
| Support | `ShiftPcMatchesNextRow` | def | Definitional | The checked shifted `PC` equals the next row's authenticated `PC` |
| Support | `PcAdjacentBridgeFrom` | def | Definitional | Packages the exact Stage-3 support witness for one adjacent pair |
| Support | `PcAdjacentBridge` | def | Definitional | Packages the exact Stage-3 support witness for all adjacent pairs |
| Theorem | `adjacentPc_of_bridge` | theorem | Theorem-Target | One adjacent Stage-3 bridge witness yields real `pc` equality for that pair |
| Theorem | `pcTemporalBound_of_adjacentBridge` | theorem | Theorem-Target | Per-pair Stage-3 bridge witnesses yield whole-trace `pc` continuity |

## Proof Obligations

- This owner must stay focused on the `pc` component only.
- It must not silently replace the theorem-facing semantic support facts with a
  weaker raw batched scalar identity.
- It must not re-own register / `I` or RAM temporal consistency.

## Assumption Ledger

- This owner assumes `ExecutionFrameBound` has already tied each row to the
  corresponding machine-state `pre` / `post` pair.
- This owner assumes the exact theorem-facing Stage-3 support facts
  `ShiftPcMatchesCurrentPcNext` and `ShiftPcMatchesNextRow` are already
  available for the adjacent pairs under consideration.
- This owner does not prove cryptographic soundness of the Stage-3 batch check;
  it proves the real-row semantic consequence once those support facts are
  supplied.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Chip8ContinuityBridge`
  - `Chip8ExecutionSemantics`
- **Downstream consumers**:
  - `Chip8TemporalConsistency`
  - `Chip8AuthenticatedTrace`
  - strong kernel soundness/digest/audit owners
