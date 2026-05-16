# Chip8ContinuityBridge Spec

## Purpose

- **What it is**: The theorem-facing Stage-3 contract for continuity and bridge
  binding in the final CHIP-8 kernel.
- **Key property**: `preparedStepBound_of_rowBinding`: if continuity is
  authenticated via `LaneShiftProof` and each exported row is bound back to
  `C_lane`, then every exported `PreparedStep` is the canonical root encoding of
  one authenticated semantic row.
- **Protocol role**: This is the Stage-3 owner for continuity support,
  start-boundary correctness, row-binding claims, and the exact handoff into
  the SuperNeo root main lane. It owns the checked Stage-3 support objects, not
  the later theorem-facing real-row `pc` consequence; that semantic bridge is
  owned upstream by `Chip8PcContinuityBridge`.

This owner is intentionally above `Chip8PaddedContinuityCheck`. The shifted and
current values used here are the refined active-prefix values after explicit
excluded-tail correction, not the raw padded-domain openings at `r_shift`.

## Target Formulas

### Continuity support relation

Stage 3 owns the checked continuity support relation:

- `PC(j+1) = PC_NEXT(j)` on real row pairs
- burst progression on intermediate memory-prefix rows
- burst reset on memory-prefix starts
- the explicit start-boundary rule `IsMemOp(0) * X_IDX(0) = 0`
- the explicit final-boundary rule
  `IsMemOp(N-1) * (1 - BURST_LAST(N-1)) = 0`

The continuity relation is checked at one Stage-3 cycle point `r_shift`.
This spec owns that checked support surface. The theorem-facing extraction of
real adjacent-frame `pc` equality from the checked shift witness is intentionally
split into `Chip8PcContinuityBridge`.

### Pair mask

Define the verifier-computable real-pair mask:

$$
\mathrm{PairMask}_N(j) =
\begin{cases}
1 & \text{if } 0 \le j < N-1, \\
0 & \text{otherwise.}
\end{cases}
$$

and let `PairMask_N(X)` be its multilinear extension over the cycle hypercube.

### LaneShift reduction

Stage 3 does not assume ad hoc access to `PC(j+1)`, `X_IDX(j+1)`, or
`IsMemOp(j+1)`. Instead it uses an explicit checked virtual reduction:

$$
\mathrm{Shift}[f](X) = \sum_{j=0}^{T-2} eq(X,j)\cdot f(j+1)
$$

for each `f` in `{PC, X_IDX, IsMemOp}`.

Define a theorem-facing `LaneShiftClaim` carrying:

- `source_commitment = C_lane`
- `source_point = r_shift`
- `source_columns = [PC, X_IDX, IsMemOp]`
- `shifted_columns = [Shift[PC], Shift[X_IDX], Shift[IsMemOp]]`
- `claimed_shift_values = [shift_pc, shift_x_idx, shift_is_memop]`

and a checked proof object:

$$
\mathrm{LaneShiftBound}(claim, proof).
$$

### Continuity check

After authenticating the shifted values, Stage 3 opens current-row lane columns
at `r_shift` and checks:

$$
\delta_{pc} = PairMask_N(r_{shift}) \cdot (shift\_pc - PC\_NEXT)
$$

$$
\delta_{burstStep}
=
PairMask_N(r_{shift}) \cdot IsMemOp \cdot (1 - BURST\_LAST)
\cdot (shift\_x\_idx - X\_IDX - 1)
$$

$$
\delta_{burstReset}
=
PairMask_N(r_{shift}) \cdot shift\_is\_memop
\cdot (1 - IsMemOp + BURST\_LAST)\cdot shift\_x\_idx
$$

and the batched identity:

$$
\delta_{pc} + \beta_1 \cdot \delta_{burstStep} + \beta_2 \cdot \delta_{burstReset} = 0.
$$

Define:

$$
\mathrm{ContinuityBound}(N, r_{shift}, claim, proof, currentRow).
$$

### Start-boundary rule

Let `j0_bits = 0^{CYCLE_BITS}`.

Stage 3 opens `IsMemOp` and `X_IDX` from `C_lane @ j0_bits` and proves:

$$
\mathrm{IsMemOp}(0)\cdot X\_IDX(0) = 0.
$$

Define:

$$
\mathrm{StartBoundaryBound}(j0Row).
$$

This is the theorem-facing statement that chunks begin on instruction
boundaries with no in-flight burst state.

### Final-boundary rule

Let `j_last_bits = bits_le(N - 1)` in the same little-endian cycle-bit order
used by `C_lane`.

Stage 3 opens `IsMemOp` and `BURST_LAST` from `C_lane @ j_last_bits` and
proves:

$$
\mathrm{IsMemOp}(N-1)\cdot (1 - \mathrm{BURST\_LAST}(N-1)) = 0.
$$

Define:

$$
\mathrm{FinalBoundaryBound}(jLastRow).
$$

This is the theorem-facing statement that a simple chunk cannot end in the
middle of a decomposed memory-prefix instruction.

Stage 3 must also expose the theorem-facing fact that these authenticated
boundary rows are the boundary projections of the semantic row when the row
index matches the corresponding boundary.

Define:

$$
\mathrm{StartBoundaryMatches}(stepIdx, j0Row, z)
$$

to mean that if `stepIdx = 0`, then `j0Row.IsMemOp = z[19]` and
`j0Row.X_IDX = z[20]`.

Define:

$$
\mathrm{FinalBoundaryMatches}(stepIdx, N, jLastRow, z)
$$

to mean that if `stepIdx + 1 = N`, then `jLastRow.IsMemOp = z[19]` and
`jLastRow.BURST_LAST = z[22]`.

### Row-binding claims

For each exported row `j`, the bridge owns:

$$
\mathrm{RowBindingClaim}_j = (C_{lane}, j_{bits},
[\mathrm{PC}, \mathrm{PC\_NEXT}, \mathrm{REG\_X}, \mathrm{REG\_Y}, \mathrm{REG\_X\_NEXT},
\mathrm{I\_REG}, \mathrm{I\_NEXT}, \mathrm{KK}, \mathrm{NNN\_ADDR},
\mathrm{NNN\_WORD}, \mathrm{MEM\_VALUE}, \mathrm{LOOKUP\_OUTPUT},
\mathrm{WritesLookupToX}, \mathrm{WritesMemToX}, \mathrm{PreservesX},
\mathrm{WritesNnnToI}, \mathrm{IsJump}, \mathrm{IsBranch}, \mathrm{IsMemOp},
\mathrm{X\_IDX}, \mathrm{Y\_IDX}, \mathrm{BURST\_LAST}, \mathrm{RAM\_ADDR}]).
$$

Define:

$$
\mathrm{RowBound}(claim_j, z_j)
$$

to mean:

- the claim opens exactly those 23 committed non-fixed lane coordinates of row
  `j` from `C_lane`, in that canonical registry order
- `z_j[0] = ONE = 1` is inserted as the fixed verifier-known coordinate
- the resulting semantic row is exactly `z_j`

### Root encoding and prepared-step binding

Let:

$$
\mathrm{RootEncode}(z_j) = (w_j, Z_j)
$$

be the canonical packed root witness encoding used by the root prover.

Normative meaning:

- `w_j` is coordinates `1..23` of the semantic row `z_j`;
- `Z_j` is obtained by:
  1. padding `z_j` at the tail with zeros to a multiple of `D`,
  2. reshaping that padded vector into a `D × cols` matrix by columns,
  3. applying the canonical Ajtai/root witness encoding determined only by the
     public root parameters;
- the imported root encoding and Ajtai commitment context is fixed by the
  public `root_params_id` / `vm_spec` boundary from `Chip8RomScheduleBinding`,
  not by hidden caller-chosen functions or witness-layout conventions;
- `RootEncode` must therefore be fixed by theorem text or an imported root
  encoding interface, not by reference to an implementation helper.

Define:

$$
\mathrm{PreparedStepBound}(z_j, step_j)
$$

to mean:

- `step_j.witness.w = w_j`
- `step_j.witness.Z = Z_j`
- `step_j.mcs.c = Ajtai_commit(Z_j)`
- `step_j.mcs.x = [1]`
- `step_j.mcs.m_in = 1`

The explicit bridge-binding leaves are protocol-binding objects:

- `BridgeBinding_j` reuses the exact same accepted row-opening path as
  `RowProjectionWitness_j`, not merely the same row index or the same direct
  claim label;
- these leaves are not direct opening claims and not theorem-level temporal
  closure objects;
- the theorem-level Stage-3 semantic closure object remains
  `Chip8PcContinuityBridge.PcAdjacentBridge`.

### Bundled Stage-3 theorem

Define:

$$
\mathrm{Stage3Bound}(N, shiftClaim, shiftProof, startRow, finalRow, rowClaims, preparedSteps)
$$

to mean:

- `ContinuityBound(...)`
- `StartBoundaryBound(startRow)`
- `FinalBoundaryBound(finalRow)`
- every exported row claim satisfies `RowBound`
- every exported prepared step satisfies `PreparedStepBound`

The main theorem target is:

$$
\mathrm{Stage3Bound}(\dots)
\Longrightarrow
\text{every exported prepared step is bound to one authenticated row of } C_{lane}.
$$

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/chip8-kernel.md`
- Anchors:
  - continuity support relation
  - `LaneShiftProof`
  - `ContinuityCheck`
  - opening boundary row-binding openings
  - bridge binding mechanism
  - prepared-step construction

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/ContinuityBridge.lean` | Stage-3 continuity and row-to-`PreparedStep` bridge-binding theorems |
| `Nightstream/Chip8/ContinuityBridgeInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Continuity | `PairMaskN` | def | Definitional | Real-pair mask for continuity over the padded cycle domain |
| Continuity | `LaneShiftClaim` | def | Definitional | Exact Stage-3 shifted-column claim object |
| Continuity | `LaneShiftBound` | def | Definitional | Checked virtual reduction against `C_lane` |
| Continuity | `ContinuityBound` | def | Definitional | Exact Stage-3 continuity identity at `r_shift` |
| Boundary | `StartBoundaryBound` | def | Definitional | Exact start-of-chunk burst-boundary rule |
| Boundary | `StartBoundaryMatches` | def | Definitional | When `stepIdx = 0`, the authenticated start-boundary row is the boundary projection of semantic row `z` |
| Boundary | `FinalBoundaryBound` | def | Definitional | Exact end-of-chunk burst-boundary rule |
| Boundary | `FinalBoundaryMatches` | def | Definitional | When `stepIdx + 1 = N`, the authenticated final-boundary row is the boundary projection of semantic row `z` |
| Bridge | `RowBindingClaim` | def | Definitional | Exact per-row direct binding claim against `C_lane` |
| Bridge | `RowBound` | def | Definitional | One row-binding claim determines one semantic row |
| Bridge | `RootEncode` | def | Definitional | Canonical root witness encoding for one semantic row |
| Bridge | `PreparedStepBound` | def | Definitional | One prepared step is exactly the root encoding of one row |
| Bundle | `Stage3Bound` | def | Definitional | Complete Stage-3 continuity and bridge-binding bundle |
| Theorem | `continuityBound_of_laneShift` | theorem | Theorem-Target | Authenticated shifted values plus current-row openings imply continuity |
| Theorem | `startBoundaryBound_of_match` | theorem | Theorem-Target | Authenticated start-boundary rows transfer to the semantic row boundary when `stepIdx = 0` |
| Theorem | `finalBoundaryBound_of_match` | theorem | Theorem-Target | Authenticated final-boundary rows transfer to the semantic row boundary when `stepIdx + 1 = N` |
| Theorem | `preparedStepBound_of_rowBinding` | theorem | Theorem-Target | Row binding implies exact prepared-step binding |
| Theorem | `stage3Bound_exports_authenticatedRows` | theorem | Theorem-Target | Stage 3 exports only authenticated rows to the root main lane |

## Proof Obligations

- `LaneShiftProof` must remain a checked virtual reduction, not an opening
  claim.
- The continuity identity must use exactly the final kernel's
  `PairMask_N`, `β1`, and `β2` batching surface.
- The Stage-3 bundle must include both the start-boundary and final-boundary
  rules from the final kernel.
- Row binding must use explicit row openings against `C_lane`, not the rejected
  aggregate-identity shortcut.
- `RootEncode` must be the exact canonical root encoding used by the root
  prover and must be fixed by the imported public root-parameter boundary, not
  by ad hoc caller choice.

## Assumption Ledger

- This module does not re-prove the root main-lane CCS proof.
- This module does not re-prove PCS opening verification.
- This module imports the canonical root encoding and Ajtai commitment surface
  from the public root-parameter boundary.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Nightstream/PCSOpeningSemantics.lean`
  - later root-encoding interface from the SuperNeo side
- **Downstream consumers**:
  - `Nightstream/Chip8/EvidenceCoverage.lean`
  - later Rust-refinement theorems for `PreparedStep` export
  - generic `MainLaneBridge`

## Implementation Plan

1. Define the continuity mask and shift-claim surface.
2. Define the checked Stage-3 continuity predicate.
3. Define row-binding and prepared-step binding.
4. Prove the bundled Stage-3 bridge theorem.

## Quality Expectations

- Keep Stage 3 separate from Stage 1 and Stage 2.
- Keep row binding explicit and commitment-based.
- Make the handoff into the root main lane mechanically obvious.
- Keep `BridgeBinding_j` explicit as protocol-binding evidence rather than
  silently inferring it from prepared-step recomputation.

## Acceptance Criteria

1. `lake build Nightstream.Chip8.ContinuityBridge` succeeds.
2. The theorem surface covers `LaneShiftProof`, `ContinuityCheck`, and explicit
   row binding.
3. The bridge uses explicit row openings, not an aggregate shortcut.
4. No `sorry`.

## Out of Scope

- Stage-1 fetch/decode proofs
- Stage-2 Twist proofs
- the root main-lane CCS proof itself
