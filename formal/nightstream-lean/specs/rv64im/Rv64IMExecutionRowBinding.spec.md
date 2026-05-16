# Rv64IMExecutionRowBinding Spec

## Purpose

- **What it is**: The theorem-facing Stage-1 owner for execution-row outputs,
  dense slot manifests, and the linkage batch back into the lane and decode
  handoff.
- **What it is not**: It is not the bytecode fetch theorem and it does not own
  Stage-2 register or RAM authentication.
- **Protocol role**: It fixes the concrete Stage-1 execution families
  (`executionRow`, `aluSubtables`, `branchCondition`), the dense Jolt-style
  slot layout, and the exact linkage from authenticated Stage-1 outputs back to
  the lane-visible row.

## Target Formulas

Define the concrete family ids:

$$
\mathrm{ExecutionRowFamily} := \mathrm{executionRow},
$$

$$
\mathrm{AluSubtableFamily} := \mathrm{aluSubtables},
$$

$$
\mathrm{BranchConditionFamily} := \mathrm{branchCondition}.
$$

Define the normative dense slot budgets:

$$
\mathrm{MAX\_ALU\_QUERY\_SLOTS} := 64,
$$

$$
\mathrm{MAX\_BRANCH\_QUERY\_SLOTS} := 16.
$$

The canonical dense slot indices are:

$$
\mathrm{byteParallelSlotIndex}(k) := k,
$$

$$
\mathrm{mulGridSlotIndex}(i,j) := 8i + j.
$$

The theorem-facing execution-result object is:

$$
\mathrm{ExecutionResults}
:=
(\mathrm{aluResult}, \mathrm{stepPc}, \mathrm{jumpTarget}, \mathrm{memAddr}, \mathrm{branchTaken}).
$$

The lane-visible Stage-1 projection object is:

$$
\mathrm{Stage1LaneView}
$$

carrying the lane-bound decoded metadata, `AdvanceArchPc`, execution results,
and `branchTakenMux`.

Define:

$$
\mathrm{branchTakenMux}(\mathrm{isBranch}, \mathrm{branchTaken})
:=
\mathrm{isBranch} \land \mathrm{branchTaken}.
$$

Define the dense slot-manifest bound:

$$
\mathrm{DenseAluSlotUsageBound}(\mathrm{slotUsedAlu})
$$

meaning every slot index `slot >= 64` is unused, and

$$
\mathrm{DenseBranchSlotUsageBound}(\mathrm{slotUsedBranch})
$$

meaning every slot index `slot >= 16` is unused.

Define:

$$
\mathrm{DenseSlotManifestBound}(usage)
$$

as the conjunction of those two tail-zero constraints.

Define the exact Stage-1 linkage batch:

$$
\mathrm{Stage1LinkageBound}(row, lane, handoff, results)
$$

meaning:

- lane `pc`, register selectors, immediate, write flags, and control flags
  equal the authenticated decoded row,
- `lane.advanceArchPc = row.isLastInSequence`,
- lane `aluOut`, `stepPc`, `jumpTarget`, and `memAddr` equal the authenticated
  execution results,
- `lane.branchTaken = results.branchTaken`,
- `lane.branchTakenMux = branchTakenMux(row.isBranch, results.branchTaken)`,
- `handoff = toDecodeHandoff(row)`.

Define the exact Stage-1 support surface:

$$
\mathrm{Stage1SupportBound}(lane)
$$

meaning:

- taken control-flow rows satisfy the authenticated target-alignment discharge,
- unsigned DIV/REM rows expose the authenticated `MULU_NO_OVERFLOW` support
  relation.

The concrete emitted projection owners are:

$$
\mathrm{ExecutionRowProjection}(p) := \mathrm{CEProjection}(\mathrm{ExecutionRowFamily}, p),
$$

$$
\mathrm{AluSubtableProjection}(p) := \mathrm{ShoutReadProjection}(\mathrm{AluSubtableFamily}, p),
$$

$$
\mathrm{BranchConditionProjection}(p) := \mathrm{ShoutReadProjection}(\mathrm{BranchConditionFamily}, p).
$$

The `executionRow` family may merge into the main lane when it is the concrete
main family at the current point. The ALU and branch families are read-only
lookup owners and therefore never enter the main lane directly.

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/riscv-kernel.md`
- Anchors:
  - canonical Stage-1 slot manifests
  - branch-condition channel
  - Stage-1 address-correctness obligations
  - Stage-1 linkage batch

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Rv64IM/Stage1/ExecutionRowBinding.lean` | Stage-1 execution families, dense slot manifests, and linkage surface |
| `Nightstream/Rv64IM/Stage1/ExecutionRowBindingInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Family id | `executionRowFamily` | def | Definitional | Fixes the CE owner for authenticated execution-row linkage |
| Family id | `aluSubtableFamily` | def | Definitional | Fixes the Shout owner for ALU subtable lookups |
| Family id | `branchConditionFamily` | def | Definitional | Fixes the Shout owner for branch-condition lookups |
| Slot budget | `maxAluQuerySlots` | def | Definitional | Fixes the dense ALU slot budget to 64 |
| Slot budget | `maxBranchQuerySlots` | def | Definitional | Fixes the dense branch slot budget to 16 |
| Slot index | `byteParallelSlotIndex` | def | Definitional | Canonical byte-parallel slot packing |
| Slot index | `mulGridSlotIndex` | def | Definitional | Canonical 8x8 multiplication-grid slot packing |
| Theorem | `byteParallelSlotIndex_lt_maxAluQuerySlots` | theorem | Theorem-Target | Canonical byte-parallel slots stay inside the ALU budget |
| Theorem | `mulGridSlotIndex_lt_maxAluQuerySlots` | theorem | Theorem-Target | Canonical multiplication-grid slots stay inside the ALU budget |
| Results | `ExecutionResults` | structure | Definitional | Packages the authenticated Stage-1 execution outputs |
| Lane view | `Stage1LaneView` | structure | Definitional | Packages the lane-visible Stage-1 row used by the linkage batch |
| Slot usage | `ExecutionSlotUsage` | structure | Definitional | Carries the per-slot usage masks for ALU and branch families |
| Boundary | `DenseAluSlotUsageBound` | def | Definitional | Slots above the ALU budget are forced inactive |
| Boundary | `DenseBranchSlotUsageBound` | def | Definitional | Slots above the branch budget are forced inactive |
| Boundary | `DenseSlotManifestBound` | def | Definitional | Combines the dense ALU and branch tail-zero constraints |
| Helper | `branchTakenMux` | def | Definitional | Fixes the root-lane `BranchTakenMux` formula |
| Boundary | `TakenTargetAlignmentBound` | def | Definitional | Fixes the authenticated taken-target alignment obligation |
| Boundary | `MulUNoOverflowBound` | def | Definitional | Fixes the authenticated unsigned DIV/REM no-overflow support relation |
| Boundary | `Stage1SupportBound` | def | Definitional | Packages Stage-1 target-alignment and unsigned no-overflow support |
| Boundary | `Stage1LinkageBound` | def | Definitional | Exact linkage from authenticated Stage-1 row/results back to lane and handoff |
| Theorem | `takenTargetAlignmentBound_of_stage1Support` | theorem | Theorem-Target | Extracts the target-alignment discharge from the Stage-1 support package |
| Theorem | `mulUNoOverflowBound_of_stage1Support` | theorem | Theorem-Target | Extracts the unsigned no-overflow relation from the Stage-1 support package |
| Package | `ExecutionRowProofPackage` | structure | Definitional | Carries one accepted execution row together with its slot-usage, support, and linkage package |
| Theorem | `stage1LinkageBound_of_executionRow` | theorem | Theorem-Target | Extracts the exact Stage-1 linkage batch from an accepted execution-row package |
| Theorem | `takenTargetAlignmentBound_of_executionRow` | theorem | Theorem-Target | Extracts the authenticated taken-target alignment discharge from an accepted execution-row package |
| Theorem | `mulUNoOverflowBound_of_executionRow` | theorem | Theorem-Target | Extracts the authenticated unsigned no-overflow support relation from an accepted execution-row package |
| Projection | `executionRowProjection` | def | Definitional | Concrete CE projection for execution-row linkage |
| Projection | `aluSubtableProjection` | def | Definitional | Concrete Shout projection for ALU subtable lookup owners |
| Projection | `branchConditionProjection` | def | Definitional | Concrete Shout projection for branch-condition owners |
| Theorem | `executionRowProjection_decide_eq_mergeMain_iff` | theorem | Theorem-Target | Execution-row CE claims merge only when they are the concrete main family |
| Theorem | `executionRowProjection_decide_eq_foldSeparate_of_supported_not_main` | theorem | Theorem-Target | Supported non-main execution-row CE claims fold separately |
| Theorem | `executionRowProjection_decide_eq_exportFinal_of_unsupported_not_main` | theorem | Theorem-Target | Unsupported non-main execution-row CE claims remain final |
| Theorem | `aluSubtableProjection_not_mainLane` | theorem | Theorem-Target | ALU subtable owners never enter the main lane directly |
| Theorem | `branchConditionProjection_not_mainLane` | theorem | Theorem-Target | Branch-condition owners never enter the main lane directly |

## Proof Obligations

- Dense slot packing is canonical and not prover-chosen.
- Stage-1 linkage is an exact equality package from authenticated Stage-1
  objects back to the lane and handoff columns.
- Stage-1 support obligations are owned explicitly rather than being left as
  prose attached to the linkage batch.
- `BranchTakenMux` is derived from authenticated `isBranch` and
  authenticated `branchTaken`, not from extra prover-side freedom.
- Taken `JAL` / branch / `JALR` targets are aligned by authenticated arithmetic
  obligations, not by unchecked witness convention.
- The unsigned DIV/REM no-overflow guard is part of the Stage-1 execution-row
  support surface.
- The execution owner split follows the protocol boundary:
  one CE owner for row linkage and two Shout owners for lookup channels.

## Out of Scope

- bytecode entrypoint/successor law
- Stage-2 register/RAM histories
- final ISA-equivalence for the enclosing committed sequence
