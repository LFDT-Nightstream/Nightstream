import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcChallenge.ChallengeWiring
import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcChallenge.Generated.SamplerLayoutData

/-!
Stable facade for the emitted three-matrix diagnostic PiRLC sampler layout.

Owns: typed affine row/column locations and fixed profile counts exported by
the generated sampler-layout artifact. Does not own row satisfaction, sampler
semantics, transcript authority, projection soundness, or row removal.

Assurance tier: artifact-checked structural metadata after the Rust drift gate.

| Surface | Fixed profile | Structural boundary |
|---|---:|---|
| scalar/block/lane/candidate/output counts | `15 / 4 / 16 / 64 / 54` | indices only |
| canonical/lane/residual/tail rows | `69 / 173 / 104 / 2599` | composite locations and extents only |
| selected outputs | `15 x 54` | exact columns shared with `ChallengeWiring` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.SamplerLayout

namespace Data

open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallengeSamplerLayoutData

abbrev initialCountBase : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.initialCountBase
abbrev initialCountStride : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.initialCountStride
abbrev initializationRowBase : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.initializationRowBase
abbrev initializationRowStride : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.initializationRowStride
abbrev selectionZeroBase : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.selectionZeroBase
abbrev selectionZeroStride : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.selectionZeroStride
abbrev selectionZeroRowBase : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.selectionZeroRowBase
abbrev selectionZeroRowStride : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.selectionZeroRowStride
abbrev fieldBase : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.fieldBase
abbrev fieldRhoStride : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.fieldRhoStride
abbrev fieldBlockStride : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.fieldBlockStride
abbrev fieldLaneStride : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.fieldLaneStride
abbrev bitStartBase : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.bitStartBase
abbrev bitStartRhoStride : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.bitStartRhoStride
abbrev bitStartBlockStride : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.bitStartBlockStride
abbrev bitStartLaneStride : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.bitStartLaneStride
abbrev canonicalRowBase : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.canonicalRowBase
abbrev canonicalRowRhoStride : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.canonicalRowRhoStride
abbrev canonicalRowBlockStride : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.canonicalRowBlockStride
abbrev canonicalRowLaneStride : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.canonicalRowLaneStride
abbrev tailFirstAllocatedBase : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.tailFirstAllocatedBase
abbrev tailFirstAllocatedStride : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.tailFirstAllocatedStride
abbrev tailRowBase : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.tailRowBase
abbrev tailRowStride : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.tailRowStride
abbrev outputOffset : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.outputOffset
abbrev outputStride : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.outputStride

end Data

abbrev scalarCount : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.scalarCount
abbrev digestBlockCount : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.digestBlockCount
abbrev laneCount : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.laneCount
abbrev lanesPerBlock : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.lanesPerBlock
abbrev chunksPerLane : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.chunksPerLane
abbrev candidateCount : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.candidateCount
abbrev outputCount : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.outputCount

abbrev canonicalRowCount : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.canonicalRows
abbrev laneRowCount : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.laneRows
abbrev samplerResidualRowCount : Nat := laneRowCount - canonicalRowCount
abbrev tailRowCount : Nat :=
  FPrimeRecursivePiRlcChallengeSamplerLayoutData.tailRows

def initialCountColumn (rho : Fin scalarCount) : Nat :=
  Data.initialCountBase + Data.initialCountStride * rho.val

def initializationRow (rho : Fin scalarCount) : Nat :=
  Data.initializationRowBase + Data.initializationRowStride * rho.val

def selectionZeroColumn (rho : Fin scalarCount) : Nat :=
  Data.selectionZeroBase + Data.selectionZeroStride * rho.val

def selectionZeroRow (rho : Fin scalarCount) : Nat :=
  Data.selectionZeroRowBase + Data.selectionZeroRowStride * rho.val

def fieldColumn
    (rho : Fin scalarCount) (block : Fin digestBlockCount)
    (lane : Fin lanesPerBlock) : Nat :=
  Data.fieldBase + Data.fieldRhoStride * rho.val +
    Data.fieldBlockStride * block.val + Data.fieldLaneStride * lane.val

private def bitStartAtNat (rho : Fin scalarCount) (block lane : Nat) : Nat :=
  Data.bitStartBase + Data.bitStartRhoStride * rho.val +
    Data.bitStartBlockStride * block + Data.bitStartLaneStride * lane

def bitStart
    (rho : Fin scalarCount) (block : Fin digestBlockCount)
    (lane : Fin lanesPerBlock) : Nat :=
  bitStartAtNat rho block.val lane.val

def tailBitStarts (rho : Fin scalarCount) : List Nat :=
  (List.ofFn fun block : Fin digestBlockCount =>
    List.ofFn fun lane : Fin lanesPerBlock => bitStart rho block lane).flatten

def predecessorColumn
    (rho : Fin scalarCount) (block : Fin digestBlockCount)
    (lane : Fin lanesPerBlock) : Nat :=
  if block.val = 0 ∧ lane.val = 0 then
    initialCountColumn rho
  else if lane.val = 0 then
    bitStartAtNat rho (block.val - 1) (lanesPerBlock - 1) + 157
  else
    bitStartAtNat rho block.val (lane.val - 1) + 157

def canonicalRow
    (rho : Fin scalarCount) (block : Fin digestBlockCount)
    (lane : Fin lanesPerBlock) : Nat :=
  Data.canonicalRowBase + Data.canonicalRowRhoStride * rho.val +
    Data.canonicalRowBlockStride * block.val + Data.canonicalRowLaneStride * lane.val

def laneResidualRow
    (rho : Fin scalarCount) (block : Fin digestBlockCount)
    (lane : Fin lanesPerBlock) : Nat :=
  canonicalRow rho block lane + canonicalRowCount

def tailFirstAllocated (rho : Fin scalarCount) : Nat :=
  Data.tailFirstAllocatedBase + Data.tailFirstAllocatedStride * rho.val

def tailRow (rho : Fin scalarCount) : Nat :=
  Data.tailRowBase + Data.tailRowStride * rho.val

def outputColumn (rho : Fin scalarCount) (output : Fin outputCount) : Nat :=
  tailFirstAllocated rho + Data.outputOffset + Data.outputStride * output.val

def outputColumnsFor (rho : Fin scalarCount) : List Nat :=
  List.ofFn fun output : Fin outputCount => outputColumn rho output

def outputColumns : List (List Nat) :=
  List.ofFn outputColumnsFor

def StructureValid : Prop :=
  (scalarCount = 15 ∧
    digestBlockCount = 4 ∧
    laneCount = 16 ∧
    lanesPerBlock = 4 ∧
    chunksPerLane = 4 ∧
    candidateCount = 64 ∧
    outputCount = 54 ∧
    canonicalRowCount = 69 ∧
    laneRowCount = 173 ∧
    samplerResidualRowCount = 104 ∧
    tailRowCount = 2599) ∧
  (∀ rho : Fin scalarCount,
    initialCountColumn rho ≠ selectionZeroColumn rho ∧
    selectionZeroColumn rho = tailFirstAllocated rho + 5 ∧
    selectionZeroRow rho = tailRow rho + 6) ∧
  outputColumns = ChallengeWiring.samplerOutputColumns

instance : Decidable StructureValid := by
  unfold StructureValid
  infer_instance

theorem structure_check : StructureValid := by
  set_option maxRecDepth 100000 in
    decide

theorem zero_columns_distinct (rho : Fin scalarCount) :
    initialCountColumn rho ≠ selectionZeroColumn rho :=
  (structure_check.2.1 rho).1

theorem selection_zero_column_eq_tail_first_allocated
    (rho : Fin scalarCount) :
    selectionZeroColumn rho = tailFirstAllocated rho + 5 :=
  (structure_check.2.1 rho).2.1

theorem selection_zero_row_eq_tail_row (rho : Fin scalarCount) :
    selectionZeroRow rho = tailRow rho + 6 :=
  (structure_check.2.1 rho).2.2

theorem output_columns_match_challenge_wiring :
    outputColumns = ChallengeWiring.samplerOutputColumns :=
  structure_check.2.2

end Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.SamplerLayout
