import Nightstream.Implementation.R1CS.Correspondence.Gadgets.TerminalRawOldBlockProjectionTensorPrefix
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact.Shape

/-!
Model-level optimization certificate for the final production Boolean-tensor
round.

All 211,797 production blocks lie below `2^18`, so bit 18 is uniformly low.
The generic prefix theorem therefore moves its common `(1 - oldBlock[18])`
factor from one K multiplication per block to one K multiplication per active
lane.

Owns: the fixed-profile all-low bound, semantic weighted-fold factorization,
and exact before/after row and derived-column arithmetic.

Does not own: emitted-row coefficients, artifact drift, Rust conformance,
physical ownership after rewriting, or permission to remove rows without the
separate artifact-refinement leaves.

Emits constraints: no; the exact counts below compare the historical direct
nineteenth round with the generated factorized compiler.

| Stable stage path | Mathematical obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.raw_old_block.final_round.domain` | every production block index is below `2^18`, so the nineteenth Boolean selector is uniformly low | computed |
| `f_prime.pi_ccs_nc.delayed.raw_old_block.final_round.factor` | move the common `(1 - oldBlock[18])` factor from every block term to one multiplication per active lane | derived |
| `f_prime.pi_ccs_nc.delayed.raw_old_block.final_round.cost` | reconcile the direct and factored row/derived-column counts and their exact difference | computed |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler
open Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionTensorPrefix

theorem productionBlockCount_lt_twoPow18 :
    blockCount productionLayout < 2 ^ 18 := by
  rw [productionBlockCount]
  decide

/-- Semantic equivalence underlying the proposed rewrite: the nineteenth
Boolean coordinate is common to every production block and is applied once
after the round-18 weighted projection. -/
theorem productionWeightedPrefixProjection_factorFinalRound
    (coordinates : List K) (value : Nat -> K) :
    weightedPrefixProjection coordinates 19
        (List.range (blockCount productionLayout)) value =
      K.mul
        (weightedPrefixProjection coordinates 18
          (List.range (blockCount productionLayout)) value)
        (K.sub K.one (coordinates.getD 18 K.zero)) := by
  have allLow : forall index,
      index ∈ List.range (blockCount productionLayout) -> index < 2 ^ 18 := by
    intro index member
    exact Nat.lt_trans (List.mem_range.mp member)
      productionBlockCount_lt_twoPow18
  simpa using weightedPrefixProjection_succ_allLow coordinates 18
    (List.range (blockCount productionLayout)) value allLow

/-- Historical direct-round rows: the current 18-round prefix plus one
five-row multiplication per block, followed by coordinate and terminal rows. -/
def productionBaselineRows : Nat :=
  5 * (tensorMultiplicationCount productionLayout +
      blockCount productionLayout) +
    2 * productionLayout.logicalWidth + 2 * productionLayout.activeLanes

/-- Historical direct-round allocated/committed derived columns: five per
prefix or final-round tensor multiplication and two per raw-coordinate
product. -/
def productionBaselineAllocatedColumns : Nat :=
  5 * (tensorMultiplicationCount productionLayout +
      blockCount productionLayout) +
    2 * productionLayout.logicalWidth

/-- Generated rows after replacing the last per-block tensor round by one
five-row K multiplication for each active lane. -/
def productionFactoredFinalRoundRows : Nat :=
  Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionFinalScaleCompiler.rowCount
    productionFactoredLayout

/-- Proposed compiler-local allocated/committed derived columns under the
same replacement. -/
def productionFactoredFinalRoundAllocatedColumns : Nat :=
  5 * tensorMultiplicationCount productionLayout +
    2 * productionLayout.logicalWidth + 5 * productionLayout.activeLanes

theorem productionBaselineRows_exact :
    productionBaselineRows = 25243884 := by
  unfold productionBaselineRows
  rw [productionTensorMultiplicationCount, productionBlockCount,
    productionLogicalWidth, productionActiveLanes]

theorem productionBaselineAllocatedColumns_exact :
    productionBaselineAllocatedColumns = 25243776 := by
  unfold productionBaselineAllocatedColumns
  rw [productionTensorMultiplicationCount, productionBlockCount,
    productionLogicalWidth]

theorem productionFactoredFinalRoundRows_exact :
    productionFactoredFinalRoundRows = 24185169 := by
  unfold productionFactoredFinalRoundRows
  exact productionRowCount_exact

theorem productionFactoredFinalRoundAllocatedColumns_exact :
    productionFactoredFinalRoundAllocatedColumns = 24185061 := by
  unfold productionFactoredFinalRoundAllocatedColumns
  rw [productionTensorMultiplicationCount, productionLogicalWidth,
    productionActiveLanes]

theorem productionFactoredFinalRoundRowSavings_exact :
    productionBaselineRows - productionFactoredFinalRoundRows =
      1058715 := by
  rw [productionBaselineRows_exact,
    productionFactoredFinalRoundRows_exact]

theorem productionFactoredFinalRoundAllocatedColumnSavings_exact :
    productionBaselineAllocatedColumns -
        productionFactoredFinalRoundAllocatedColumns = 1058715 := by
  rw [productionBaselineAllocatedColumns_exact,
    productionFactoredFinalRoundAllocatedColumns_exact]

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact
