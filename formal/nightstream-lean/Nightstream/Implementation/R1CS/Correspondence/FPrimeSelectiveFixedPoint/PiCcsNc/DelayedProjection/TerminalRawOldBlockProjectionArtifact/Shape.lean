import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact.Layout

/-!
Kernel proof of the fixed production raw-old-block projection shape.

This leaf proves the compact eighteen-round prefix-tensor shape, the explicit
nineteenth-coordinate factor association, and the exact optimized profile
arithmetic without constructing the 24,185,169 rows or any
production-sized list.  Generated tensor formulas remain data; the proofs
below establish their compiler meaning symbolically.

Owns: exact production profile arithmetic, tensor-level cardinalities,
canonical coefficient bounds, rectangle coverage, and the total symbolic row
count used by subsequent ownership proofs.

Does not own: detailed tensor trace agreement, physical row permutation,
runtime column placement, assignment values, row satisfaction, or semantic
acceptance.

Emits constraints: no.

| Stable stage path | Mathematical obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.projection_shape.profile` | radix 2, fourteen children, 54 active lanes, eighteen tensor variables, one factored variable, and 211,797 blocks match production | checked / derived |
| `f_prime.pi_ccs_nc.delayed.projection_shape.tensor` | eighteen tensor levels have the exact multiplication widths and canonical coefficients | derived |
| `f_prime.pi_ccs_nc.delayed.projection_shape.factor` | generated full-point coordinate 18 and final-scale outputs are associated with the optimized layout | direct dataflow |
| `f_prime.pi_ccs_nc.delayed.projection_shape.rows` | tensor, child-coordinate, final-scale, and terminal multiplicities produce the exact optimized total | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt

@[simp] theorem productionRadix : productionLayout.radix = 2 := by
  rfl

@[simp] theorem productionChildCount : productionLayout.childCount = 14 := by
  rfl

@[simp] theorem productionActiveLanes :
    productionLayout.activeLanes = 54 := by
  rfl

@[simp] theorem productionLogicalWidth :
    productionLayout.logicalWidth = 11437038 := by
  rfl

@[simp] theorem productionBlockVariables :
    productionLayout.blockVariables = 18 := by
  rfl

@[simp] theorem productionFullBlockVariables :
    Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionPlan.blockVariables = 19 := by
  rfl

@[simp] theorem productionFactoredBase :
    productionFactoredLayout.base = productionLayout := by
  rfl

@[simp] theorem productionFactorEnabled :
    productionFactoredLayout.factor.factorFinalRound = true := by
  rfl

@[simp] theorem productionTensorVariables :
    productionFactoredLayout.factor.tensorVariables = 18 := by
  rfl

@[simp] theorem productionFactoredVariable :
    productionFactoredLayout.factor.factoredVariable = 18 := by
  rfl

@[simp] theorem productionPrefixPointColumn
    (round : Fin productionLayout.blockVariables) :
    productionLayout.oldBlock round =
      productionFactoredLayout.factor.fullOldBlock round.val := by
  rfl

@[simp] theorem productionFullOldBlockColumn (round : Nat) :
    productionFactoredLayout.factor.fullOldBlock round =
      oldBlockColumnsNat round := by
  rfl

@[simp] theorem productionFinalPointColumn :
    productionFactoredLayout.factor.finalPoint = oldBlockColumnsNat 18 := by
  rfl

@[simp] theorem productionFinalScaleTrace
    (lane : Fin productionFactoredLayout.base.activeLanes) :
    productionFactoredLayout.scale lane = finalScaleTrace lane.val := by
  rfl

@[simp] theorem productionFinalScaleOutput
    (lane : Fin productionFactoredLayout.base.activeLanes) :
    (productionFactoredLayout.scale lane).output =
      finalScaleOutput lane.val := by
  rfl

@[simp] theorem productionProductFirst :
    productionLayout.productFirst = productFirstColumn := by
  rfl

@[simp] theorem productionChildWitnessFirst
    (child : Fin productionLayout.childCount) :
    productionLayout.childWitnessFirst child = childWitnessFirst child := by
  rfl

theorem productionBlockCount : blockCount productionLayout = 211797 := by
  rfl

theorem productionPositiveLanes : 0 < productionLayout.activeLanes := by
  decide

theorem productionPositiveBlocks : 0 < blockCount productionLayout := by
  rw [productionBlockCount]
  decide

theorem productionRectangle :
    productionLayout.logicalWidth =
      blockCount productionLayout * productionLayout.activeLanes := by
  rfl

theorem productionLevelCount :
    productionLayout.tensorLevels.length =
      productionLayout.blockVariables := by
  rfl

@[simp] theorem productionTensorMultiplicationCount :
    tensorMultiplicationCount productionLayout = 262143 := by
  rfl

@[simp] theorem productionFactoredTensorMultiplicationCount :
    Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionFinalScaleCompiler.tensorMultiplicationCount
      productionFactoredLayout = 262143 := by
  rfl

@[simp] theorem productionPrefixRowCount :
    rowCount productionLayout = 24184899 := by
  rfl

theorem productionRowCount :
    Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionFinalScaleCompiler.rowCount
      productionFactoredLayout = totalRows := by
  rfl

@[simp] theorem productionRowCount_exact :
    Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionFinalScaleCompiler.rowCount
      productionFactoredLayout = 24185169 := by
  rfl

theorem productionBlocksFitPrefix :
    blockCount productionLayout <= 2 ^ productionLayout.blockVariables := by
  rw [productionBlockCount, productionBlockVariables]
  decide

private theorem primitiveNatMin_eq_left {left right : Nat}
    (le : left ≤ right) : Nat.min left right = left := by
  exact (Nat.min_eq_min left).trans (Nat.min_eq_left le)

private theorem primitiveNatMin_eq_right {left right : Nat}
    (le : right ≤ left) : Nat.min left right = right := by
  exact (Nat.min_eq_min left).trans (Nat.min_eq_right le)

/-- A compact-prefix tensor round doubles its addressable prefix, capped at
the production block count.  The right summand is exactly the live high
half.  Keeping this arithmetic fact independent of the generated schedule
prevents elaboration from unfolding the generated row program. -/
theorem natMin_doublePrefix (blocks power : Nat) :
    Nat.min blocks (power * 2) =
      Nat.min blocks power + Nat.min (blocks - power) power := by
  by_cases capped : blocks ≤ power
  · have cappedTwice : blocks ≤ power * 2 := by omega
    calc
      Nat.min blocks (power * 2) = blocks :=
        primitiveNatMin_eq_left cappedTwice
      _ = blocks + 0 := by omega
      _ = Nat.min blocks power + Nat.min (blocks - power) power := by
        rw [primitiveNatMin_eq_left capped, Nat.sub_eq_zero_of_le capped]
        simp
  · have powerLt : power < blocks := Nat.lt_of_not_ge capped
    by_cases fits : blocks ≤ power * 2
    · have remainderLe : blocks - power ≤ power := by omega
      calc
        Nat.min blocks (power * 2) = blocks :=
          primitiveNatMin_eq_left fits
        _ = power + (blocks - power) := by omega
        _ = Nat.min blocks power + Nat.min (blocks - power) power := by
          rw [primitiveNatMin_eq_right (Nat.le_of_lt powerLt),
            primitiveNatMin_eq_left remainderLe]
    · have doubledLe : power * 2 ≤ blocks := Nat.le_of_not_ge fits
      have powerLeRemainder : power ≤ blocks - power := by omega
      calc
        Nat.min blocks (power * 2) = power * 2 :=
          primitiveNatMin_eq_right doubledLe
        _ = power + power := by omega
        _ = Nat.min blocks power + Nat.min (blocks - power) power := by
          rw [primitiveNatMin_eq_right (Nat.le_of_lt powerLt),
            primitiveNatMin_eq_right powerLeRemainder]

theorem generatedTrace_definitions_canonical
    (trace : KMulTrace) (definition : Definition)
    (member : definition ∈ trace.definitions) :
    definition.Canonical := by
  simp only [KMulTrace.definitions, List.mem_cons, List.not_mem_nil,
    or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl <;>
    simp [Definition.Canonical, CanonicalTerms, goldilocksP]

theorem generatedTensorTrace_sumLayout
    (round parent : Nat) :
    (Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.tensorTrace
      round parent).SumLayoutValid := by
  constructor <;>
    simp [Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.tensorTrace]

theorem productionFinalScaleDefinitionsCanonical
    (lane : Fin productionFactoredLayout.base.activeLanes)
    (definition : Definition)
    (member : definition ∈
      (productionFactoredLayout.scale lane).definitions) :
    definition.Canonical := by
  exact generatedTrace_definitions_canonical
    (finalScaleTrace lane.val) definition (by simpa using member)

theorem productionFinalScaleTrace_sumLayout
    (lane : Fin productionFactoredLayout.base.activeLanes) :
    (productionFactoredLayout.scale lane).SumLayoutValid := by
  constructor <;> simp [productionFactoredLayout, finalScaleTrace]

theorem productionRawCoefficientsCanonical :
    forall child, child < productionLayout.childCount ->
      0 < radixCoefficient productionLayout child /\
        radixCoefficient productionLayout child < goldilocksP := by
  intro child childInRange
  have childLt : child < 14 := by simpa [productionLayout] using childInRange
  have exponentLe : child ≤ 13 := by omega
  have powerLe : 2 ^ child ≤ 2 ^ 13 :=
    Nat.pow_le_pow_right (by decide) exponentLe
  have powerLt : 2 ^ child < goldilocksP :=
    Nat.lt_of_le_of_lt powerLe (by decide)
  have powerPositive : 0 < 2 ^ child := Nat.pow_pos (by decide)
  change 0 < 2 ^ child % goldilocksP /\
    2 ^ child % goldilocksP < goldilocksP
  rw [Nat.mod_eq_of_lt powerLt]
  exact ⟨powerPositive, powerLt⟩

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact
