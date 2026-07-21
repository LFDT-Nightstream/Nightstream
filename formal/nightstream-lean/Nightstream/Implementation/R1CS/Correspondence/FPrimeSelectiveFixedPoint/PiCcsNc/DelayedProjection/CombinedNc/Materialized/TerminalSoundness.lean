import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.TerminalProgram

/-!
Kernel interpretation of the fixed production combined-NC terminal program.

Owns: semantic consequences of the exact straight-line schedule and its final
two equality rows. Does not own generated-row equality, padding truth,
transcript ordering, witness authority, commitment binding, or row removal.
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.TerminalProgram

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc

/-- Independent executable terminal expression at the last three arithmetic
nodes.  The preceding trace owners separately compute its ordinary and
delayed inputs from the verifier-owned boundary columns. -/
def terminalExpression (assignment : Nat → Nat) : K :=
  K.add (ordinaryProduct.output.value assignment)
    (delayedProduct.output.value assignment)

def ordinaryExpression (assignment : Nat → Nat) : K :=
  K.mul
    (K.mul (blockEquality.output.value assignment)
      (laneEquality.output.value assignment))
    (ordinarySum.output.value assignment)

def delayedExpression (assignment : Nat → Nat) : K :=
  K.mul
    (K.mul
      (K.mul (batchWeightColumns.value assignment)
        (oldBlockEquality.output.value assignment))
      (selectorOutput.value assignment))
    (runningSum.output.value assignment)

def fullTerminalExpression (assignment : Nat → Nat) : K :=
  K.add (ordinaryExpression assignment) (delayedExpression assignment)

/-- Exact semantic consequences needed by the parent bridge.  `definitions`
is the complete straight-line computation; neither the RHS value nor the
final SumCheck equality is accepted from the caller. -/
structure Computed (assignment : Nat → Nat) : Prop where
  definitionsHold : DefinitionsHold assignment definitions
  chiInitial : chiInitial.output.value assignment = oneTerms.value assignment
  chiLayers : ∀ layer ∈ chiLayers,
    layer.output.map (fun output => output.value assignment) =
      layer.expected assignment
  outputs : ∀ trace ∈ outputTraces, trace.Computed assignment
  gammaPowerValues : gammaPowers.powers.map
      (fun power => power.value assignment) =
    K.powersFrom (gammaColumns.value assignment) K.one outputCount
  ordinarySumValue : ordinarySum.output.value assignment =
    dotValue gammaPowers.powers outputResiduals assignment
  blockEqualityComputed : blockEquality.Computed assignment
  laneEqualityComputed : laneEquality.Computed assignment
  radixConstantValue : radixConstant.output.value assignment =
    baseTwoTerms.value assignment
  radixPowerValues : radixPowers.powers.map
      (fun power => power.value assignment) =
    K.powersFrom (radixConstant.output.value assignment) K.one
      runningValues.length
  runningSumValue : runningSum.output.value assignment =
    dotValue radixPowers.powers runningValues assignment
  oldBlockEqualityComputed : oldBlockEquality.Computed assignment
  selectorInitialValue : selectorInitial.output.value assignment =
    oneTerms.value assignment
  selectorStepsComputed : ∀ trace ∈ selectorSteps, trace.Computed assignment
  rhs : terminalRhsColumns.value assignment = terminalExpression assignment
  fullRhs : terminalRhsColumns.value assignment =
    fullTerminalExpression assignment
  final : finalSumColumns.value assignment = terminalRhsColumns.value assignment

private theorem definitionsHold_append_left
    {assignment : Nat → Nat} {left right : List Definition}
    (holds : DefinitionsHold assignment (left ++ right)) :
    DefinitionsHold assignment left :=
  fun definition member =>
    holds definition (List.mem_append_left right member)

private theorem definitionsHold_append_right
    {assignment : Nat → Nat} {left right : List Definition}
    (holds : DefinitionsHold assignment (left ++ right)) :
    DefinitionsHold assignment right :=
  fun definition member =>
    holds definition (List.mem_append_right left member)

private theorem prefixDefinitionsHold
    {assignment : Nat → Nat}
    (holds : DefinitionsHold assignment definitions) :
    DefinitionsHold assignment prefixDefinitions := by
  have split := holds
  rw [definitions_eq_prefix_append_final] at split
  exact definitionsHold_append_left split

private theorem laneComputationDefinitionsHold
    {assignment : Nat → Nat}
    (holds : DefinitionsHold assignment definitions) :
    DefinitionsHold assignment laneComputationDefinitions := by
  have split := prefixDefinitionsHold holds
  rw [prefixDefinitions_eq_lane_append_suffix] at split
  exact definitionsHold_append_left split

private theorem suffixDefinitionsHold
    {assignment : Nat → Nat}
    (holds : DefinitionsHold assignment definitions) :
    DefinitionsHold assignment suffixDefinitions := by
  have split := prefixDefinitionsHold holds
  rw [prefixDefinitions_eq_lane_append_suffix] at split
  exact definitionsHold_append_right split

private theorem ordinaryDefinitionsHold
    {assignment : Nat → Nat}
    (holds : DefinitionsHold assignment definitions) :
    DefinitionsHold assignment ordinaryDefinitions := by
  have split := suffixDefinitionsHold holds
  rw [suffixDefinitions_eq_ordinary_append_delayed] at split
  exact definitionsHold_append_left split

private theorem delayedDefinitionsHold
    {assignment : Nat → Nat}
    (holds : DefinitionsHold assignment definitions) :
    DefinitionsHold assignment delayedDefinitions := by
  have split := suffixDefinitionsHold holds
  rw [suffixDefinitions_eq_ordinary_append_delayed] at split
  exact definitionsHold_append_right split

private theorem delayedPreSelectorDefinitionsHold
    {assignment : Nat → Nat}
    (holds : DefinitionsHold assignment definitions) :
    DefinitionsHold assignment delayedPreSelectorDefinitions := by
  have split := delayedDefinitionsHold holds
  rw [delayedDefinitions_eq_segments] at split
  intro definition member
  apply split definition
  rw [List.mem_append]
  apply Or.inl
  rw [List.mem_append]
  exact Or.inl member

private theorem delayedSelectorDefinitionsHold
    {assignment : Nat → Nat}
    (holds : DefinitionsHold assignment definitions) :
    DefinitionsHold assignment
      (selectorSteps.flatMap SelectorStep.definitions) := by
  have split := delayedDefinitionsHold holds
  rw [delayedDefinitions_eq_segments] at split
  intro definition member
  apply split definition
  rw [List.mem_append]
  apply Or.inl
  rw [List.mem_append]
  exact Or.inr member

private theorem delayedPostSelectorDefinitionsHold
    {assignment : Nat → Nat}
    (holds : DefinitionsHold assignment definitions) :
    DefinitionsHold assignment delayedPostSelectorDefinitions := by
  have split := delayedDefinitionsHold holds
  rw [delayedDefinitions_eq_segments] at split
  intro definition member
  apply split definition
  rw [List.mem_append]
  exact Or.inr member

theorem outputTraces_computed
    {assignment : Nat → Nat}
    (holds : DefinitionsHold assignment definitions) :
    ∀ trace ∈ outputTraces, trace.Computed assignment := by
  have laneSegment := laneComputationDefinitionsHold holds
  have outputSegment : DefinitionsHold assignment outputDefinitions := by
    intro definition member
    apply laneSegment definition
    simp [laneComputationDefinitions, member]
  intro trace traceMember
  apply trace.sound assignment
  intro definition member
  apply outputSegment definition
  exact List.mem_flatMap.mpr ⟨trace, traceMember, member⟩

theorem chiInitial_computed
    {assignment : Nat → Nat}
    (holds : DefinitionsHold assignment definitions) :
    chiInitial.output.value assignment = oneTerms.value assignment := by
  have laneSegment := laneComputationDefinitionsHold holds
  apply chiInitial.sound assignment
  intro definition member
  apply laneSegment definition
  simp [laneComputationDefinitions, chiDefinitions, member]

theorem chiLayers_computed
    {assignment : Nat → Nat}
    (holds : DefinitionsHold assignment definitions) :
    ∀ layer ∈ chiLayers,
      layer.output.map (fun output => output.value assignment) =
        layer.expected assignment := by
  have laneSegment := laneComputationDefinitionsHold holds
  have chiSegment : DefinitionsHold assignment chiDefinitions := by
    intro definition member
    apply laneSegment definition
    simp [laneComputationDefinitions, member]
  intro layer layerMember
  apply layer.sound assignment
  intro definition member
  apply chiSegment definition
  unfold chiDefinitions
  exact List.mem_append_right chiInitial.definitions
    (List.mem_flatMap.mpr ⟨layer, layerMember, member⟩)

theorem gammaPowers_computed
    {assignment : Nat → Nat}
    (constantOne : assignment 0 = 1)
    (holds : DefinitionsHold assignment definitions) :
    gammaPowers.powers.map (fun power => power.value assignment) =
      K.powersFrom (gammaColumns.value assignment) K.one outputCount := by
  have laneSegment := laneComputationDefinitionsHold holds
  have powerHolds : DefinitionsHold assignment gammaPowers.definitions := by
    intro definition member
    apply laneSegment definition
    simp [laneComputationDefinitions, member]
  exact gammaPowers.sound assignment constantOne (by decide)
    (gammaPowers.layout (by decide)) powerHolds

theorem ordinarySum_computed
    {assignment : Nat → Nat}
    (holds : DefinitionsHold assignment definitions) :
    ordinarySum.output.value assignment =
      dotValue gammaPowers.powers outputResiduals assignment := by
  have laneSegment := laneComputationDefinitionsHold holds
  apply ordinarySum.sound assignment
  intro definition member
  apply laneSegment definition
  simp [laneComputationDefinitions, member]

theorem blockEquality_computed
    {assignment : Nat → Nat}
    (holds : DefinitionsHold assignment definitions) :
    blockEquality.Computed assignment := by
  have ordinarySegment := ordinaryDefinitionsHold holds
  apply blockEquality.sound_of_first assignment blockEqualityFirst
    blockEquality_first
  intro definition member
  apply ordinarySegment definition
  simp [ordinaryDefinitions, member]

theorem laneEquality_computed
    {assignment : Nat → Nat}
    (holds : DefinitionsHold assignment definitions) :
    laneEquality.Computed assignment := by
  have ordinarySegment := ordinaryDefinitionsHold holds
  apply laneEquality.sound_of_first assignment laneEqualityFirst
    laneEquality_first
  intro definition member
  apply ordinarySegment definition
  simp [ordinaryDefinitions, member]

theorem radixConstant_computed
    {assignment : Nat → Nat}
    (holds : DefinitionsHold assignment definitions) :
    radixConstant.output.value assignment = baseTwoTerms.value assignment := by
  have delayedSegment := delayedPreSelectorDefinitionsHold holds
  apply radixConstant.sound assignment
  intro definition member
  apply delayedSegment definition
  simp [delayedPreSelectorDefinitions, member]

theorem radixPowers_computed
    {assignment : Nat → Nat}
    (constantOne : assignment 0 = 1)
    (holds : DefinitionsHold assignment definitions) :
    radixPowers.powers.map (fun power => power.value assignment) =
      K.powersFrom (radixConstant.output.value assignment) K.one
        runningValues.length := by
  have delayedSegment := delayedPreSelectorDefinitionsHold holds
  have powerHolds : DefinitionsHold assignment radixPowers.definitions := by
    intro definition member
    apply delayedSegment definition
    simp [delayedPreSelectorDefinitions, member]
  have countPositive : 0 < radixPowers.count := by
    simp [radixPowers, runningValues_length, outputCount]
  exact radixPowers.sound assignment constantOne countPositive
    (radixPowers.layout countPositive) powerHolds

theorem runningSum_computed
    {assignment : Nat → Nat}
    (holds : DefinitionsHold assignment definitions) :
    runningSum.output.value assignment =
      dotValue radixPowers.powers runningValues assignment := by
  have delayedSegment := delayedPreSelectorDefinitionsHold holds
  apply runningSum.sound assignment
  intro definition member
  apply delayedSegment definition
  simp [delayedPreSelectorDefinitions, member]

theorem oldBlockEquality_computed
    {assignment : Nat → Nat}
    (holds : DefinitionsHold assignment definitions) :
    oldBlockEquality.Computed assignment := by
  have delayedSegment := delayedPreSelectorDefinitionsHold holds
  apply oldBlockEquality.sound_of_first assignment oldBlockEqualityFirst
    oldBlockEquality_first
  intro definition member
  apply delayedSegment definition
  simp [delayedPreSelectorDefinitions, member]

theorem selectorInitial_computed
    {assignment : Nat → Nat}
    (holds : DefinitionsHold assignment definitions) :
    selectorInitial.output.value assignment = oneTerms.value assignment := by
  have delayedSegment := delayedPreSelectorDefinitionsHold holds
  apply selectorInitial.sound assignment
  intro definition member
  apply delayedSegment definition
  simp [delayedPreSelectorDefinitions, member]

theorem selectorSteps_computed
    {assignment : Nat → Nat}
    (holds : DefinitionsHold assignment definitions) :
    ∀ trace ∈ selectorSteps, trace.Computed assignment := by
  have delayedSegment := delayedSelectorDefinitionsHold holds
  intro trace traceMember
  apply trace.sound assignment
  intro definition member
  apply delayedSegment definition
  exact List.mem_flatMap.mpr ⟨trace, traceMember, member⟩

theorem ordinaryProduct_eq_expression
    {assignment : Nat → Nat}
    (holds : DefinitionsHold assignment definitions) :
    ordinaryProduct.output.value assignment = ordinaryExpression assignment := by
  have segment := ordinaryDefinitionsHold holds
  have equalityHolds : DefinitionsHold assignment equalityProduct.definitions := by
    intro definition member
    apply segment definition
    simp [ordinaryDefinitions, member]
  have ordinaryHolds : DefinitionsHold assignment ordinaryProduct.definitions := by
    intro definition member
    apply segment definition
    simp [ordinaryDefinitions, member]
  have equality := mulColumnsAt_sound laneEquality.next
    blockEquality.output laneEquality.output assignment equalityHolds
  have ordinary := mulColumnsAt_sound (mulNext equalityProduct)
    equalityProduct.output ordinarySum.output assignment ordinaryHolds
  have equality' : equalityProduct.output.value assignment =
      K.mul (blockEquality.output.value assignment)
        (laneEquality.output.value assignment) := by
    simpa [equalityProduct] using equality
  have ordinary' : ordinaryProduct.output.value assignment =
      K.mul (equalityProduct.output.value assignment)
        (ordinarySum.output.value assignment) := by
    simpa [ordinaryProduct] using ordinary
  unfold ordinaryExpression
  rw [ordinary', equality']

theorem delayedProduct_eq_expression
    {assignment : Nat → Nat}
    (holds : DefinitionsHold assignment definitions) :
    delayedProduct.output.value assignment = delayedExpression assignment := by
  have segment := delayedPostSelectorDefinitionsHold holds
  have oldHolds : DefinitionsHold assignment weightedOldEquality.definitions := by
    intro definition member
    apply segment definition
    simp [delayedPostSelectorDefinitions, member]
  have selectorHolds : DefinitionsHold assignment weightedSelector.definitions := by
    intro definition member
    apply segment definition
    simp [delayedPostSelectorDefinitions, member]
  have delayedHolds : DefinitionsHold assignment delayedProduct.definitions := by
    intro definition member
    apply segment definition
    simp [delayedPostSelectorDefinitions, member]
  have old := mulColumnsAt_sound selectorNext batchWeightColumns
    oldBlockEquality.output assignment oldHolds
  have selector := mulColumnsAt_sound (mulNext weightedOldEquality)
    weightedOldEquality.output selectorOutput assignment selectorHolds
  have delayed := mulColumnsAt_sound (mulNext weightedSelector)
    weightedSelector.output runningSum.output assignment delayedHolds
  have old' : weightedOldEquality.output.value assignment =
      K.mul (batchWeightColumns.value assignment)
        (oldBlockEquality.output.value assignment) := by
    simpa [weightedOldEquality] using old
  have selector' : weightedSelector.output.value assignment =
      K.mul (weightedOldEquality.output.value assignment)
        (selectorOutput.value assignment) := by
    simpa [weightedSelector] using selector
  have delayed' : delayedProduct.output.value assignment =
      K.mul (weightedSelector.output.value assignment)
        (runningSum.output.value assignment) := by
    simpa [delayedProduct] using delayed
  unfold delayedExpression
  rw [delayed', selector', old']

/-- Kernel interpretation of the final addition, assuming the exact
definition equations already derived from row satisfaction. -/
theorem terminalRhs_eq_expression
    {assignment : Nat → Nat}
    (holds : DefinitionsHold assignment definitions) :
    terminalRhsColumns.value assignment = terminalExpression assignment := by
  have finalHolds : DefinitionsHold assignment finalAddition.definitions := by
    intro definition member
    apply holds definition
    unfold definitions
    exact List.mem_append_right prefixDefinitions member
  have linear := finalAddition.sound assignment finalHolds
  calc
    terminalRhsColumns.value assignment =
        (addTerms ordinaryProduct.output delayedProduct.output).value
          assignment := by simpa [finalAddition] using linear
    _ = K.add (ordinaryProduct.output.value assignment)
        (delayedProduct.output.value assignment) :=
      addTerms_value ordinaryProduct.output delayedProduct.output assignment
    _ = terminalExpression assignment := rfl

/-- The last addition plus its two multiplication chains yields the literal
ordinary-plus-delayed Rust terminal expression. -/
theorem terminalRhs_eq_fullExpression
    {assignment : Nat → Nat}
    (holds : DefinitionsHold assignment definitions) :
    terminalRhsColumns.value assignment = fullTerminalExpression assignment := by
  rw [terminalRhs_eq_expression holds]
  unfold terminalExpression fullTerminalExpression
  rw [ordinaryProduct_eq_expression holds,
    delayedProduct_eq_expression holds]

/-- The two physical assertion rows bind the final SumCheck claim to the
computed terminal RHS. -/
theorem finalEquality_sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (satisfies : Satisfies finalEqualityRows assignment) :
    finalSumColumns.value assignment = terminalRhsColumns.value assignment := by
  have lowRow : RowHolds assignment
      (builderLinearRow finalSumColumns.c0 [(terminalRhsColumns.c0, 1)]) := by
    exact satisfies _ (by simp [finalEqualityRows])
  have highRow : RowHolds assignment
      (builderLinearRow finalSumColumns.c1 [(terminalRhsColumns.c1, 1)]) := by
    exact satisfies _ (by simp [finalEqualityRows])
  have low := builderLinearRow_sound canonical constantOne
    finalSumColumns.c0 [(terminalRhsColumns.c0, 1)] (by
      simp [CanonicalTerms, goldilocksP]) lowRow
  have high := builderLinearRow_sound canonical constantOne
    finalSumColumns.c1 [(terminalRhsColumns.c1, 1)] (by
      simp [CanonicalTerms, goldilocksP]) highRow
  simp only [KColumns.value, K.mk.injEq]
  constructor
  · apply Fin.ext
    simpa [baseAt, residue, lcEval] using
      congrArg (fun value => value % goldilocksP) low
  · apply Fin.ext
    simpa [baseAt, residue, lcEval] using
      congrArg (fun value => value % goldilocksP) high

/-- Satisfaction of the exact terminal rows yields the straight-line
definition semantics and the final equality.  Canonicality of the definition
stream is structural and remains an explicit premise until the generated-row
artifact discharges it in bounded shards. -/
theorem sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (definitionsCanonical :
      ∀ definition ∈ definitions, definition.Canonical)
    (satisfies : Satisfies rows assignment) :
    Computed assignment := by
  have identitySatisfies : Satisfies identityRows assignment := by
    intro row member
    exact satisfies row (List.mem_append_left _ member)
  have finalSatisfies : Satisfies finalEqualityRows assignment := by
    intro row member
    exact satisfies row (List.mem_append_right identityRows member)
  have definitionsHold := builderDefinitions_sound canonical constantOne
    definitionsCanonical identitySatisfies
  exact
    { definitionsHold
      chiInitial := chiInitial_computed definitionsHold
      chiLayers := chiLayers_computed definitionsHold
      outputs := outputTraces_computed definitionsHold
      gammaPowerValues := gammaPowers_computed constantOne definitionsHold
      ordinarySumValue := ordinarySum_computed definitionsHold
      blockEqualityComputed := blockEquality_computed definitionsHold
      laneEqualityComputed := laneEquality_computed definitionsHold
      radixConstantValue := radixConstant_computed definitionsHold
      radixPowerValues := radixPowers_computed constantOne definitionsHold
      runningSumValue := runningSum_computed definitionsHold
      oldBlockEqualityComputed := oldBlockEquality_computed definitionsHold
      selectorInitialValue := selectorInitial_computed definitionsHold
      selectorStepsComputed := selectorSteps_computed definitionsHold
      rhs := terminalRhs_eq_expression definitionsHold
      fullRhs := terminalRhs_eq_fullExpression definitionsHold
      final := finalEquality_sound canonical constantOne finalSatisfies }

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.TerminalProgram
