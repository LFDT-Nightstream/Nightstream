import NightstreamFPrime.Export.Stage1.PiRLCCombinationInvocations
import NightstreamFPrime.Export.Stage1.PiRLCSamplerRows
import NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily
import NightstreamFPrime.Layout.PiRLC.v1_1.Composition

/-!
Owns list-level conformance between the compact PiRLC combination export and
the exact production parent constraints.

The proofs preserve the parent's source order and its finite-product index
order. This module does not own templates, package assembly, or witness data.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCCombinationConformance

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Export.Stage1.PiRLCCombinationInvocations

def sourceConstraints (logicalStart blockCount cellCount valueStride source : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat) : List Expr :=
  List.ofFn fun index : Fin
      (CombinationStep.privateCount blockCount cellCount) =>
    let coordinates := CombinationStep.coordinates index
    sourceConstraint logicalStart blockCount cellCount valueStride source
      coordinates.1.val coordinates.2.2.val valueSourceStart coordinates.2.1

theorem sourceConstraints_eq_childConstraints
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : CombinationFamily.Interface blockCount cellCount)
    (logicalStart valueStride source : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat)
    (pointwise : ∀ (block : Fin blockCount) (lane : Fin ringDegree)
      (cell : Fin cellCount),
      sourceConstraint logicalStart blockCount cellCount valueStride source
          block.val cell.val valueSourceStart lane =
        CombinationStep.output
            (CombinationFamily.stepOffset logicalStart source blockCount cellCount)
            (CombinationStep.indexOf block lane cell) -
          CombinationStep.recipe
            (CombinationFamily.stepInterface interface logicalStart source)
            (CombinationFamily.stepOffset logicalStart source blockCount cellCount)
            (CombinationStep.indexOf block lane cell)) :
    sourceConstraints logicalStart blockCount cellCount valueStride source
        valueSourceStart =
      NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.childConstraints
        interface logicalStart source := by
  unfold sourceConstraints
    NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.childConstraints
    NightstreamFPrime.Layout.PiRLC.v1_1.CombinationStep.logicalConstraints
  rw [stepFlatConstraints_eq_assertions]
  apply congrArg List.ofFn
  funext index
  let coordinates := CombinationStep.coordinates index
  have equality := pointwise coordinates.1 coordinates.2.1 coordinates.2.2
  have indexEq : CombinationStep.indexOf coordinates.1 coordinates.2.1
      coordinates.2.2 = index := by
    exact indexOf_coordinates index
  rw [indexEq] at equality
  exact equality

theorem sourceConstraint_freshCount_eq
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : CombinationFamily.Interface blockCount cellCount)
    (logicalStart valueStride source : Nat)
    (sourceLt : source < CombinationFamily.sourceCount)
    (valueSourceStart : Nat → Nat → Nat → Nat)
    (inputs :
      NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.ProductionInputs
        interface logicalStart)
    (block : Fin blockCount) (lane : Fin ringDegree) (cell : Fin cellCount)
    (constraintEq :
      sourceConstraint logicalStart blockCount cellCount valueStride source
          block.val cell.val valueSourceStart lane =
        CombinationStep.output
            (CombinationFamily.stepOffset logicalStart source blockCount cellCount)
            (CombinationStep.indexOf block lane cell) -
          CombinationStep.recipe
            (CombinationFamily.stepInterface interface logicalStart source)
            (CombinationFamily.stepOffset logicalStart source blockCount cellCount)
            (CombinationStep.indexOf block lane cell)) :
    R1CS.constraintFreshCount
        (sourceConstraint logicalStart blockCount cellCount valueStride source
          block.val cell.val valueSourceStart lane) =
      laneFreshCost lane.val := by
  rw [constraintEq]
  unfold CombinationStep.output
  rw [NightstreamFPrime.Layout.PiRLC.v1_1.CombinationStep.constraintFreshCountEqLane
    (CombinationFamily.stepInterface interface logicalStart source)
    (CombinationFamily.stepOffset logicalStart source blockCount cellCount)
    (NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.stepProductionInputs
      interface logicalStart source sourceLt inputs)
    (CombinationStep.indexOf block lane cell)]
  rw [CombinationStep.laneOf_indexOf]
  simp [laneFreshCost, laneFreshCosts, lane.isLt]

theorem commitmentSourceConstraints_eq_childConstraints
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    (source : Fin CombinationFamily.sourceCount) :
    sourceConstraints PiRLCStarts.commitmentLogicalStart
        CommitmentCombination.blockCount CommitmentCombination.cellCount 1
        source.val commitmentValueSourceStart =
      NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.childConstraints
        (productionCommitmentFamilyInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits))
        PiRLCStarts.commitmentLogicalStart source.val := by
  apply sourceConstraints_eq_childConstraints
  intro block lane cell
  exact commitmentSourceConstraint_eq_stepAssertion
    (logicalWidth := logicalWidth) (publicFits := publicFits)
    source block lane cell

theorem publicInputSourceConstraints_eq_childConstraints
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    (source : Fin CombinationFamily.sourceCount) :
    sourceConstraints PiRLCStarts.publicInputLogicalStart
        PublicInputCombination.blockCount PublicInputCombination.cellCount 1
        source.val publicInputValueSourceStart =
      NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.childConstraints
        (productionPublicInputFamilyInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits))
        PiRLCStarts.publicInputLogicalStart source.val := by
  apply sourceConstraints_eq_childConstraints
  intro block lane cell
  exact publicInputSourceConstraint_eq_stepAssertion
    (logicalWidth := logicalWidth) (publicFits := publicFits)
    source block lane cell

theorem evalKSourceConstraints_eq_childConstraints
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    (source : Fin CombinationFamily.sourceCount) :
    sourceConstraints PiRLCStarts.evalKLogicalStart
        EvalKCombination.blockCount RingKCombination.cellCount 2
        source.val evalKValueSourceStart =
      NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.childConstraints
        (productionEvalKFamilyInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits))
        PiRLCStarts.evalKLogicalStart source.val := by
  apply sourceConstraints_eq_childConstraints
  intro block lane cell
  exact evalKSourceConstraint_eq_stepAssertion
    (logicalWidth := logicalWidth) (publicFits := publicFits)
    source block lane cell

theorem evalASourceConstraints_eq_childConstraints
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    (source : Fin CombinationFamily.sourceCount) :
    sourceConstraints PiRLCStarts.evalALogicalStart
        EvalACombination.blockCount RingKCombination.cellCount 2
        source.val evalAValueSourceStart =
      NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.childConstraints
        (productionEvalAFamilyInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits))
        PiRLCStarts.evalALogicalStart source.val := by
  apply sourceConstraints_eq_childConstraints
  intro block lane cell
  exact evalASourceConstraint_eq_stepAssertion
    (logicalWidth := logicalWidth) (publicFits := publicFits)
    source block lane cell

def familyConstraints (logicalStart blockCount cellCount valueStride : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat) : List Expr :=
  ((List.range sourceCount).map fun source =>
    sourceConstraints logicalStart blockCount cellCount valueStride source
      valueSourceStart).flatten

theorem familyConstraints_eq_orderedConstraints
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : CombinationFamily.Interface blockCount cellCount)
    (logicalStart valueStride : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat)
    (each : ∀ source : Fin CombinationFamily.sourceCount,
      sourceConstraints logicalStart blockCount cellCount valueStride source.val
          valueSourceStart =
        NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.childConstraints
          interface logicalStart source.val) :
    familyConstraints logicalStart blockCount cellCount valueStride
        valueSourceStart =
      NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.orderedConstraints
        interface logicalStart := by
  have countEq : sourceCount = CombinationFamily.sourceCount := by
    rw [CombinationFamily.sourceCount_eq]
    rfl
  unfold familyConstraints
    NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.orderedConstraints
    NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.childConstraintLists
  rw [countEq]
  apply congrArg List.flatten
  apply List.map_congr_left
  intro source member
  exact each ⟨source, List.mem_range.mp member⟩

theorem commitmentFamilyConstraints_eq_parent
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth} :
    familyConstraints PiRLCStarts.commitmentLogicalStart
        CommitmentCombination.blockCount CommitmentCombination.cellCount 1
        commitmentValueSourceStart =
      NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.logicalConstraints
        (productionCommitmentFamilyInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits)) PiRLCStarts.commitmentLogicalStart := by
  rw [NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.logicalConstraints_eq_ordered]
  apply familyConstraints_eq_orderedConstraints
  intro source
  exact commitmentSourceConstraints_eq_childConstraints source

theorem publicInputFamilyConstraints_eq_parent
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth} :
    familyConstraints PiRLCStarts.publicInputLogicalStart
        PublicInputCombination.blockCount PublicInputCombination.cellCount 1
        publicInputValueSourceStart =
      NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.logicalConstraints
        (productionPublicInputFamilyInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits)) PiRLCStarts.publicInputLogicalStart := by
  rw [NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.logicalConstraints_eq_ordered]
  apply familyConstraints_eq_orderedConstraints
  intro source
  exact publicInputSourceConstraints_eq_childConstraints source

theorem evalKFamilyConstraints_eq_parent
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth} :
    familyConstraints PiRLCStarts.evalKLogicalStart
        EvalKCombination.blockCount RingKCombination.cellCount 2
        evalKValueSourceStart =
      NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.logicalConstraints
        (productionEvalKFamilyInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits)) PiRLCStarts.evalKLogicalStart := by
  rw [NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.logicalConstraints_eq_ordered]
  apply familyConstraints_eq_orderedConstraints
  intro source
  exact evalKSourceConstraints_eq_childConstraints source

theorem evalAFamilyConstraints_eq_parent
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth} :
    familyConstraints PiRLCStarts.evalALogicalStart
        EvalACombination.blockCount RingKCombination.cellCount 2
        evalAValueSourceStart =
      NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.logicalConstraints
        (productionEvalAFamilyInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits)) PiRLCStarts.evalALogicalStart := by
  rw [NightstreamFPrime.Layout.PiRLC.v1_1.CombinationFamily.logicalConstraints_eq_ordered]
  apply familyConstraints_eq_orderedConstraints
  intro source
  exact evalASourceConstraints_eq_childConstraints source

def combinationConstraints : List Expr :=
  familyConstraints PiRLCStarts.commitmentLogicalStart
      CommitmentCombination.blockCount CommitmentCombination.cellCount 1
      commitmentValueSourceStart ++
    familyConstraints PiRLCStarts.publicInputLogicalStart
      PublicInputCombination.blockCount PublicInputCombination.cellCount 1
      publicInputValueSourceStart ++
    familyConstraints PiRLCStarts.evalKLogicalStart
      EvalKCombination.blockCount RingKCombination.cellCount 2
      evalKValueSourceStart ++
    familyConstraints PiRLCStarts.evalALogicalStart
      EvalACombination.blockCount RingKCombination.cellCount 2
      evalAValueSourceStart

/-- Satisfaction of one intended compact invocation implies its exact source
constraint under the Stage 1 column pullback. -/
theorem invocationRows_imply_sourceConstraint
    (logicalStart rowStart freshStart blockCount cellCount valueStride source
      block cell : Nat) (lane : Fin ringDegree)
    (valueSourceStart : Nat → Nat → Nat → Nat)
    (sourceLocal : Spartan.piCcsPhaseOffset ≤
      invocationFreshSource freshStart blockCount cellCount source block
        lane.val cell)
    (valueAffine : ∀ offset, offset < ringDegree →
      Spartan.sourceToSpartan
          (valueSourceStart source block cell + offset * valueStride) =
        Spartan.sourceToSpartan (valueSourceStart source block cell) +
          offset * valueStride)
    (env : Env)
    (holds : R1CS.RowsHold env
      (CompactRows.instantiateRows
        (CompactRows.inputColumnOfRanges
          (invocation logicalStart rowStart freshStart blockCount cellCount
            valueStride source block lane.val cell valueSourceStart).inputRanges)
        (invocation logicalStart rowStart freshStart blockCount cellCount
          valueStride source block lane.val cell valueSourceStart).localStart
        (PiRLCCombinationTemplates.template (firstSource source) lane))) :
    (sourceConstraint logicalStart blockCount cellCount valueStride source block
      cell valueSourceStart lane).eval (Spartan.pullback env) = 0 := by
  rw [invocationRows_eq_remappedSource logicalStart rowStart freshStart
    blockCount cellCount valueStride source block cell lane valueSourceStart
    sourceLocal valueAffine] at holds
  apply R1CS.lowerGenericConstraint_sound
  exact (Spartan.remapRows_hold env _).mp holds

/-- The exact Lean-lowered source rows imply the matching compact invocation
after the Stage 1 column map. This is the constructive direction used by
package completeness; it does not execute a second witness program. -/
theorem sourceRows_imply_invocationRows
    (logicalStart rowStart freshStart blockCount cellCount valueStride source
      block cell : Nat) (lane : Fin ringDegree)
    (valueSourceStart : Nat → Nat → Nat → Nat)
    (sourceLocal : Spartan.piCcsPhaseOffset ≤
      invocationFreshSource freshStart blockCount cellCount source block
        lane.val cell)
    (valueAffine : ∀ offset, offset < ringDegree →
      Spartan.sourceToSpartan
          (valueSourceStart source block cell + offset * valueStride) =
        Spartan.sourceToSpartan (valueSourceStart source block cell) +
          offset * valueStride)
    (env : Env)
    (holds : R1CS.RowsHold (Spartan.pullback env)
      (R1CS.lowerGenericConstraint
        (sourceConstraint logicalStart blockCount cellCount valueStride source
          block cell valueSourceStart lane)
        (invocationFreshSource freshStart blockCount cellCount source block
          lane.val cell)).rows) :
    R1CS.RowsHold env
      (CompactRows.instantiateRows
        (CompactRows.inputColumnOfRanges
          (invocation logicalStart rowStart freshStart blockCount cellCount
            valueStride source block lane.val cell valueSourceStart).inputRanges)
        (invocation logicalStart rowStart freshStart blockCount cellCount
          valueStride source block lane.val cell valueSourceStart).localStart
        (PiRLCCombinationTemplates.template (firstSource source) lane)) := by
  rw [invocationRows_eq_remappedSource logicalStart rowStart freshStart
    blockCount cellCount valueStride source block cell lane valueSourceStart
    sourceLocal valueAffine]
  exact (Spartan.remapRows_hold env _).mpr holds

theorem commitmentInvocationRows_imply_sourceConstraint
    (source block cell : Nat) (lane : Fin ringDegree)
    (sourceLt : source < sourceCount) (blockLt : block < 22)
    (env : Env)
    (holds : R1CS.RowsHold env
      (CompactRows.instantiateRows
        (CompactRows.inputColumnOfRanges
          (invocation PiRLCStarts.commitmentLogicalStart
            PiRLCStarts.commitmentRowStart PiRLCStarts.commitmentFreshStart
            22 1 1 source block lane.val cell
            commitmentValueSourceStart).inputRanges)
        (invocation PiRLCStarts.commitmentLogicalStart
          PiRLCStarts.commitmentRowStart PiRLCStarts.commitmentFreshStart
          22 1 1 source block lane.val cell
          commitmentValueSourceStart).localStart
        (PiRLCCombinationTemplates.template (firstSource source) lane))) :
    (sourceConstraint PiRLCStarts.commitmentLogicalStart 22 1 1 source block
      cell commitmentValueSourceStart lane).eval (Spartan.pullback env) = 0 := by
  apply invocationRows_imply_sourceConstraint
  · exact invocationFreshSource_local _ _ _ _ _ _ _
      commitmentFreshStart_local
  · intro offset offsetLt
    simpa using commitmentValueSource_affine source block cell offset sourceLt
      blockLt offsetLt
  · exact holds

theorem publicInputInvocationRows_imply_sourceConstraint
    (source block cell : Nat) (lane : Fin ringDegree)
    (sourceLt : source < sourceCount) (blockLt : block < 5) (env : Env)
    (holds : R1CS.RowsHold env
      (CompactRows.instantiateRows
        (CompactRows.inputColumnOfRanges
          (invocation PiRLCStarts.publicInputLogicalStart
            PiRLCStarts.publicInputRowStart PiRLCStarts.publicInputFreshStart
            5 1 1 source block lane.val cell
            publicInputValueSourceStart).inputRanges)
        (invocation PiRLCStarts.publicInputLogicalStart
          PiRLCStarts.publicInputRowStart PiRLCStarts.publicInputFreshStart
          5 1 1 source block lane.val cell
          publicInputValueSourceStart).localStart
        (PiRLCCombinationTemplates.template (firstSource source) lane))) :
    (sourceConstraint PiRLCStarts.publicInputLogicalStart 5 1 1 source block
      cell publicInputValueSourceStart lane).eval (Spartan.pullback env) = 0 := by
  apply invocationRows_imply_sourceConstraint
  · exact invocationFreshSource_local _ _ _ _ _ _ _
      publicInputFreshStart_local
  · intro offset offsetLt
    simpa using publicInputValueSource_affine source block cell offset sourceLt
      blockLt offsetLt
  · exact holds

theorem evalKInvocationRows_imply_sourceConstraint
    (source block cell : Nat) (lane : Fin ringDegree)
    (sourceLt : source < sourceCount) (cellLt : cell < 2) (env : Env)
    (holds : R1CS.RowsHold env
      (CompactRows.instantiateRows
        (CompactRows.inputColumnOfRanges
          (invocation PiRLCStarts.evalKLogicalStart PiRLCStarts.evalKRowStart
            PiRLCStarts.evalKFreshStart 1 2 2 source block lane.val cell
            evalKValueSourceStart).inputRanges)
        (invocation PiRLCStarts.evalKLogicalStart PiRLCStarts.evalKRowStart
          PiRLCStarts.evalKFreshStart 1 2 2 source block lane.val cell
          evalKValueSourceStart).localStart
        (PiRLCCombinationTemplates.template (firstSource source) lane))) :
    (sourceConstraint PiRLCStarts.evalKLogicalStart 1 2 2 source block cell
      evalKValueSourceStart lane).eval (Spartan.pullback env) = 0 := by
  apply invocationRows_imply_sourceConstraint
  · exact invocationFreshSource_local _ _ _ _ _ _ _ evalKFreshStart_local
  · intro offset offsetLt
    exact evalKValueSource_affine source block cell offset sourceLt cellLt
      offsetLt
  · exact holds

theorem evalAInvocationRows_imply_sourceConstraint
    (source block cell : Nat) (lane : Fin ringDegree)
    (sourceLt : source < sourceCount) (blockLt : block < 14)
    (cellLt : cell < 2) (env : Env)
    (holds : R1CS.RowsHold env
      (CompactRows.instantiateRows
        (CompactRows.inputColumnOfRanges
          (invocation PiRLCStarts.evalALogicalStart PiRLCStarts.evalARowStart
            PiRLCStarts.evalAFreshStart 14 2 2 source block lane.val cell
            evalAValueSourceStart).inputRanges)
        (invocation PiRLCStarts.evalALogicalStart PiRLCStarts.evalARowStart
          PiRLCStarts.evalAFreshStart 14 2 2 source block lane.val cell
          evalAValueSourceStart).localStart
        (PiRLCCombinationTemplates.template (firstSource source) lane))) :
    (sourceConstraint PiRLCStarts.evalALogicalStart 14 2 2 source block cell
      evalAValueSourceStart lane).eval (Spartan.pullback env) = 0 := by
  apply invocationRows_imply_sourceConstraint
  · exact invocationFreshSource_local _ _ _ _ _ _ _ evalAFreshStart_local
  · intro offset offsetLt
    exact evalAValueSource_affine source block cell offset sourceLt blockLt
      cellLt offsetLt
  · exact holds

def FamilyInvocationRowsHold
    (logicalStart rowStart freshStart blockCount cellCount valueStride : Nat)
    [NeZero cellCount] (valueSourceStart : Nat → Nat → Nat → Nat)
    (env : Env) : Prop :=
  ∀ source : Fin sourceCount,
    ∀ index : Fin (CombinationStep.privateCount blockCount cellCount),
      let coordinates := CombinationStep.coordinates index
      R1CS.RowsHold env
        (CompactRows.instantiateRows
          (CompactRows.inputColumnOfRanges
            (invocation logicalStart rowStart freshStart blockCount cellCount
              valueStride source.val coordinates.1.val coordinates.2.1.val
                coordinates.2.2.val valueSourceStart).inputRanges)
          (invocation logicalStart rowStart freshStart blockCount cellCount
            valueStride source.val coordinates.1.val coordinates.2.1.val
              coordinates.2.2.val valueSourceStart).localStart
          (PiRLCCombinationTemplates.template (firstSource source.val)
            coordinates.2.1))

theorem familyConstraintZeros_imply_prefix
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : CombinationFamily.Interface blockCount cellCount)
    (logicalStart valueStride : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat)
    (env : Env)
    (zeros : ∀ source : Fin sourceCount,
      ∀ index : Fin (CombinationStep.privateCount blockCount cellCount),
        let coordinates := CombinationStep.coordinates index
        (sourceConstraint logicalStart blockCount cellCount valueStride
          source.val coordinates.1.val coordinates.2.2.val valueSourceStart
            coordinates.2.1).eval (Spartan.pullback env) = 0)
    (constraintEq : ∀ source : Fin sourceCount,
      ∀ block : Fin blockCount, ∀ lane : Fin ringDegree,
      ∀ cell : Fin cellCount,
        sourceConstraint logicalStart blockCount cellCount valueStride source.val
            block.val cell.val valueSourceStart lane =
          CombinationStep.output
              (CombinationFamily.stepOffset logicalStart source.val blockCount
                cellCount)
              (CombinationStep.indexOf block lane cell) -
            CombinationStep.recipe
              (CombinationFamily.stepInterface interface logicalStart source.val)
              (CombinationFamily.stepOffset logicalStart source.val blockCount
                cellCount)
              (CombinationStep.indexOf block lane cell)) :
    CombinationFamily.PrefixHolds interface logicalStart
      (Spartan.pullback env) := by
  intro source index
  let coordinates := CombinationStep.coordinates index
  have zero := zeros source index
  dsimp only at zero
  have assertionEq := constraintEq source coordinates.1 coordinates.2.1
    coordinates.2.2
  rw [assertionEq] at zero
  have indexEq : CombinationStep.indexOf coordinates.1 coordinates.2.1
      coordinates.2.2 = index := by
    exact indexOf_coordinates index
  rw [indexEq] at zero
  have outputEq :
      (CombinationStep.output
        (CombinationFamily.stepOffset logicalStart source.val blockCount
          cellCount) index).eval (Spartan.pullback env) =
      (CombinationStep.recipe
        (CombinationFamily.stepInterface interface logicalStart source.val)
        (CombinationFamily.stepOffset logicalStart source.val blockCount
          cellCount) index).eval (Spartan.pullback env) :=
    sub_eq_zero.mp (by simpa [Expr.eval_sub] using zero)
  exact outputEq.trans
    (CombinationStep.recipe_eval
      (CombinationFamily.stepInterface interface logicalStart source.val)
      (CombinationFamily.stepOffset logicalStart source.val blockCount cellCount)
      (Spartan.pullback env) index)

theorem commitmentFamilyRows_imply_canonical
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    (env : Env)
    (rows : FamilyInvocationRowsHold PiRLCStarts.commitmentLogicalStart
      PiRLCStarts.commitmentRowStart PiRLCStarts.commitmentFreshStart 22 1 1
      commitmentValueSourceStart env) :
    CombinationFamily.CanonicalHolds
      (productionCommitmentFamilyInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits))
      PiRLCStarts.commitmentLogicalStart (Spartan.pullback env) := by
  apply CombinationFamily.relation_implies_canonical
  apply CombinationFamily.parentCoverage
  apply familyConstraintZeros_imply_prefix
    (productionCommitmentFamilyInterface (logicalWidth := logicalWidth)
      (publicFits := publicFits))
    PiRLCStarts.commitmentLogicalStart 1 commitmentValueSourceStart env
  · intro source index
    let coordinates := CombinationStep.coordinates index
    apply commitmentInvocationRows_imply_sourceConstraint source.val
      coordinates.1.val coordinates.2.2.val coordinates.2.1 source.isLt
      coordinates.1.isLt env
    exact rows source index
  · intro source block lane cell
    exact commitmentSourceConstraint_eq_stepAssertion
      (logicalWidth := logicalWidth) (publicFits := publicFits)
      source block lane cell

theorem publicInputFamilyRows_imply_canonical
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    (env : Env)
    (rows : FamilyInvocationRowsHold PiRLCStarts.publicInputLogicalStart
      PiRLCStarts.publicInputRowStart PiRLCStarts.publicInputFreshStart 5 1 1
      publicInputValueSourceStart env) :
    CombinationFamily.CanonicalHolds
      (productionPublicInputFamilyInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits))
      PiRLCStarts.publicInputLogicalStart (Spartan.pullback env) := by
  apply CombinationFamily.relation_implies_canonical
  apply CombinationFamily.parentCoverage
  apply familyConstraintZeros_imply_prefix
    (productionPublicInputFamilyInterface (logicalWidth := logicalWidth)
      (publicFits := publicFits))
    PiRLCStarts.publicInputLogicalStart 1 publicInputValueSourceStart env
  · intro source index
    let coordinates := CombinationStep.coordinates index
    apply publicInputInvocationRows_imply_sourceConstraint source.val
      coordinates.1.val coordinates.2.2.val coordinates.2.1 source.isLt
      coordinates.1.isLt env
    exact rows source index
  · intro source block lane cell
    exact publicInputSourceConstraint_eq_stepAssertion
      (logicalWidth := logicalWidth) (publicFits := publicFits)
      source block lane cell

theorem evalKFamilyRows_imply_canonical
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    (env : Env)
    (rows : FamilyInvocationRowsHold PiRLCStarts.evalKLogicalStart
      PiRLCStarts.evalKRowStart PiRLCStarts.evalKFreshStart 1 2 2
      evalKValueSourceStart env) :
    CombinationFamily.CanonicalHolds
      (productionEvalKFamilyInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits))
      PiRLCStarts.evalKLogicalStart (Spartan.pullback env) := by
  apply CombinationFamily.relation_implies_canonical
  apply CombinationFamily.parentCoverage
  apply familyConstraintZeros_imply_prefix
    (productionEvalKFamilyInterface (logicalWidth := logicalWidth)
      (publicFits := publicFits))
    PiRLCStarts.evalKLogicalStart 2 evalKValueSourceStart env
  · intro source index
    let coordinates := CombinationStep.coordinates index
    apply evalKInvocationRows_imply_sourceConstraint source.val
      coordinates.1.val coordinates.2.2.val coordinates.2.1 source.isLt
      coordinates.2.2.isLt env
    exact rows source index
  · intro source block lane cell
    exact evalKSourceConstraint_eq_stepAssertion
      (logicalWidth := logicalWidth) (publicFits := publicFits)
      source block lane cell

theorem evalAFamilyRows_imply_canonical
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    (env : Env)
    (rows : FamilyInvocationRowsHold PiRLCStarts.evalALogicalStart
      PiRLCStarts.evalARowStart PiRLCStarts.evalAFreshStart 14 2 2
      evalAValueSourceStart env) :
    CombinationFamily.CanonicalHolds
      (productionEvalAFamilyInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits))
      PiRLCStarts.evalALogicalStart (Spartan.pullback env) := by
  apply CombinationFamily.relation_implies_canonical
  apply CombinationFamily.parentCoverage
  apply familyConstraintZeros_imply_prefix
    (productionEvalAFamilyInterface (logicalWidth := logicalWidth)
      (publicFits := publicFits))
    PiRLCStarts.evalALogicalStart 2 evalAValueSourceStart env
  · intro source index
    let coordinates := CombinationStep.coordinates index
    apply evalAInvocationRows_imply_sourceConstraint source.val
      coordinates.1.val coordinates.2.2.val coordinates.2.1 source.isLt
      coordinates.1.isLt coordinates.2.2.isLt env
    exact rows source index
  · intro source block lane cell
    exact evalASourceConstraint_eq_stepAssertion
      (logicalWidth := logicalWidth) (publicFits := publicFits)
      source block lane cell

end NightstreamFPrime.Export.Stage1.PiRLCCombinationConformance
