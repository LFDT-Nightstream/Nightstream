import NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransport
import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalEncodes
import NightstreamFPrime.Export.Stage1.PiRLCProductMatrixProgramSemantics
import NightstreamFPrime.Export.Stage1.PiRLCRetainedInputs

/-!
Owns the value-level interpreter for the compact Phi81 quotient recipe
in the per-application assignment transport. The interpreter reads only the
physical base assignment. It does not construct retained coordinates or the
final 26-block assignment.
-/

namespace NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransportProducts

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransport
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PiRLC.v1_2
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open PerApplicationAssignmentPlan

abbrev Program := Lifecycle.Stage1.Application.Program

variable {program : Program}

/-- The physical-base assignment read by every derived-product recipe. -/
abbrev BaseValues (program : Program) :=
  Fin (PiRLCProductPlan.baseSourceWidth program) → F

def familyOrdinal : PiRLCProductSchedule.Family → Nat
  | .commitment => 0
  | .publicInput => 1
  | .evalK => 2
  | .evalA => 3

def emptyFamilyShape : Phi81FamilyShape := ⟨0, 0, 0⟩

/-- Select one family shape without expanding its invocations. Invalid recipe
indices select the empty shape and are rejected by the sealed-package parser. -/
def familyShape (recipe : Phi81QuotientRecipe)
    (family : PiRLCProductSchedule.Family) : Phi81FamilyShape :=
  recipe.familyShapes.getD (familyOrdinal family) emptyFamilyShape

def shapeInvocationCount (recipe : Phi81QuotientRecipe)
    (shape : Phi81FamilyShape) : Nat :=
  shape.sourceCount * shape.blockCount * recipe.ringDegree * shape.cellCount

/-- Prefix count of invocations in the recipe's fixed family order. -/
def familyOffset (recipe : Phi81QuotientRecipe) :
    PiRLCProductSchedule.Family → Nat
  | .commitment => 0
  | .publicInput =>
      shapeInvocationCount recipe (familyShape recipe .commitment)
  | .evalK =>
      shapeInvocationCount recipe (familyShape recipe .commitment) +
        shapeInvocationCount recipe (familyShape recipe .publicInput)
  | .evalA =>
      shapeInvocationCount recipe (familyShape recipe .commitment) +
        shapeInvocationCount recipe (familyShape recipe .publicInput) +
          shapeInvocationCount recipe (familyShape recipe .evalK)

/-- Flat source-major, block-major, lane-major, cell-major recipe index. -/
def invocationIndex (recipe : Phi81QuotientRecipe)
    (descriptor : PiRLCProductSchedule.Descriptor) : Nat :=
  let shape := familyShape recipe descriptor.family
  familyOffset recipe descriptor.family +
    descriptor.source.val * shape.blockCount * recipe.ringDegree *
        shape.cellCount +
      descriptor.block.val * recipe.ringDegree * shape.cellCount +
        descriptor.lane.val * shape.cellCount + descriptor.cell.val

@[simp] private theorem canonical_familyShape
    (family : PiRLCProductSchedule.Family) :
    familyShape (phi81QuotientRecipe program) family =
      match family with
      | .commitment => ⟨17, 22, 1⟩
      | .publicInput => ⟨17, 5, 1⟩
      | .evalK => ⟨17, 1, 2⟩
      | .evalA => ⟨17, 7, 2⟩ := by
  cases family <;> rfl

/-- The assignment recipe uses the authoritative flat product index. -/
private theorem canonical_invocationIndex
    (descriptor : PiRLCProductSchedule.Descriptor) :
    invocationIndex (phi81QuotientRecipe program) descriptor = descriptor.invocation.val := by
  rw [PiRLCProductSchedule.Descriptor.invocation_val]
  rcases descriptor with ⟨family, source, block, lane, cell⟩
  cases family <;>
    simp only [invocationIndex, familyOffset, canonical_familyShape, shapeInvocationCount,
      PiRLCProductSchedule.Family.blockCount, PiRLCProductSchedule.Family.cellCount,
      show (phi81QuotientRecipe program).ringDegree = 54 from rfl]
  all_goals omega

/-- Read one compact block source from the physical base. An invalid slot or
a source outside the physical base evaluates to zero; the sealed decoder
rejects both conditions before execution. -/
def baseBlockValue (program : Program) (base : BaseValues program)
    (kind : BlockKind) (slot : Nat) : F :=
  if slotBound : slot <
      (PerApplicationAssignmentBlocks.entry program kind).block.slotCount then
    let source := PerApplicationAssignmentBlocks.sourceIndex program kind
      ⟨slot, slotBound⟩
    if sourceBound : source < PiRLCProductPlan.baseSourceWidth program then
      base ⟨source, sourceBound⟩
    else
      0
  else
    0

private theorem baseBlockValue_eq_source (program : Program) (base : BaseValues program)
    (kind : BlockKind) (slot : Nat)
    (slotBound : slot <
      (PerApplicationAssignmentBlocks.entry program kind).block.slotCount)
    (sourceBound :
      PerApplicationAssignmentBlocks.sourceIndex program kind
          ⟨slot, slotBound⟩ < PiRLCProductPlan.baseSourceWidth program) :
    baseBlockValue program base kind slot =
      base ⟨PerApplicationAssignmentBlocks.sourceIndex program kind
        ⟨slot, slotBound⟩, sourceBound⟩ := by
  unfold baseBlockValue
  rw [dif_pos slotBound]
  dsimp only
  rw [dif_pos sourceBound]

/-- Recipe arithmetic selects the same source and coefficient word. -/
private theorem canonical_challengeSlot
    (source : Fin PiRLCCombinationInvocations.sourceCount) (lane : Fin ringDegree) :
    (phi81QuotientRecipe program).challengeSlotBase +
        source.val * (phi81QuotientRecipe program).challengeSourceStride + lane.val =
      (PiRLCProductSourceBlocks.challengeIndex source lane).val := by
  exact PiRLCProductMatrixProgram.challengeSlot_eq source lane

private theorem canonical_challenge_read
    (raw : PerApplicationCanonicalAssignment.RawValues program)
    (descriptor : PiRLCProductSchedule.Descriptor) (lane : Fin ringDegree) :
    baseBlockValue program raw.base .challengeWords
        (PiRLCProductSourceBlocks.challengeIndex descriptor.source lane).val =
      PiRLCProductPlan.baseEnv program raw.base (descriptor.challengeColumn lane) := by
  let index := PiRLCProductSourceBlocks.challengeIndex descriptor.source lane
  have slotBound : index.val <
      (PerApplicationAssignmentBlocks.entry program .challengeWords).block.slotCount := index.isLt
  have sourceEq : PerApplicationAssignmentBlocks.sourceIndex program .challengeWords
        ⟨index.val, slotBound⟩ = (PiRLCProductPlan.challengeColumn program descriptor lane).val := by
    change ((PiRLCProductSourceBlocks.challengeBlock program).source index).val = _
    rw [PiRLCProductSourceBlocks.challengeBlock_source]
  have sourceBound : (PiRLCProductPlan.challengeColumn program descriptor lane).val <
      PiRLCProductPlan.baseSourceWidth program :=
    PiRLCProductPlan.baseColumn_val_lt_baseSourceWidth program _ _
  have read := baseBlockValue_eq_source program raw.base .challengeWords index.val
    slotBound (by rw [sourceEq]; exact sourceBound)
  have same : (⟨PerApplicationAssignmentBlocks.sourceIndex program .challengeWords
      ⟨index.val, slotBound⟩, by rw [sourceEq]; exact sourceBound⟩ :
      Fin (PiRLCProductPlan.baseSourceWidth program)) =
      ⟨(PiRLCProductPlan.challengeColumn program descriptor lane).val, sourceBound⟩ := by
    apply Fin.ext
    exact sourceEq
  rw [same] at read
  have encoded := PiRLCRetainedPreservation.sourceAssignment_base program raw.base raw.groupValue
    ⟨(PiRLCProductPlan.challengeColumn program descriptor lane).val, sourceBound⟩
  change PiRLCRetainedPreservation.sourceAssignment program raw.base raw.groupValue
      (PiRLCProductPlan.challengeColumn program descriptor lane) = _ at encoded
  rw [PiRLCRetainedPreservation.sourceAssignment_challengeColumn] at encoded
  exact read.trans encoded.symm

/-- Challenge ring selected by the checked coefficient-word block. -/
def challengeRing (recipe : Phi81QuotientRecipe) (program : Program)
    (base : BaseValues program) (descriptor : PiRLCProductSchedule.Descriptor) :
    RingF :=
  fun lane =>
    baseBlockValue program base recipe.challengeBlock
        (recipe.challengeSlotBase +
          descriptor.source.val * recipe.challengeSourceStride + lane.val) -
      Poseidon2.ofNat recipe.challengeShift

/-- Value ring selected by the recipe's family-major product-input block. -/
def valueRing (recipe : Phi81QuotientRecipe) (program : Program)
    (base : BaseValues program) (descriptor : PiRLCProductSchedule.Descriptor) :
    RingF :=
  fun lane =>
    SourceCompiler.sourceEnv base <|
      AffineRuns.sourceAt recipe.valueSources
        (invocationIndex recipe (descriptor.withLane lane))

/-- Read one quotient coefficient from the complete physical base. The
single retained slot uses the original lane index. -/
def phi81GroupValue (recipe : Phi81QuotientRecipe) (program : Program)
    (base : BaseValues program)
    (invocation : Fin PiRLCProductSchedule.invocationCount)
    (_group : Nat) : F :=
  let ring := PiRLCProductRingSchedule.ringInvocation invocation
  let representative := PiRLCProductRingSchedule.laneInvocation ring PiRLCProductRingSchedule.zeroLane
  let descriptor := PiRLCProductSchedule.descriptor representative
  Phi81Relation.QuotientProduct.quotientCoeff
    (challengeRing recipe program base descriptor)
    (valueRing recipe program base descriptor)
    (PiRLCProductSchedule.descriptor invocation).lane

/-- The physical-base executor computes the quotient coefficient used by the plan. -/
theorem canonical_phi81GroupValue_eq_honestGroupValue
    {program : Program}
    (raw : PerApplicationCanonicalAssignment.RawValues program)
    (invocation : Fin PiRLCProductSchedule.invocationCount)
    (group : Fin 1) :
    phi81GroupValue (phi81QuotientRecipe program) program raw.base invocation group.val =
      PiRLCProductPlan.honestGroupValue
        (PiRLCProductMatrixProgram.inputs
          (PerApplicationCanonicalEncodes.piCcsOrdinaryGeometry program))
        raw.assignment invocation group := by
  let geometry := PerApplicationCanonicalEncodes.retainedGeometry program
  let values := PiRLCValueWiring.form
    (PerApplicationCanonicalEncodes.piCcsOrdinaryGeometry program)
  let inputs := PiRLCRetainedInputs.productInputs values geometry
  let ring := PiRLCProductRingSchedule.ringInvocation invocation
  let representative := PiRLCProductRingSchedule.laneInvocation ring PiRLCProductRingSchedule.zeroLane
  let descriptor := PiRLCProductSchedule.descriptor representative
  have one : raw.assignment inputs.oneColumn = 1 := by
    exact PerApplicationCanonicalAssignment.assignment_one raw
  have encodes := PerApplicationCanonicalEncodes.retainedEncodes raw
  have preserves := PiRLCRetainedPreservation.productInputs_preserves
    values geometry raw.assignment raw.base raw.groupValue
      (PerApplicationCanonicalEncodes.productValuesPreserve raw) encodes
  have challengeRead :
      challengeRing (phi81QuotientRecipe program) program raw.base descriptor =
        PiRLCProductPlan.challengeRing program raw.base descriptor := by
    funext lane
    unfold challengeRing PiRLCProductPlan.challengeRing
    dsimp only [phi81QuotientRecipe]
    have slotEq := canonical_challengeSlot (program := program) descriptor.source lane
    dsimp only [phi81QuotientRecipe] at slotEq
    rw [slotEq]
    rw [canonical_challenge_read raw descriptor lane]
    rfl
  have valueRead :
      valueRing (phi81QuotientRecipe program) program raw.base descriptor =
        PiRLCProductPlan.valueRing program raw.base descriptor := by
    funext lane
    unfold valueRing PiRLCProductPlan.valueRing
    rw [canonical_invocationIndex]
    conv_lhs =>
      arg 2
      arg 1
      dsimp only [phi81QuotientRecipe]
    rw [phi81ValueSources_at,
      PiRLCProductSchedule.descriptor_invocation]
    simp only [PiRLCProductPlan.valueColumn,
      PiRLCProductSchedule.Descriptor.withLane_valueColumn]
    rw [PiRLCProductPlan.baseEnv_valueColumn]
    exact SourceCompiler.sourceEnv_at raw.base _
  have challengeStateEval :
      Phi81ProductPlan.evalState raw.assignment
          (PiRLCProductPlan.challengeState inputs representative) =
        PiRLCProductPlan.challengeRing program raw.base descriptor := by
    funext lane
    have challengePreserves :
        (PiRLCProductPlan.challengeForm inputs representative lane).eval
            raw.assignment =
          PiRLCProductPlan.baseEnv program raw.base
            (descriptor.challengeColumn lane) := by
      simpa only [descriptor] using preserves.challenge representative lane
    simp [Phi81ProductPlan.evalState, PiRLCProductPlan.challengeState,
      PiRLCProductPlan.challengeRing, challengePreserves, one,
      sub_eq_add_neg]
  have valueStateEval :
      Phi81ProductPlan.evalState raw.assignment
          (PiRLCProductPlan.valueState inputs representative) =
        PiRLCProductPlan.valueRing program raw.base descriptor := by
    funext lane
    exact preserves.value representative lane
  have challengeEq :
      challengeRing (phi81QuotientRecipe program) program raw.base descriptor =
        Phi81ProductPlan.evalState raw.assignment
          (PiRLCProductPlan.challengeState inputs representative) :=
    challengeRead.trans challengeStateEval.symm
  have valueEq :
      valueRing (phi81QuotientRecipe program) program raw.base descriptor =
        Phi81ProductPlan.evalState raw.assignment
          (PiRLCProductPlan.valueState inputs representative) :=
    valueRead.trans valueStateEval.symm
  unfold phi81GroupValue
  change Phi81Relation.QuotientProduct.quotientCoeff
      (challengeRing (phi81QuotientRecipe program) program raw.base descriptor)
      (valueRing (phi81QuotientRecipe program) program raw.base descriptor)
      (PiRLCProductSchedule.descriptor invocation).lane = _
  rw [challengeEq, valueEq]
  rfl


end NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransportProducts
