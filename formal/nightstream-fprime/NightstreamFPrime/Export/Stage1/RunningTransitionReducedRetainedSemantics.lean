import NightstreamFPrime.Export.Stage1.RunningTransitionReducedRetainedBlocks
import NightstreamFPrime.Export.Stage1.RunningTransitionReducedMatrixProgram
import NightstreamFPrime.Export.Stage1.RunningTransitionDirectPlan

/-!
Bind the reduced blocks to their actual physical source values and construct
their canonical 42-coordinate witness. Arbitrary old field encodings are
recomposed from all 41 coordinates before the flag is encoded as one bit.
This module does not select the new aggregate assignment layout.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.RunningTransitionReducedRetainedSemantics

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1 NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open RunningTransitionReducedRetainedBlocks

abbrev BaseValues (program : ApplicationProgram) :=
  Fin (PiRLCProductPlan.baseSourceWidth program) → F

/-- Pull the actual shifted physical packet back to its original source indices. -/
def sourceEnv (program : ApplicationProgram) (base : BaseValues program) : Env :=
  Spartan.pullback (RunningTransitionDirectPlan.packageEnv program base)

theorem inverseWire_eq (program : ApplicationProgram) :
    RunningTransitionReducedMatrixProgram.inverseWire program =
      RetainedBlock.ofSemantic (inverseBlock program) (inverseStart program) := by rfl

theorem flagWire_eq (program : ApplicationProgram) :
    RunningTransitionReducedMatrixProgram.flagWire program =
      RetainedBlock.ofSemantic (flagBlock program) (flagStart program) := by rfl

theorem inverseSource_value (program : ApplicationProgram) (base : BaseValues program)
    (groups : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F) :
    PiRLCRetainedPreservation.sourceAssignment program base groups products (inverseSource program) =
      sourceEnv program base RunningTransitionInputs.phaseOffset := by
  exact RunningTransitionDirectPlan.sourceAssignment_packageSource program base groups products
    RunningTransitionInputs.phaseOffset _

theorem flagSource_value (program : ApplicationProgram) (base : BaseValues program)
    (groups : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F) :
    PiRLCRetainedPreservation.sourceAssignment program base groups products (flagSource program) =
      sourceEnv program base RunningTransitionReducedRows.flagIndex := by
  exact RunningTransitionDirectPlan.sourceAssignment_packageSource program base groups products
    RunningTransitionReducedRows.flagIndex _

/-- Validity comes from the old physical rows and actual selected source function. -/
theorem blocks_valid {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
    (program : ApplicationProgram) (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (base : BaseValues program)
    (groups : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (physical : RunningTransitionLayout.PhysicalHolds logicalWidth publicFits (sourceEnv program base)) :
    (∀ slot, LowNormSlot.Valid (inverseBlock program).kind
      (PiRLCRetainedPreservation.sourceAssignment program base groups products
        ((inverseBlock program).source slot))) ∧
    (∀ slot, LowNormSlot.Valid (flagBlock program).kind
      (PiRLCRetainedPreservation.sourceAssignment program base groups products
        ((flagBlock program).source slot))) := by
  constructor
  · intro slot
    trivial
  · intro slot
    change LowNormSlot.Valid .bit
      (PiRLCRetainedPreservation.sourceAssignment program base groups products (flagSource program))
    rw [flagSource_value]
    exact RunningTransitionReducedRows.physical_firstScratch_boolean relation
      (sourceEnv program base) physical

/-- Canonical local witness: 41 inverse trits followed by the decoded flag value. -/
def coordinates (inverse flag : F) : List F :=
  LowNormSlot.encode .field inverse ++ LowNormSlot.encode .bit flag

theorem coordinates_length (inverse flag : F) : (coordinates inverse flag).length = 42 := by
  rw [coordinates, List.length_append, LowNormSlot.encode_length, LowNormSlot.encode_length]
  rfl

theorem coordinates_inverse (inverse flag : F) :
    BalancedTernary.recompose ((coordinates inverse flag).take 41) = inverse := by
  have width : (LowNormSlot.encode .field inverse).length = 41 := LowNormSlot.encode_length _ _
  rw [coordinates, ← width, List.take_left, LowNormSlot.recompose_encode]

theorem coordinates_flag (inverse flag : F) : (coordinates inverse flag).getD 41 0 = flag := by
  rw [coordinates, List.getD_append_right]
  · rw [LowNormSlot.encode_length]
    rfl
  · rw [LowNormSlot.encode_length]
    exact Nat.le_refl _

theorem coordinates_norm (inverse flag : F) (bit : flag = 0 ∨ flag = 1) :
    normBounded 2 (coordinates inverse flag) := by
  intro value member
  rcases List.mem_append.mp member with inverseMember | flagMember
  · exact LowNormSlot.encode_norm .field inverse trivial value inverseMember
  · exact LowNormSlot.encode_norm .bit flag bit value flagMember

/-- The existing full witness supplies an admissible local witness without new hints. -/
def fromPhysical (program : ApplicationProgram) (base : BaseValues program) : List F :=
  coordinates (sourceEnv program base RunningTransitionInputs.phaseOffset)
    (sourceEnv program base RunningTransitionReducedRows.flagIndex)

/-- Local canonical coordinates before the aggregate layout assigns their start. -/
def localAssignment (program : ApplicationProgram) (base : BaseValues program) : Fin 42 → F :=
  fun column => (fromPhysical program base).getD column.val 0

theorem localAssignment_encodes (program : ApplicationProgram) (base : BaseValues program)
    (groups : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F) :
    (inverseBlock program).EncodesAt 0 (by change 0 + 41 ≤ 42; decide)
      (localAssignment program base)
      (PiRLCRetainedPreservation.sourceAssignment program base groups products) ∧
    (flagBlock program).EncodesAt 41 (by change 41 + 1 ≤ 42; decide)
      (localAssignment program base)
      (PiRLCRetainedPreservation.sourceAssignment program base groups products) := by
  constructor
  · intro slot coordinate
    have slotZero : slot.val = 0 := by have bound := slot.isLt; change slot.val < 1 at bound; omega
    change (fromPhysical program base).getD (0 + (slot.val * 41 + coordinate.val)) 0 =
      (LowNormSlot.encode .field (PiRLCRetainedPreservation.sourceAssignment program base
        groups products (inverseSource program))).getD coordinate.val 0
    rw [inverseSource_value, slotZero]
    simp only [Nat.zero_mul, Nat.zero_add]
    rw [fromPhysical, coordinates, List.getD_append]
    rw [LowNormSlot.encode_length]
    exact coordinate.isLt
  · intro slot coordinate
    have slotZero : slot.val = 0 := by have bound := slot.isLt; change slot.val < 1 at bound; omega
    have coordinateZero : coordinate.val = 0 := by
      have bound := coordinate.isLt
      change coordinate.val < 1 at bound
      omega
    change (fromPhysical program base).getD (41 + (slot.val * 1 + coordinate.val)) 0 =
      (LowNormSlot.encode .bit (PiRLCRetainedPreservation.sourceAssignment program base
        groups products (flagSource program))).getD coordinate.val 0
    rw [flagSource_value, slotZero, coordinateZero]
    simp only [Nat.zero_mul, Nat.add_zero]
    rw [fromPhysical, coordinates_flag]
    rfl

theorem fromPhysical_valid {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
    (program : ApplicationProgram) (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (base : BaseValues program)
    (physical : RunningTransitionLayout.PhysicalHolds logicalWidth publicFits (sourceEnv program base)) :
    (fromPhysical program base).length = 42 ∧ normBounded 2 (fromPhysical program base) :=
  ⟨coordinates_length _ _, coordinates_norm _ _
    (RunningTransitionReducedRows.physical_firstScratch_boolean relation
      (sourceEnv program base) physical)⟩

/-- Decode arbitrary old field slots in full, then construct the canonical local witness.
This does not assume that the first old trit equals the represented field value. -/
def fromDecoded (inverse flag : Fin BalancedTernary.width → F) : List F :=
  coordinates (BalancedTernary.recompose (List.ofFn inverse))
    (BalancedTernary.recompose (List.ofFn flag))

theorem fromDecoded_preserves (inverse flag : Fin BalancedTernary.width → F) :
    BalancedTernary.recompose ((fromDecoded inverse flag).take 41) =
        BalancedTernary.recompose (List.ofFn inverse) ∧
      (fromDecoded inverse flag).getD 41 0 = BalancedTernary.recompose (List.ofFn flag) :=
  ⟨coordinates_inverse _ _, coordinates_flag _ _⟩

theorem fromDecoded_norm (inverse flag : Fin BalancedTernary.width → F)
    (bit : BalancedTernary.recompose (List.ofFn flag) = 0 ∨
      BalancedTernary.recompose (List.ofFn flag) = 1) :
    normBounded 2 (fromDecoded inverse flag) := coordinates_norm _ _ bit

end NightstreamFPrime.Export.Stage1.RunningTransitionReducedRetainedSemantics
