import NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransportExecution

/-!
Cache the canonical product recipes, selected source blocks and base width.
The existing transport owns all arithmetic and source-column selection.
The prepared record separates structural construction from coefficient reads.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.CachedAssignmentProducts

open NightstreamFPrime.Spec
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransport
open NightstreamFPrime.Export.Stage1.PerApplicationCanonicalAssignment

abbrev Program := Lifecycle.Stage1.Application.Program
abbrev BaseValues (program : Program) :=
  PerApplicationAssignmentTransportProducts.BaseValues program

/-- All runtime boundaries and block records are data. Their proof fields
link them to the same canonical transport used by canonicalRawValues. -/
structure Prepared (program : Program) where
  baseWidth : Nat
  baseWidth_eq : baseWidth = PiRLCProductPlan.baseSourceWidth program
  phi81 : Phi81QuotientRecipe
  phi81_eq : phi81 = phi81QuotientRecipe program
  challenge : CanonicalBlockAssignment.BlockValue
  challenge_eq : challenge =
    PerApplicationAssignmentBlocks.entry program phi81.challengeBlock

/-- Return a record, not a curried slot reader. Each selected block and the
numeric source width are constructed before any product callback is used. -/
@[noinline] def prepare (program : Program) : Prepared program :=
  let phi81 := phi81QuotientRecipe program
  { baseWidth := PiRLCProductPlan.baseSourceWidth program
    baseWidth_eq := rfl
    phi81 := phi81
    phi81_eq := rfl
    challenge := PerApplicationAssignmentBlocks.entry program phi81.challengeBlock
    challenge_eq := rfl }

/-- Reuse the total source reader with the stored numeric width. Fin.cast
changes only the erased bound proof, not the physical source coordinate. -/
@[noinline] private def sourceValue {program : Program}
    (prepared : Prepared program) (base : BaseValues program) (column : Nat) : F :=
  @SourceCompiler.sourceEnv prepared.baseWidth
    (fun source => base (Fin.cast prepared.baseWidth_eq source)) column

private theorem sourceEnv_cast {left right : Nat} (widthEq : left = right)
    (source : Fin right → F) :
    @SourceCompiler.sourceEnv left (fun column => source (Fin.cast widthEq column)) =
      SourceCompiler.sourceEnv source := by
  cases widthEq
  rfl

private theorem sourceValue_eq {program : Program}
    (prepared : Prepared program) (base : BaseValues program) (column : Nat) :
    sourceValue prepared base column = SourceCompiler.sourceEnv base column := by
  exact congrFun (sourceEnv_cast prepared.baseWidth_eq base) column

/-- Keep both original guards: the slot must exist, and its physical source
must lie within the base domain. No block is reconstructed here. -/
@[noinline] private def blockValue {program : Program}
    (prepared : Prepared program) (base : BaseValues program)
    (selected : CanonicalBlockAssignment.BlockValue) (slot : Nat) : F :=
  if bounded : slot < selected.block.slotCount then
    sourceValue prepared base (selected.block.source ⟨slot, bounded⟩).val
  else 0

private theorem blockValue_eq {program : Program}
    (prepared : Prepared program) (base : BaseValues program)
    (selected : CanonicalBlockAssignment.BlockValue)
    (kind : PerApplicationAssignmentPlan.BlockKind)
    (selectedEq : selected = PerApplicationAssignmentBlocks.entry program kind)
    (slot : Nat) :
    blockValue prepared base selected slot =
      PerApplicationAssignmentTransportProducts.baseBlockValue program base kind slot := by
  subst selected
  unfold blockValue PerApplicationAssignmentTransportProducts.baseBlockValue
  by_cases bounded :
      slot < (PerApplicationAssignmentBlocks.entry program kind).block.slotCount
  · simp only [dif_pos bounded, sourceValue_eq, SourceCompiler.sourceEnv,
      PerApplicationAssignmentBlocks.sourceIndex]
  · simp only [dif_neg bounded]

private def challengeRing {program : Program}
    (prepared : Prepared program) (base : BaseValues program)
    (descriptor : PiRLCProductSchedule.Descriptor) : RingF :=
  fun lane =>
    blockValue prepared base prepared.challenge
      (prepared.phi81.challengeSlotBase +
        descriptor.source.val * prepared.phi81.challengeSourceStride + lane.val) -
      Poseidon2.ofNat prepared.phi81.challengeShift

private def valueRing {program : Program}
    (prepared : Prepared program) (base : BaseValues program)
    (descriptor : PiRLCProductSchedule.Descriptor) : RingF :=
  fun lane =>
    sourceValue prepared base <|
      AffineRuns.sourceAt prepared.phi81.valueSources
        (PerApplicationAssignmentTransportProducts.invocationIndex prepared.phi81
          (descriptor.withLane lane))

private theorem challengeRing_eq {program : Program}
    (prepared : Prepared program) (base : BaseValues program)
    (descriptor : PiRLCProductSchedule.Descriptor) :
    challengeRing prepared base descriptor =
      PerApplicationAssignmentTransportProducts.challengeRing
        prepared.phi81 program base descriptor := by
  funext lane
  unfold challengeRing PerApplicationAssignmentTransportProducts.challengeRing
  rw [blockValue_eq prepared base prepared.challenge
    prepared.phi81.challengeBlock prepared.challenge_eq]

private theorem valueRing_eq {program : Program}
    (prepared : Prepared program) (base : BaseValues program)
    (descriptor : PiRLCProductSchedule.Descriptor) :
    valueRing prepared base descriptor =
      PerApplicationAssignmentTransportProducts.valueRing
        prepared.phi81 program base descriptor := by
  funext lane
  exact sourceValue_eq prepared base _

/-- Reuse the checked quotient formula on the prepared ring operands. -/
def groupValue {program : Program} (prepared : Prepared program)
    (base : BaseValues program)
    (invocation : Fin PiRLCProductSchedule.invocationCount) (_group : Nat) : F :=
  let ring := PiRLCProductRingSchedule.ringInvocation invocation
  let representative := PiRLCProductRingSchedule.laneInvocation ring PiRLCProductRingSchedule.zeroLane
  let descriptor := PiRLCProductSchedule.descriptor representative
  Phi81Relation.QuotientProduct.quotientCoeff
    (challengeRing prepared base descriptor) (valueRing prepared base descriptor)
    (PiRLCProductSchedule.descriptor invocation).lane

/-- Exact equality for every retained Phi81 quotient coefficient. -/
theorem groupValue_eq {program : Program} (prepared : Prepared program)
    (base : BaseValues program)
    (invocation : Fin PiRLCProductSchedule.invocationCount) (group : Fin 1) :
    groupValue prepared base invocation group.val =
      (PerApplicationAssignmentTransportExecution.canonicalRawValues program base).groupValue
        invocation group := by
  dsimp only [groupValue]
  rw [challengeRing_eq, valueRing_eq]
  change PerApplicationAssignmentTransportProducts.phi81GroupValue
    prepared.phi81 program base invocation group.val = _
  rw [prepared.phi81_eq]
  rfl

/-- Supply the same physical base and the cached Phi81 product callback. -/
def rawValues {program : Program} (prepared : Prepared program)
    (base : BaseValues program) : RawValues program where
  base := base
  groupValue := fun invocation group => groupValue prepared base invocation group.val

/-- The entire packet is unchanged, so the existing assignment schedule and
all of its refinement theorems apply without a new assignment premise. -/
theorem rawValues_eq {program : Program} (prepared : Prepared program)
    (base : BaseValues program) :
    rawValues prepared base =
      PerApplicationAssignmentTransportExecution.canonicalRawValues program base := by
  apply congrArg (RawValues.mk base)
  funext invocation group
  exact groupValue_eq prepared base invocation group

end NightstreamFPrime.Export.Stage1.CachedAssignmentProducts
