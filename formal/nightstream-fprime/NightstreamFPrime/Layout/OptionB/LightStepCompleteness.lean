import NightstreamFPrime.Layout.OptionB.LightStepBounds
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.Completeness.Core

/-! Completeness of the light step's gadgets (assembler A3c). When the shared words lie below the
layout's offset, the application step holds on them, the active slot's depth is below the cap,
and the digest words and the fresh instance's public input carry the hashes of their preimages,
the nine gadgets before the light Π_CCS can be filled one after another
(`completeGadgetsFrom`). The application, range and hash gadgets compute or check only shared
words, so their specs carry over from the starting values; the high-bit rows read the range
gadget's bits, which the range rows give (`DepthGuard.highBitsZero_of_lt`). The slot-select rows
come from honest words by `SlotSelect.rows_of_select`. -/

namespace NightstreamFPrime.Layout.OptionB.LightStepCompleteness

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.OptionB

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Gadgets.Range
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Lifecycle.PiCCS.v1_2.Formal.CompletenessSupport
open NightstreamFPrime.Lifecycle.OptionB.LightStepLayout

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}
  (program : Stage1.Application.Program) (circuit : Fin 2)
  (interface : LightStepLayout.Interface logicalWidth publicFits program.witnessWordCount)
  (offset : Nat)

/-- Completeness of the light step's nine gadgets after a prefix that ends at the layout's offset.
The gadgets do not read the relation, which only fixes the layout's input scoping. -/
theorem completeGadgetsFrom (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    {initial : Env} {base : Nat} (before : Sequence.Prefix initial base)
    (startEq : base + localLength before.operations = offset)
    (below : LightStepBounds.InputsBelow interface offset)
    (application : (applicationCircuit program interface).spec offset before.current)
    (depthSmall : (interface.active.depth.eval before.current).val <
      Folding.Schedule.productionLightFoldCap)
    (priorSlots : ∀ slot,
      OutputHash.SpecHolds (priorSlotInterface program interface slot) offset before.current)
    (priorState :
      PriorStateHash.SpecHolds (priorStateInterface program interface) offset before.current)
    (nextSlots : ∀ slot,
      OutputHash.SpecHolds (nextSlotInterface program interface slot) offset before.current)
    (nextState :
      OutputHash.SpecHolds (nextStateInterface program circuit interface) offset before.current) :
    ∃ after : Sequence.Prefix initial base,
      after.operations = before.operations ++ gadgetOps program circuit interface offset ∧
      base + localLength after.operations = piccsOffset program circuit interface offset ∧
      Sequence.PreservesPrefix before after := by
  have assumptionsAt := LightStepBounds.assumptions program relation circuit interface offset
    below
  have agreesAt : ∀ completed : Sequence.Prefix initial base,
      Sequence.PreservesPrefix before completed →
      ∀ index, index < offset → completed.current index = before.current index :=
    fun completed preserves index bound => preserves.values index (by omega)
  have priorSlotAt : ∀ slot (completed : Sequence.Prefix initial base),
      Sequence.PreservesPrefix before completed → ∀ childOffset,
      OutputHash.SpecHolds (priorSlotInterface program interface slot) childOffset
        completed.current :=
    fun slot completed preserves _ => OutputHash.specHolds_of_agree_below
      (priorSlotInterface program interface slot) offset before.current completed.current
      ⟨LightStepBounds.slotPreimage_below _ _ (below.prior slot), below.priorDigest slot⟩
      (agreesAt completed preserves) (priorSlots slot)
  have nextSlotAt : ∀ slot (completed : Sequence.Prefix initial base),
      Sequence.PreservesPrefix before completed → ∀ childOffset,
      OutputHash.SpecHolds (nextSlotInterface program interface slot) childOffset
        completed.current :=
    fun slot completed preserves _ => OutputHash.specHolds_of_agree_below
      (nextSlotInterface program interface slot) offset before.current completed.current
      ⟨LightStepBounds.slotPreimage_below _ _ (below.next slot), below.nextDigest slot⟩
      (agreesAt completed preserves) (nextSlots slot)
  have priorStateAt : ∀ completed : Sequence.Prefix initial base,
      Sequence.PreservesPrefix before completed → ∀ childOffset,
      PriorStateHash.SpecHolds (priorStateInterface program interface) childOffset
        completed.current :=
    fun completed preserves _ => PriorStateHash.specHolds_of_agree_below
      (priorStateInterface program interface) offset before.current completed.current
      ⟨LightStepBounds.statePreimage_below program interface _ _ _ _ offset below.verifierKeys
          below.initialState below.iteration below.currentState below.priorPc below.priorDigest,
        fun column => below.freshPublicInput _ column⟩
      (agreesAt completed preserves) priorState
  have nextStateAt : ∀ completed : Sequence.Prefix initial base,
      Sequence.PreservesPrefix before completed → ∀ childOffset,
      OutputHash.SpecHolds (nextStateInterface program circuit interface) childOffset
        completed.current :=
    fun completed preserves _ => OutputHash.specHolds_of_agree_below
      (nextStateInterface program circuit interface) offset before.current completed.current
      ⟨LightStepBounds.statePreimage_below program interface _ _ _ _ offset below.verifierKeys
          below.initialState (Expr.VarsBelow.add _ _ _ below.iteration trivial)
          below.nextState trivial below.nextDigest,
        below.outputDigest⟩
      (agreesAt completed preserves) nextState
  -- Application step and depth guard.
  obtain ⟨built1, agrees1, rows1⟩ := (applicationCircuit program interface).completeness
    before.current offset (assumptionsAt before.current).application application
  obtain ⟨p1, ops1, end1, preserves1, -⟩ := appendBuiltAt before "light.application"
    (applicationCircuit program interface) offset startEq
    (Stage1.Application.Program.scope program (applicationInterface program interface) offset
      before.current ⟨below.currentState, below.witness, below.nextState⟩
      (assumptionsAt before.current).application) built1 agrees1 rows1
  obtain ⟨built2, agrees2, rows2⟩ := CanonicalU64.complete (rangeInterface program interface)
    p1.current (rangeOffset program interface offset) (assumptionsAt p1.current).range
  obtain ⟨p2, ops2, end2, preserves2, rangeRows⟩ := appendBuiltAt p1 "light.depth_range"
    (rangeCircuit program interface) (rangeOffset program interface offset) end1
    (fun expression member => by
      have scope := CanonicalU64.flatConstraints_varsBelow (rangeInterface program interface)
        (rangeOffset program interface offset) (assumptionsAt p1.current).range expression member
      rw [← CanonicalU64.localLength_eq (rangeInterface program interface)
        (rangeOffset program interface offset)] at scope
      exact scope)
    built2 agrees2 rows2
  have rangeSpec := (rangeCircuit program interface).soundness p2.current
    (rangeOffset program interface offset) (assumptionsAt p2.current).range
    (holdsFlat_implies_holds _ _ rangeRows)
  have depthEq : interface.active.depth.eval p2.current = interface.active.depth.eval before.current :=
    interface.active.depth.eval_eq_of_agree_below offset p2.current before.current
      below.activeDepth (agreesAt p2 (preserves1.trans preserves2))
  have highSpec := DepthGuard.highBitsZero_of_lt (rangeInterface program interface) p2.current
    (rangeOffset program interface offset) rangeSpec (by
      change (interface.active.depth.eval p2.current).val < _
      rw [depthEq]
      exact depthSmall)
  obtain ⟨built3, agrees3, rows3⟩ := (highBitsCircuit program interface offset).completeness
    p2.current (highBitsOffset program interface offset) trivial highSpec
  have rangeLength : localLength (Circuit.ops (rangeCircuit program interface).main
      (rangeOffset program interface offset)) = CanonicalU64.auxiliaryCount :=
    CanonicalU64.localLength_eq _ _
  obtain ⟨p3, ops3, end3, preserves3, -⟩ := appendBuiltAt p2 "light.depth_high_bits"
    (highBitsCircuit program interface offset) (highBitsOffset program interface offset) end2
    (fun expression member => by
      apply Expr.VarsBelow.mono expression
        (DepthGuard.highBitOps_varsBelow _ expression member)
      unfold highBitsOffset Formal.nextOffset Formal.childLength
      rw [rangeLength]
      simp only [CanonicalU64.bitCount, CanonicalU64.auxiliaryCount]
      omega)
    built3 agrees3 rows3
  -- Prior slot and state hashes.
  have through3 := preserves1.trans (preserves2.trans preserves3)
  obtain ⟨built4, agrees4, rows4⟩ := OutputHash.completeness
    (priorSlotInterface program interface 0) p3.current
    (priorSlotZeroOffset program interface offset) (assumptionsAt p3.current).priorSlotZero
    (priorSlotAt 0 p3 through3 _)
  obtain ⟨p4, ops4, end4, preserves4, -⟩ := appendBuiltAt p3 "light.prior_slot_0"
    (priorSlotCircuit program interface 0) (priorSlotZeroOffset program interface offset) end3
    (OutputHash.flatConstraints_varsBelow _ _ (env := p3.current) (assumptionsAt p3.current).priorSlotZero)
    built4 agrees4 rows4
  have through4 := through3.trans preserves4
  obtain ⟨built5, agrees5, rows5⟩ := OutputHash.completeness
    (priorSlotInterface program interface 1) p4.current
    (priorSlotOneOffset program interface offset) (assumptionsAt p4.current).priorSlotOne
    (priorSlotAt 1 p4 through4 _)
  obtain ⟨p5, ops5, end5, preserves5, -⟩ := appendBuiltAt p4 "light.prior_slot_1"
    (priorSlotCircuit program interface 1) (priorSlotOneOffset program interface offset) end4
    (OutputHash.flatConstraints_varsBelow _ _ (env := p4.current) (assumptionsAt p4.current).priorSlotOne)
    built5 agrees5 rows5
  have through5 := through4.trans preserves5
  obtain ⟨built6, agrees6, rows6⟩ := PriorStateHash.completeness
    (priorStateInterface program interface) p5.current (priorStateOffset program interface offset)
    (assumptionsAt p5.current).priorState
    (priorStateAt p5 through5 _)
  obtain ⟨p6, ops6, end6, preserves6, -⟩ := appendBuiltAt p5 "light.prior_state"
    (priorStateCircuit program interface) (priorStateOffset program interface offset) end5
    (PriorStateHash.flatConstraints_varsBelow _ _ (env := p5.current) (assumptionsAt p5.current).priorState)
    built6 agrees6 rows6
  -- Next slot and state hashes.
  have through6 := through5.trans preserves6
  obtain ⟨built7, agrees7, rows7⟩ := OutputHash.completeness
    (nextSlotInterface program interface 0) p6.current
    (nextSlotZeroOffset program interface offset) (assumptionsAt p6.current).nextSlotZero
    (nextSlotAt 0 p6 through6 _)
  obtain ⟨p7, ops7, end7, preserves7, -⟩ := appendBuiltAt p6 "light.next_slot_0"
    (nextSlotCircuit program interface 0) (nextSlotZeroOffset program interface offset) end6
    (OutputHash.flatConstraints_varsBelow _ _ (env := p6.current) (assumptionsAt p6.current).nextSlotZero)
    built7 agrees7 rows7
  have through7 := through6.trans preserves7
  obtain ⟨built8, agrees8, rows8⟩ := OutputHash.completeness
    (nextSlotInterface program interface 1) p7.current
    (nextSlotOneOffset program interface offset) (assumptionsAt p7.current).nextSlotOne
    (nextSlotAt 1 p7 through7 _)
  obtain ⟨p8, ops8, end8, preserves8, -⟩ := appendBuiltAt p7 "light.next_slot_1"
    (nextSlotCircuit program interface 1) (nextSlotOneOffset program interface offset) end7
    (OutputHash.flatConstraints_varsBelow _ _ (env := p7.current) (assumptionsAt p7.current).nextSlotOne)
    built8 agrees8 rows8
  have through8 := through7.trans preserves8
  obtain ⟨built9, agrees9, rows9⟩ := OutputHash.completeness
    (nextStateInterface program circuit interface) p8.current
    (nextStateOffset program interface offset) (assumptionsAt p8.current).nextState
    (nextStateAt p8 through8 _)
  obtain ⟨p9, ops9, end9, preserves9, -⟩ := appendBuiltAt p8 "light.next_state"
    (nextStateCircuit program circuit interface) (nextStateOffset program interface offset) end8
    (OutputHash.flatConstraints_varsBelow _ _ (env := p8.current) (assumptionsAt p8.current).nextState)
    built9 agrees9 rows9
  refine ⟨p9, ?_, end9, through8.trans preserves9⟩
  rw [ops9, ops8, ops7, ops6, ops5, ops4, ops3, ops2, ops1]
  simp only [gadgetOps, List.append_assoc, List.cons_append, List.nil_append]

end NightstreamFPrime.Layout.OptionB.LightStepCompleteness
