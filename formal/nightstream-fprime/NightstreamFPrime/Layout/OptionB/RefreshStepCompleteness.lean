import NightstreamFPrime.Layout.OptionB.RefreshStepBounds
import NightstreamFPrime.Layout.OptionB.LightStepCompleteness
import NightstreamFPrime.Lifecycle.PiDEC.v1_2.Completeness
import NightstreamFPrime.Lifecycle.PiRLC.v1_2.Completeness

/-! Completeness of the refresh step's gadgets (CW-3.7b). When the shared words lie below the
layout's offset, the application step holds on them, and the digest words and the fresh
instance's public input carry the hashes of their preimages, the five gadgets before Π_DEC can
be filled one after another (`completeGadgetsFrom`). The three prior hashes among them fill from
the first prior slot hash's offset (`completePriorHashesFrom`), as the finish step places them
without the application. Today's Π_DEC and Π_RLC phases fill their own rows from a prefix that
ends at their offset (`PiDEC.v1_2.Formal.completePrefix`,
`PiRLC.v1_2.Formal.completePrefix_with_phase`); `appendPrefix` joins such a phase to the step's
prefix. -/

namespace NightstreamFPrime.Layout.OptionB.RefreshStepCompleteness

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.OptionB

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Lifecycle.PiCCS.v1_2.Formal.CompletenessSupport
open NightstreamFPrime.Lifecycle.OptionB.RefreshStepLayout

/-- A prefix that starts where another ends joins it: the joined prefix has both operation
lists, the second prefix's values, and keeps the first prefix's values. -/
theorem appendPrefix {initial : Env} {base : Nat} (before : Sequence.Prefix initial base)
    {offset : Nat} (startEq : base + localLength before.operations = offset)
    (chunk : Sequence.Prefix before.current offset) :
    ∃ after : Sequence.Prefix initial base,
      after.operations = before.operations ++ chunk.operations ∧
      after.current = chunk.current ∧
      base + localLength after.operations = offset + localLength chunk.operations ∧
      Sequence.PreservesPrefix before after := by
  subst startEq
  have keep : ∀ index, index < base + localLength before.operations →
      chunk.current index = before.current index :=
    fun index bound => chunk.agrees index (Or.inl bound)
  refine ⟨{
    current := chunk.current
    operations := before.operations ++ chunk.operations
    agrees := ?_
    scope := ?_
    rows := ?_ }, rfl, rfl, by rw [Sequence.localLength_append]; omega, ⟨?_, ?_⟩⟩
  · rw [Sequence.localLength_append]
    intro index outside
    rcases outside with below | above
    · rw [chunk.agrees index (Or.inl (by omega))]
      exact before.agrees index (Or.inl below)
    · rw [chunk.agrees index (Or.inr (by omega))]
      exact before.agrees index (Or.inr (by omega))
  · intro expression member
    rw [flatConstraints_append, List.mem_append] at member
    rw [Sequence.localLength_append, ← Nat.add_assoc]
    rcases member with member | member
    · exact Expr.VarsBelow.mono _ (before.scope expression member) (by omega)
    · exact chunk.scope expression member
  · intro expression member
    rw [flatConstraints_append, List.mem_append] at member
    rcases member with member | member
    · rw [expression.eval_eq_of_agree_below (base + localLength before.operations)
        chunk.current before.current (before.scope expression member) keep]
      exact before.rows expression member
    · exact chunk.rows expression member
  · rw [Sequence.localLength_append]
    omega
  · exact keep

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}
  (program : Stage1.Application.Program) (circuit : Fin 2)
  (interface : RefreshStepLayout.Interface logicalWidth publicFits program.witnessWordCount)
  (offset : Nat)

omit circuit in
/-- Completeness of the refresh step's two prior slot hashes and prior state hash after a prefix
that ends at the first prior slot hash's offset, when the shared words lie below the layout's
offset and the prior digest words and the fresh instance's public input carry the hashes of their
preimages. -/
theorem completePriorHashesFrom
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    {initial : Env} {base : Nat} (before : Sequence.Prefix initial base)
    (startEq : base + localLength before.operations = priorSlotZeroOffset program interface offset)
    (below : RefreshStepBounds.InputsBelow interface offset)
    (priorSlots : ∀ slot,
      OutputHash.SpecHolds (priorSlotInterface program interface slot) offset before.current)
    (priorState :
      PriorStateHash.SpecHolds (priorStateInterface program interface) offset before.current) :
    ∃ after : Sequence.Prefix initial base,
      after.operations = before.operations ++
        [Formal.childOp "refresh.prior_slot_0" (priorSlotCircuit program interface 0)
            (priorSlotZeroOffset program interface offset),
          Formal.childOp "refresh.prior_slot_1" (priorSlotCircuit program interface 1)
            (priorSlotOneOffset program interface offset),
          Formal.childOp "refresh.prior_state" (priorStateCircuit program interface)
            (priorStateOffset program interface offset)] ∧
      base + localLength after.operations = nextStateOffset program interface offset ∧
      Sequence.PreservesPrefix before after := by
  have assumptionsAt := RefreshStepBounds.assumptions program relation 0 interface offset below
  have offsetLe : offset ≤ base + localLength before.operations := by
    rw [startEq]
    exact Nat.le_add_right offset _
  have agreesAt : ∀ completed : Sequence.Prefix initial base,
      Sequence.PreservesPrefix before completed →
      ∀ index, index < offset → completed.current index = before.current index :=
    fun completed preserves index bound => preserves.values index (by omega)
  have here : Sequence.PreservesPrefix before before := ⟨Nat.le_refl _, fun _ _ => rfl⟩
  have priorSlotAt : ∀ slot (completed : Sequence.Prefix initial base),
      Sequence.PreservesPrefix before completed → ∀ childOffset,
      OutputHash.SpecHolds (priorSlotInterface program interface slot) childOffset
        completed.current :=
    fun slot completed preserves _ => OutputHash.specHolds_of_agree_below
      (priorSlotInterface program interface slot) offset before.current completed.current
      ⟨LightStepBounds.slotPreimage_below _ _ (below.prior slot), below.priorDigest slot⟩
      (agreesAt completed preserves) (priorSlots slot)
  have priorStateAt : ∀ completed : Sequence.Prefix initial base,
      Sequence.PreservesPrefix before completed → ∀ childOffset,
      PriorStateHash.SpecHolds (priorStateInterface program interface) childOffset
        completed.current :=
    fun completed preserves _ => PriorStateHash.specHolds_of_agree_below
      (priorStateInterface program interface) offset before.current completed.current
      ⟨RefreshStepBounds.statePreimage_below program interface _ _ _ _ offset below.verifierKeys
          below.initialState below.iteration below.currentState below.priorPc below.priorDigest,
        fun column => below.freshPublicInput _ column⟩
      (agreesAt completed preserves) priorState
  obtain ⟨built1, agrees1, rows1⟩ := OutputHash.completeness
    (priorSlotInterface program interface 0) before.current
    (priorSlotZeroOffset program interface offset) (assumptionsAt before.current).priorSlotZero
    (priorSlotAt 0 before here _)
  obtain ⟨p1, ops1, end1, preserves1, -⟩ := appendBuiltAt before "refresh.prior_slot_0"
    (priorSlotCircuit program interface 0) (priorSlotZeroOffset program interface offset) startEq
    (OutputHash.flatConstraints_varsBelow _ _ (env := before.current)
      (assumptionsAt before.current).priorSlotZero)
    built1 agrees1 rows1
  obtain ⟨built2, agrees2, rows2⟩ := OutputHash.completeness
    (priorSlotInterface program interface 1) p1.current
    (priorSlotOneOffset program interface offset) (assumptionsAt p1.current).priorSlotOne
    (priorSlotAt 1 p1 preserves1 _)
  obtain ⟨p2, ops2, end2, preserves2, -⟩ := appendBuiltAt p1 "refresh.prior_slot_1"
    (priorSlotCircuit program interface 1) (priorSlotOneOffset program interface offset) end1
    (OutputHash.flatConstraints_varsBelow _ _ (env := p1.current)
      (assumptionsAt p1.current).priorSlotOne)
    built2 agrees2 rows2
  have through2 := preserves1.trans preserves2
  obtain ⟨built3, agrees3, rows3⟩ := PriorStateHash.completeness
    (priorStateInterface program interface) p2.current (priorStateOffset program interface offset)
    (assumptionsAt p2.current).priorState (priorStateAt p2 through2 _)
  obtain ⟨p3, ops3, end3, preserves3, -⟩ := appendBuiltAt p2 "refresh.prior_state"
    (priorStateCircuit program interface) (priorStateOffset program interface offset) end2
    (PriorStateHash.flatConstraints_varsBelow _ _ (env := p2.current)
      (assumptionsAt p2.current).priorState)
    built3 agrees3 rows3
  refine ⟨p3, ?_, end3, through2.trans preserves3⟩
  rw [ops3, ops2, ops1]
  simp only [List.append_assoc, List.cons_append, List.nil_append]

/-- Completeness of the refresh step's five gadgets before Π_DEC after a prefix that ends at
the layout's offset. The gadgets do not read the relation, which only fixes the layout's input
scoping. -/
theorem completeGadgetsFrom (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    {initial : Env} {base : Nat} (before : Sequence.Prefix initial base)
    (startEq : base + localLength before.operations = offset)
    (below : RefreshStepBounds.InputsBelow interface offset)
    (application : (applicationCircuit program interface).spec offset before.current)
    (priorSlots : ∀ slot,
      OutputHash.SpecHolds (priorSlotInterface program interface slot) offset before.current)
    (priorState :
      PriorStateHash.SpecHolds (priorStateInterface program interface) offset before.current)
    (nextState :
      OutputHash.SpecHolds (nextStateInterface program circuit interface) offset before.current) :
    ∃ after : Sequence.Prefix initial base,
      after.operations = before.operations ++ gadgetOps program circuit interface offset ∧
      base + localLength after.operations = decOffset program circuit interface offset ∧
      Sequence.PreservesPrefix before after := by
  have assumptionsAt := RefreshStepBounds.assumptions program relation circuit interface offset
    below
  have agreesAt : ∀ completed : Sequence.Prefix initial base,
      Sequence.PreservesPrefix before completed →
      ∀ index, index < offset → completed.current index = before.current index :=
    fun completed preserves index bound => preserves.values index (by omega)
  have nextStateAt : ∀ completed : Sequence.Prefix initial base,
      Sequence.PreservesPrefix before completed → ∀ childOffset,
      OutputHash.SpecHolds (nextStateInterface program circuit interface) childOffset
        completed.current :=
    fun completed preserves _ => OutputHash.specHolds_of_agree_below
      (nextStateInterface program circuit interface) offset before.current completed.current
      ⟨RefreshStepBounds.statePreimage_below program interface _ _ _ _ offset below.verifierKeys
          below.initialState (Expr.VarsBelow.add _ _ _ below.iteration trivial)
          below.nextState trivial below.nextDigest,
        below.outputDigest⟩
      (agreesAt completed preserves) nextState
  obtain ⟨built1, agrees1, rows1⟩ := (applicationCircuit program interface).completeness
    before.current offset (assumptionsAt before.current).application application
  obtain ⟨p1, ops1, end1, preserves1, -⟩ := appendBuiltAt before "refresh.application"
    (applicationCircuit program interface) offset startEq
    (Stage1.Application.Program.scope program (applicationInterface program interface) offset
      before.current ⟨below.currentState, below.witness, below.nextState⟩
      (assumptionsAt before.current).application) built1 agrees1 rows1
  obtain ⟨p4, ops4, end4, preserves4⟩ := completePriorHashesFrom program interface offset relation
    p1 end1 below
    (fun slot => OutputHash.specHolds_of_agree_below (priorSlotInterface program interface slot)
      offset before.current p1.current
      ⟨LightStepBounds.slotPreimage_below _ _ (below.prior slot), below.priorDigest slot⟩
      (agreesAt p1 preserves1) (priorSlots slot))
    (PriorStateHash.specHolds_of_agree_below (priorStateInterface program interface) offset
      before.current p1.current
      ⟨RefreshStepBounds.statePreimage_below program interface _ _ _ _ offset below.verifierKeys
          below.initialState below.iteration below.currentState below.priorPc below.priorDigest,
        fun column => below.freshPublicInput _ column⟩
      (agreesAt p1 preserves1) priorState)
  have through4 := preserves1.trans preserves4
  obtain ⟨built5, agrees5, rows5⟩ := OutputHash.completeness
    (nextStateInterface program circuit interface) p4.current
    (nextStateOffset program interface offset) (assumptionsAt p4.current).nextState
    (nextStateAt p4 through4 _)
  obtain ⟨p5, ops5, end5, preserves5, -⟩ := appendBuiltAt p4 "refresh.next_state"
    (nextStateCircuit program circuit interface) (nextStateOffset program interface offset) end4
    (OutputHash.flatConstraints_varsBelow _ _ (env := p4.current)
      (assumptionsAt p4.current).nextState)
    built5 agrees5 rows5
  refine ⟨p5, ?_, end5, through4.trans preserves5⟩
  rw [ops5, ops4, ops1]
  simp only [gadgetOps, List.append_assoc, List.cons_append, List.nil_append]

/-- Today's Π_DEC fills its rows after a prefix that ends at its offset, when its phase holds
on the prefix's values. -/
theorem completeDecFrom (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    {initial : Env} {base : Nat} (before : Sequence.Prefix initial base)
    (startEq : base + localLength before.operations = decOffset program circuit interface offset)
    (below : RefreshStepBounds.InputsBelow interface offset)
    (phase : Lifecycle.PiDEC.v1_2.Semantics.PhaseHolds relation ajtai
      (decInterface program interface) (decOffset program circuit interface offset)
      before.current) :
    ∃ after : Sequence.Prefix initial base,
      after.operations = before.operations ++
        Circuit.ops (Lifecycle.PiDEC.v1_2.Formal.main relation (decInterface program interface))
          (decOffset program circuit interface offset) ∧
      base + localLength after.operations = piccsOffset program circuit interface offset ∧
      Sequence.PreservesPrefix before after := by
  obtain ⟨chunk, chunkOps⟩ := Lifecycle.PiDEC.v1_2.Formal.completePrefix relation ajtai
    (decInterface program interface) before.current (decOffset program circuit interface offset)
    (RefreshStepBounds.decAssumptions program relation circuit interface offset below
      before.current) phase
  obtain ⟨after, afterOps, -, endEq, preserves⟩ := appendPrefix before startEq chunk
  refine ⟨after, ?_, ?_, preserves⟩
  · rw [afterOps, chunkOps]
    rfl
  · rw [endEq, chunkOps, ← Lifecycle.PiDEC.v1_2.Formal.main_ops,
      Lifecycle.PiDEC.v1_2.Formal.localLength_eq]
    rfl

omit program circuit interface offset in
/-- Today's Π_RLC fills its rows after a prefix that ends at its offset `start`, and its phase
then holds on the filled values. The interface and the offset are variables here, so the kernel
never evaluates a layout length. -/
theorem completeRlcPhaseFrom (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (rlc : Lifecycle.PiRLC.v1_2.Formal.Interface logicalWidth publicFits) (start : Nat)
    {initial : Env} {base : Nat} (before : Sequence.Prefix initial base)
    (startEq : base + localLength before.operations = start)
    (assumptions : Lifecycle.PiRLC.v1_2.Formal.Assumptions relation rlc start before.current) :
    ∃ after : Sequence.Prefix initial base,
      after.operations = before.operations ++
        Circuit.ops (Lifecycle.PiRLC.v1_2.Formal.main relation rlc) start ∧
      base + localLength after.operations = Lifecycle.PiRLC.v1_2.Formal.finalOffset start ∧
      Sequence.PreservesPrefix before after ∧
      Lifecycle.PiRLC.v1_2.Semantics.PhaseHolds relation ajtai rlc start after.current := by
  obtain ⟨chunk, chunkOps, phase⟩ := Lifecycle.PiRLC.v1_2.Formal.completePrefix_with_phase
    relation ajtai rlc before.current start assumptions
  obtain ⟨after, afterOps, currentEq, endEq, preserves⟩ := appendPrefix before startEq chunk
  refine ⟨after, ?_, ?_, preserves, currentEq ▸ phase⟩
  · rw [afterOps, chunkOps]
    rfl
  · rw [endEq, chunkOps, ← Lifecycle.PiRLC.v1_2.Formal.main_ops,
      Lifecycle.PiRLC.v1_2.Formal.localLength_eq]
    have samplerCount := Lifecycle.PiRLC.v1_2.SamplerChain.counts.1
    unfold Lifecycle.PiRLC.v1_2.Formal.finalOffset
      Lifecycle.PiRLC.v1_2.Formal.outputBindingOffset Lifecycle.PiRLC.v1_2.Formal.evalAOffset
      Lifecycle.PiRLC.v1_2.Formal.evalKOffset Lifecycle.PiRLC.v1_2.Formal.publicInputOffset
      Lifecycle.PiRLC.v1_2.Formal.commitmentOffset Lifecycle.PiRLC.v1_2.Formal.samplerOffset
      Lifecycle.PiRLC.v1_2.Formal.logicalPrivateCount
    omega

end NightstreamFPrime.Layout.OptionB.RefreshStepCompleteness
