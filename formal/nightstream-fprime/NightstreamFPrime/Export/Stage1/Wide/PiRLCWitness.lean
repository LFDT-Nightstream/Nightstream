import NightstreamFPrime.Export.Stage1.Wide.PiRLCValues
import NightstreamFPrime.Export.Stage1.Wide.FieldAssignment

/-! Constructive compact PiRLC witness. The caller supplies the initial
transcript and right operands below this phase. Outputs and quotients are
computed directly; no R1CS ring-multiplication scratch is read or stored. -/

namespace NightstreamFPrime.Export.Stage1.Wide.PiRLCWitness

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation PiRlcWideSampler
open Spec.Folding.PiCCS.PaperJoint
open Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open PiRLCGeometry

structure InputsBefore {columns : Nat} (interface : Interface columns) : Prop where
  sampler : PiRlcWideSampler.Completeness.InputsBefore (sampler interface)
  value : ∀ ring lane entry, entry ∈ (interface.value ring lane).entries → entry.column.val < interface.start

def initial {columns : Nat} (interface : Interface columns) (base : Assignment F columns) : PiRLCValues.Initial :=
  SparseLayer.evalState base interface.initialState

def inputValues {columns : Nat} (interface : Interface columns) (base : Assignment F columns) : PiRLCValues.Values :=
  fun ring => Phi81ProductPlan.evalState base (interface.value ring)

def fieldValues {columns : Nat} (interface : Interface columns) (base : Assignment F columns) : Fin fieldBlock.slotCount → F :=
  PiRLCValues.fieldValue (initial interface base) (inputValues interface base)

def fieldBase {columns : Nat} (interface : Interface columns) (base : Assignment F columns) : Assignment F columns :=
  FieldAssignment.write (fieldStart interface) base (fieldValues interface base)

def assignment {columns : Nat} (interface : Interface columns) (base : Assignment F columns) : Assignment F columns :=
  Witness.assignment (sampler interface) (fieldBase interface base) (initial interface base)

theorem fieldBase_before {columns : Nat} (interface : Interface columns) (base : Assignment F columns)
    (column : Fin columns) (before : column.val < interface.start) :
    fieldBase interface base column = base column := by
  exact FieldAssignment.outside _ _ _ _ (Or.inl (by unfold fieldStart; omega))

theorem assignment_before {columns : Nat} (interface : Interface columns) (base : Assignment F columns)
    (column : Fin columns) (before : column.val < interface.start) :
    assignment interface base column = base column := by
  rw [assignment, Witness.assignment_outside _ _ _ _ (Or.inl before)]
  exact fieldBase_before interface base column before

/-- Completing this phase leaves both earlier and later allocations unchanged. -/
theorem assignment_outside {columns : Nat} (interface : Interface columns) (base : Assignment F columns)
    (column : Fin columns)
    (outside : column.val < interface.start ∨
      interface.start + PiRLCGeometry.coordinateCount ≤ column.val) :
    assignment interface base column = base column := by
  rcases outside with before | after
  · exact assignment_before interface base column before
  · rw [assignment, Witness.assignment_outside (sampler interface) (fieldBase interface base)
      (initial interface base) column (Or.inr (by
        change interface.start + 135813 ≤ column.val
        rw [PiRLCGeometry.coordinateCount_eq] at after
        omega))]
    apply FieldAssignment.outside
    right
    simpa only [PiRLCGeometry.coordinateCount, BatchPlan.coordinateCount_eq, fieldStart,
      LowNormBlock.Block.coordinateCount, Nat.add_assoc] using after

theorem disjoint_form {columns : Nat} (interface : Interface columns) (base : Assignment F columns)
    (form : SparseForm columns)
    (outside : ∀ entry ∈ form.entries, entry.column.val < interface.start ∨
      interface.start + PiRLCGeometry.coordinateCount ≤ entry.column.val) :
    form.eval (assignment interface base) = form.eval base := by
  apply FieldAssignment.form_eval_eq
  intro entry member
  exact assignment_outside interface base entry.column (outside entry member)

theorem fieldBase_initial {columns : Nat} (interface : Interface columns) (base : Assignment F columns)
    (before : InputsBefore interface) :
    SparseLayer.evalState (fieldBase interface base) interface.initialState = initial interface base := by
  funext lane
  apply FieldAssignment.form_eval_eq
  intro entry member
  exact fieldBase_before interface base entry.column (before.sampler.input lane entry member)

theorem assignment_initial {columns : Nat} (interface : Interface columns) (base : Assignment F columns)
    (before : InputsBefore interface) :
    SparseLayer.evalState (assignment interface base) interface.initialState = initial interface base := by
  funext lane
  apply FieldAssignment.form_eval_eq
  intro entry member
  exact assignment_before interface base entry.column (before.sampler.input lane entry member)

theorem one {columns : Nat} (interface : Interface columns) (base : Assignment F columns)
    (before : InputsBefore interface) (unit : base interface.oneColumn = 1) :
    assignment interface base interface.oneColumn = 1 := by
  rw [assignment_before interface base interface.oneColumn before.sampler.one, unit]

theorem sampler_rows {columns : Nat} (compiled : RangePlan.Compiled) (interface : Interface columns)
    (base : Assignment F columns) (before : InputsBefore interface) (unit : base interface.oneColumn = 1) :
    (BatchPlan.plan compiled (sampler interface)).RowsZero (assignment interface base) := by
  have complete := PiRlcWideSampler.Completeness.complete compiled (sampler interface)
    (fieldBase interface base) before.sampler
    (by change fieldBase interface base interface.oneColumn = 1
        rw [fieldBase_before interface base interface.oneColumn before.sampler.one]; exact unit)
  change (BatchPlan.plan compiled (sampler interface)).RowsZero
    (Witness.assignment (sampler interface) (fieldBase interface base)
      (SparseLayer.evalState (fieldBase interface base) interface.initialState)) at complete
  rw [fieldBase_initial interface base before] at complete
  exact complete

theorem fields_encodes {columns : Nat} (interface : Interface columns) (base : Assignment F columns) :
    fieldBlock.EncodesAt (fieldStart interface) (fieldFits interface) (assignment interface base)
      (fieldValues interface base) := by
  intro slot digit
  have preserved := Witness.assignment_outside (sampler interface) (fieldBase interface base)
    (initial interface base) (fieldBlock.column (fieldStart interface) (fieldFits interface) slot digit)
    (Or.inr (by change interface.start + 135813 ≤ interface.start + 135813 + _; omega))
  rw [show assignment interface base _ = fieldBase interface base _ from preserved]
  exact FieldAssignment.encodes (count := 2 * outputCount) (fieldStart interface)
    (fieldFits interface) base (fieldValues interface base) slot digit

theorem output_eq {columns : Nat} (interface : Interface columns) (base : Assignment F columns) (ring : RingIndex) :
    Phi81ProductPlan.evalState (assignment interface base) (PiRLCGeometry.output interface ring) =
      PiRLCValues.output (initial interface base) (inputValues interface base) ring := by
  funext lane
  change (fieldBlock.form _ _ (outputSlot ring lane)).eval _ = _
  rw [LowNormBlock.Block.form_eval _ _ _ _ _ (fields_encodes interface base)]
  exact PiRLCValues.fieldValue_output _ _ ring lane

theorem quotient_eq {columns : Nat} (interface : Interface columns) (base : Assignment F columns) (ring : RingIndex) :
    Phi81ProductPlan.evalState (assignment interface base) (PiRLCGeometry.quotient interface ring) =
      PiRLCValues.quotient (initial interface base) (inputValues interface base) ring := by
  funext lane
  change (fieldBlock.form _ _ (quotientSlot ring lane)).eval _ = _
  rw [LowNormBlock.Block.form_eval _ _ _ _ _ (fields_encodes interface base)]
  exact PiRLCValues.fieldValue_quotient _ _ ring lane

theorem value_eq {columns : Nat} (interface : Interface columns) (base : Assignment F columns)
    (before : InputsBefore interface) (ring : RingIndex) :
    Phi81ProductPlan.evalState (assignment interface base) (interface.value ring) = inputValues interface base ring := by
  funext lane
  apply FieldAssignment.form_eval_eq
  intro entry member
  exact assignment_before interface base entry.column (before.value ring lane entry member)

theorem challenge_eq {columns : Nat} (interface : Interface columns) (base : Assignment F columns)
    (before : InputsBefore interface) (ring : RingIndex) :
    PiRLCPlan.challenge (planInterface interface) (assignment interface base) ring =
      PiRLCValues.challenge (initial interface base) (PiRLCProductRingSchedule.descriptor ring).source.val := by
  unfold PiRLCPlan.challenge StateSemantics.state
  change Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Transcript.challengeAt
    (List.ofFn (SparseLayer.evalState (assignment interface base) interface.initialState)) _ = _
  rw [assignment_initial interface base before]
  rfl

theorem prior_eq {columns : Nat} (interface : Interface columns) (base : Assignment F columns) (ring : RingIndex) :
    Phi81ProductPlan.evalState (assignment interface base) (prior interface ring) =
      if first : (PiRLCProductRingSchedule.descriptor ring).source.val = 0 then fun _ => 0
      else PiRLCValues.output (initial interface base) (inputValues interface base)
        (previous (PiRLCProductRingSchedule.descriptor ring) first).invocation := by
  by_cases first : (PiRLCProductRingSchedule.descriptor ring).source.val = 0
  · simp only [prior, first, ↓reduceDIte]
    funext lane
    exact SparseForm.empty_eval _
  · simp only [prior, first, ↓reduceDIte]
    exact output_eq interface base _

theorem complete {columns : Nat} (compiled : RangePlan.Compiled) (interface : Interface columns)
    (base : Assignment F columns) (before : InputsBefore interface) (unit : base interface.oneColumn = 1) :
    (plan compiled interface).RowsZero (assignment interface base) := by
  apply PiRLCPlan.completeness compiled (planInterface interface) (assignment interface base)
    (one interface base before unit) (sampler_rows compiled interface base before unit)
  · intro ring
    change Phi81ProductPlan.evalState (assignment interface base) (PiRLCGeometry.output interface ring) =
      ringFAdd (Phi81ProductPlan.evalState (assignment interface base) (prior interface ring))
        (ringFMul (PiRLCPlan.challenge (planInterface interface) (assignment interface base) ring)
          (Phi81ProductPlan.evalState (assignment interface base) (interface.value ring)))
    rw [output_eq, prior_eq, challenge_eq interface base before, value_eq interface base before]
    exact PiRLCValues.output_step _ _ ring
  · intro ring
    change Phi81ProductPlan.evalState (assignment interface base) (PiRLCGeometry.quotient interface ring) = _
    rw [quotient_eq]
    unfold PiRLCPlan.honestQuotient
    change PiRLCValues.quotient _ _ ring =
      fun lane => Phi81Relation.QuotientProduct.quotientCoeff
        (PiRLCPlan.challenge (planInterface interface) (assignment interface base) ring)
        (Phi81ProductPlan.evalState (assignment interface base) (interface.value ring)) lane
    rw [challenge_eq interface base before, value_eq interface base before]
    rfl

theorem owned_norm {columns : Nat} (interface : Interface columns) (base : Assignment F columns)
    (column : Fin columns) (owned : interface.start ≤ column.val ∧ column.val < interface.start + PiRLCGeometry.coordinateCount) :
    centeredMagnitude (assignment interface base column) < 2 := by
  by_cases sample : column.val < interface.start + 135813
  · unfold assignment Witness.assignment
    rw [dif_pos (by exact ⟨owned.1, sample⟩)]
    exact PiRlcWideSampler.Norm.coordinate_norm _ _
  · rw [assignment, Witness.assignment_outside (sampler interface) (fieldBase interface base)
      (initial interface base) column (Or.inr (by exact Nat.le_of_not_gt sample))]
    apply FieldAssignment.owned_norm
    constructor
    · exact Nat.le_of_not_gt sample
    · simpa only [PiRLCGeometry.coordinateCount, BatchPlan.coordinateCount_eq,
        fieldStart, LowNormBlock.Block.coordinateCount, Nat.add_assoc] using owned.2

theorem preserves_norm {columns : Nat} (interface : Interface columns) (base : Assignment F columns)
    (bounded : ∀ column, centeredMagnitude (base column) < 2) :
    ∀ column, centeredMagnitude (assignment interface base column) < 2 := by
  intro column
  by_cases owned : interface.start ≤ column.val ∧
      column.val < interface.start + PiRLCGeometry.coordinateCount
  · exact owned_norm interface base column owned
  · rw [assignment_outside interface base column (by omega)]
    exact bounded column

end NightstreamFPrime.Export.Stage1.Wide.PiRLCWitness
