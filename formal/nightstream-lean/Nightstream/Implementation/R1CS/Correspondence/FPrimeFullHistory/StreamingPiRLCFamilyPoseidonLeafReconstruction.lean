import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafCertificate

/-!
Contract: same-assignment source reconstruction for one generated production
PiRLC Poseidon2 leaf.

Assurance tier: artifact-checked leaf reconstruction.

Owns: source-column ownership by the 86 compact slots, radix-3 run scaling,
bounded exact port-layout checks, and derivation of every per-run link from
constant-one plus one value equality per used source slot.

Does not own: the absolute Rust column map, derivation of slot values from
retained source rows, selector authority, row satisfaction, lifecycle
semantics, or permission to remove constraints.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 65536

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafReconstruction

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel.Wire
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports

def outputColumn? (step : Wire.Step) : Option SourceColumn :=
  if step.output.constant = 0 then
    match step.output.terms with
    | [term] =>
        if term.coefficient = 1 then some term.column else none
    | _ => none
  else
    none

def findLocalSlot :
    List Wire.Step → Nat → SourceColumn → Option Slot
  | [], _, _ => none
  | step :: tail, index, column =>
      if outputColumn? step = some column then
        if bounded : index < 86 then
          some (.local ⟨index, bounded⟩)
        else
          none
      else
        findLocalSlot tail (index + 1) column

def sourceSlot (steps : List Wire.Step) : SourceColumn → Option Slot
  | .externalA lane => some (.externalA lane)
  | .externalB lane => some (.externalB lane)
  | column@(.local _) => findLocalSlot steps 0 column

def slotValue (final : FinalAssignment) (slot : Slot) : F :=
  geometricAction { slot, initial := 1, ratio := 3 } final

/-- Exact remaining source-image boundary. Every used source value must equal
the radix-3 value of its verifier-selected final slot. -/
def SourceSlotsLinked
    (steps : List Wire.Step) (source : SourceAssignment)
    (final : FinalAssignment) : Prop :=
  ∀ column slot, sourceSlot steps column = some slot →
    slotValue final slot = sourceValue source column

def expectedExplicit (constant : F) : List ExplicitTerm :=
  if constant = 0 then [] else [{ column := .one, coefficient := constant }]

structure RunLayoutMatches
    (steps : List Wire.Step) (term : SourceTerm) (run : GeometricRun) : Prop where
  slot : sourceSlot steps term.column = some run.slot
  initial : run.initial = term.coefficient
  ratio : run.ratio = 3

instance (steps : List Wire.Step) (term : SourceTerm) (run : GeometricRun) :
    Decidable (RunLayoutMatches steps term run) :=
  if slot : sourceSlot steps term.column = some run.slot then
    if initial : run.initial = term.coefficient then
      if ratio : run.ratio = 3 then
        isTrue ⟨slot, initial, ratio⟩
      else
        isFalse (fun matched => ratio matched.ratio)
    else
      isFalse (fun matched => initial matched.initial)
  else
    isFalse (fun matched => slot matched.slot)

structure PortLayoutMatches
    (steps : List Wire.Step) (value : SourceLinearCombination)
    (port : Port) : Prop where
  explicit : port.explicit = expectedExplicit value.constant
  runs : List.Forall₂ (RunLayoutMatches steps) value.terms port.geometric

instance (steps : List Wire.Step) (value : SourceLinearCombination)
    (port : Port) : Decidable (PortLayoutMatches steps value port) :=
  if explicit : port.explicit = expectedExplicit value.constant then
    if runs : List.Forall₂ (RunLayoutMatches steps)
        value.terms port.geometric then
      isTrue ⟨explicit, runs⟩
    else
      isFalse (fun matched => runs matched.runs)
  else
    isFalse (fun matched => explicit matched.explicit)

structure StepPortLayoutsMatch
    (steps : List Wire.Step) (step : Wire.Step) (row : Wire.Row) : Prop where
  input : PortLayoutMatches steps step.input
    (row.port Role.sboxInput.index)
  output : PortLayoutMatches steps step.output
    (row.port Role.c.index)

instance (steps : List Wire.Step) (step : Wire.Step) (row : Wire.Row) :
    Decidable (StepPortLayoutsMatch steps step row) :=
  if input : PortLayoutMatches steps step.input
      (row.port Role.sboxInput.index) then
    if output : PortLayoutMatches steps step.output
        (row.port Role.c.index) then
      isTrue ⟨input, output⟩
    else
      isFalse (fun matched => output matched.output)
  else
    isFalse (fun matched => input matched.input)

private theorem geometricCoefficient_scale (initial : F) (index : Nat) :
    geometricCoefficient initial 3 index =
      initial * geometricCoefficient 1 3 index := by
  induction index with
  | zero => exact (Fin.mul_one initial).symm
  | succ index inductionHypothesis =>
      simp only [geometricCoefficient]
      rw [inductionHypothesis, Fin.mul_assoc]

private theorem geometric_sum_scale
    (initial : F) (final : FinalAssignment) (slot : Slot) :
    ∀ indices : List (Fin 41),
      sum (indices.map fun index =>
        geometricCoefficient initial 3 index.val * final.digit slot index) =
      initial * sum (indices.map fun index =>
        geometricCoefficient 1 3 index.val * final.digit slot index) := by
  intro indices
  induction indices with
  | nil => exact (Fin.mul_zero initial).symm
  | cons index tail inductionHypothesis =>
      simp only [List.map_cons, sum]
      rw [geometricCoefficient_scale, inductionHypothesis, Fin.mul_assoc]
      exact (Lean.Grind.Fin.left_distrib _ _ _).symm

theorem geometricAction_eq_scaled_slotValue
    (run : GeometricRun) (final : FinalAssignment)
    (ratio : run.ratio = 3) :
    geometricAction run final = run.initial * slotValue final run.slot := by
  unfold geometricAction slotValue
  rw [ratio]
  simpa only [List.map_ofFn, Function.comp_apply] using
    geometric_sum_scale run.initial final run.slot
      (List.ofFn fun index : Fin 41 => index)

private theorem runs_realized
    (steps : List Wire.Step) (source : SourceAssignment)
    (final : FinalAssignment)
    (linked : SourceSlotsLinked steps source final) :
    ∀ {terms : List SourceTerm} {runs : List GeometricRun},
      List.Forall₂ (RunLayoutMatches steps) terms runs →
      List.Forall₂
        (fun term run =>
          geometricAction run final =
            term.coefficient * sourceValue source term.column)
        terms runs := by
  intro terms runs layouts
  induction layouts with
  | nil => exact .nil
  | cons head tail inductionHypothesis =>
      apply List.Forall₂.cons
      · rw [geometricAction_eq_scaled_slotValue _ final head.ratio,
          head.initial, linked _ _ head.slot]
      · exact inductionHypothesis

theorem port_realized_of_layout
    (steps : List Wire.Step) (source : SourceAssignment)
    (final : FinalAssignment)
    (one : final.explicit .one = 1)
    {value : SourceLinearCombination} {port : Port}
    (layout : PortLayoutMatches steps value port)
    (linked : SourceSlotsLinked steps source final) :
    PortRealized value port source final := by
  refine { constant := ?_, terms := runs_realized steps source final linked layout.runs }
  rw [layout.explicit]
  unfold expectedExplicit
  split <;> rename_i constantZero
  · simp [constantZero, sum]
  · simp [one, sum, Fin.mul_one]

def stepPortLayoutCheck
    (step : Wire.Step) (row : Wire.Row) : Bool :=
  decide (StepPortLayoutsMatch decodedSteps step row)

def allStepPortLayoutsCheck : List Wire.Step → List Wire.Row → Bool
  | [], [] => true
  | step :: stepTail, row :: rowTail =>
      stepPortLayoutCheck step row &&
        allStepPortLayoutsCheck stepTail rowTail
  | _, _ => false

theorem stepPortLayoutCheck_sound
    (step : Wire.Step) (row : Wire.Row)
    (checked : stepPortLayoutCheck step row = true) :
    StepPortLayoutsMatch decodedSteps step row := by
  unfold stepPortLayoutCheck at checked
  exact of_decide_eq_true checked

private theorem allStepPortLayoutsCheck_sound :
    ∀ (steps : List Wire.Step) (rows : List Wire.Row),
      allStepPortLayoutsCheck steps rows = true →
      List.Forall₂ (StepPortLayoutsMatch decodedSteps) steps rows
  | [], [], _ => .nil
  | [], _ :: _, checked => by simp [allStepPortLayoutsCheck] at checked
  | _ :: _, [], checked => by simp [allStepPortLayoutsCheck] at checked
  | step :: stepTail, row :: rowTail, checked => by
      simp only [allStepPortLayoutsCheck, Bool.and_eq_true] at checked
      exact .cons (stepPortLayoutCheck_sound step row checked.1)
        (allStepPortLayoutsCheck_sound stepTail rowTail checked.2)

theorem decoded_head_port_layouts_checked :
    allStepPortLayoutsCheck decodedStepHead decodedRowHead = true := by
  rfl

theorem decoded_tail_port_layouts_checked :
    allStepPortLayoutsCheck decodedStepTail decodedRowTail = true := by
  rfl

private theorem forall₂_append
    {α β : Type} {relation : α → β → Prop}
    {leftSteps rightSteps : List α} {leftRows rightRows : List β}
    (left : List.Forall₂ relation leftSteps leftRows)
    (right : List.Forall₂ relation rightSteps rightRows) :
    List.Forall₂ relation (leftSteps ++ rightSteps) (leftRows ++ rightRows) := by
  induction left with
  | nil => exact right
  | cons head tail inductionHypothesis =>
      exact .cons head inductionHypothesis

theorem decoded_step_port_layouts :
    List.Forall₂ (StepPortLayoutsMatch decodedSteps)
      decodedSteps decodedRows := by
  unfold decodedSteps decodedRows
  exact forall₂_append
    (allStepPortLayoutsCheck_sound decodedStepHead decodedRowHead
      decoded_head_port_layouts_checked)
    (allStepPortLayoutsCheck_sound decodedStepTail decodedRowTail
      decoded_tail_port_layouts_checked)

private theorem step_ports_realized_of_layouts
    (source : SourceAssignment) (final : FinalAssignment)
    (one : final.explicit .one = 1)
    (linked : SourceSlotsLinked decodedSteps source final) :
    ∀ {steps : List Wire.Step} {rows : List Wire.Row},
      List.Forall₂ (StepPortLayoutsMatch decodedSteps) steps rows →
      List.Forall₂ (StepPortsRealized source final) steps rows := by
  intro steps rows layouts
  induction layouts with
  | nil => exact .nil
  | cons head tail inductionHypothesis =>
      exact .cons
        { input := port_realized_of_layout decodedSteps source final one
            head.input linked
          output := port_realized_of_layout decodedSteps source final one
            head.output linked }
        inductionHypothesis

theorem decoded_step_ports_realized
    (source : SourceAssignment) (final : FinalAssignment)
    (one : final.explicit .one = 1)
    (linked : SourceSlotsLinked decodedSteps source final) :
    List.Forall₂ (StepPortsRealized source final)
      decodedSteps decodedRows :=
  step_ports_realized_of_layouts source final one linked
    decoded_step_port_layouts

/-- Canonical source witness reconstructed only from the final radix-3 slots.
Missing local owners fail closed to zero; the exact generated layout proves
that every source column used by an S-box step has an owner. -/
def reconstructedSource (final : FinalAssignment) : SourceAssignment where
  externalA := fun lane => slotValue final (.externalA lane)
  externalB := fun lane => slotValue final (.externalB lane)
  localValue := fun offset =>
    match sourceSlot decodedSteps (.local offset) with
    | some slot => slotValue final slot
    | none => 0

theorem reconstructedSource_slots_linked (final : FinalAssignment) :
    SourceSlotsLinked decodedSteps (reconstructedSource final) final := by
  intro column slot linked
  cases column with
  | externalA lane =>
      simp only [sourceSlot, Option.some.injEq] at linked
      subst slot
      rfl
  | externalB lane =>
      simp only [sourceSlot, Option.some.injEq] at linked
      subst slot
      rfl
  | «local» offset =>
      change slotValue final slot =
        match sourceSlot decodedSteps (.local offset) with
        | some owner => slotValue final owner
        | none => 0
      rw [linked]

/-- All generated rows plus constant-one and exact source-slot value links
imply all 86 typed source S-box equations on the same assignment. -/
theorem decoded_rows_imply_step_sboxes_from_slot_links
    (source : SourceAssignment) (final : FinalAssignment)
    (one : final.explicit .one = 1)
    (selectorOne : final.explicit .selector = 1)
    (linked : SourceSlotsLinked decodedSteps source final)
    (holds : ∀ row ∈ decodedRows, residual row final = 0) :
    ∀ step ∈ decodedSteps, StepSboxHolds source step :=
  decoded_rows_and_port_links_imply_step_sboxes source final selectorOne
    (decoded_step_ports_realized source final one linked) holds

/-- All generated rows imply all 86 typed S-box equations for the canonical
source witness reconstructed from the same final assignment. -/
theorem decoded_rows_imply_reconstructed_step_sboxes
    (final : FinalAssignment)
    (one : final.explicit .one = 1)
    (selectorOne : final.explicit .selector = 1)
    (holds : ∀ row ∈ decodedRows, residual row final = 0) :
    ∀ step ∈ decodedSteps, StepSboxHolds (reconstructedSource final) step :=
  decoded_rows_imply_step_sboxes_from_slot_links
    (reconstructedSource final) final one selectorOne
    (reconstructedSource_slots_linked final) holds

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafReconstruction
