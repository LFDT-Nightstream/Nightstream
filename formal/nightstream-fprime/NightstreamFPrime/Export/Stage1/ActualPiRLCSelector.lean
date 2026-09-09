import NightstreamFPrime.Export.Stage1.PiRLCRetainedInputs
import NightstreamFPrime.Gadgets.Sampling.First54.Semantics

/-!
Decodes one First54 trace from its actual retained forms. The existing child
specification owns selection semantics. No canonical source encoding is an
input to this decoder or its row-soundness boundary.
-/

namespace NightstreamFPrime.Export.Stage1.ActualPiRLCSelector

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open PiRLCFirst54DirectPlan

variable {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}

def decodedEnv (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (source : Fin PiRLCFirst54DirectSchedule.sourceCount) : Env := fun column =>
  if round : column / First54.roundPrivateCount < First54.candidateCount then
    if slot : column % First54.roundPrivateCount < First54Step.slotCount then
      (inputs.position ⟨⟨source, ⟨_, round⟩⟩, ⟨_, slot⟩⟩).eval assignment
    else
      (inputs.value ⟨⟨source, ⟨_, round⟩⟩,
        ⟨column % First54.roundPrivateCount - First54Step.slotCount, by
          have bounded := Nat.mod_lt column (by decide : 0 < First54.roundPrivateCount)
          change column % First54.roundPrivateCount < 109 at bounded
          change ¬ column % First54.roundPrivateCount < 55 at slot
          change column % First54.roundPrivateCount - 55 < 54
          omega⟩⟩).eval assignment
  else 0

def interface (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (source : Fin PiRLCFirst54DirectSchedule.sourceCount) : First54.Interface where
  accepted := fun _ round => .const (1 - (inputs.reject ⟨source, round⟩).eval assignment)
  symbol := fun _ round => .const ((inputs.symbol ⟨source, round⟩).eval assignment)

theorem position_eval (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (source : Fin PiRLCFirst54DirectSchedule.sourceCount)
    (round : Fin First54.candidateCount) (slot : Fin First54Step.slotCount) :
    (First54Step.output (First54.positionOffset 0 round.val) slot).eval
        (decodedEnv inputs assignment source) =
      (inputs.position ⟨⟨source, round⟩, slot⟩).eval assignment := by
  have slotLt : slot.val < 55 := slot.isLt
  have division : (round.val * 109 + slot.val) / 109 = round.val := by omega
  have remainder : (round.val * 109 + slot.val) % 109 = slot.val := by omega
  change decodedEnv inputs assignment source (0 + round.val * 109 + slot.val) = _
  rw [Nat.zero_add]
  simp only [decodedEnv, show First54.roundPrivateCount = 109 by rfl, division, remainder,
    dif_pos round.isLt, dif_pos slot.isLt]
  rfl

theorem value_eval (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (source : Fin PiRLCFirst54DirectSchedule.sourceCount)
    (round : Fin First54.candidateCount) (slot : Fin First54ValueStep.outputCount) :
    (First54ValueStep.output (First54.valueOffset 0 round.val) slot).eval
        (decodedEnv inputs assignment source) =
      (inputs.value ⟨⟨source, round⟩, slot⟩).eval assignment := by
  have slotLt : slot.val < 54 := slot.isLt
  have division : (round.val * 109 + 55 + slot.val) / 109 = round.val := by omega
  have remainder : (round.val * 109 + 55 + slot.val) % 109 = 55 + slot.val := by omega
  change decodedEnv inputs assignment source (0 + round.val * 109 + 55 + slot.val) = _
  rw [Nat.zero_add]
  simp only [decodedEnv, show First54.roundPrivateCount = 109 by rfl, division, remainder,
    dif_pos round.isLt, First54Step.slotCount, Nat.add_sub_cancel_left]
  rw [dif_neg (by omega)]
  rfl

private theorem accepted_eval (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth) (one : assignment inputs.oneColumn = 1)
    (candidate : PiRLCFirst54DirectSchedule.Candidate) :
    (acceptedForm inputs candidate).eval assignment =
      1 - (inputs.reject candidate).eval assignment := by
  rw [acceptedForm, subtract_eval]
  simp [oneForm, rejectForm, one]

theorem prior_position_eval (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth) (one : assignment inputs.oneColumn = 1)
    (source : Fin PiRLCFirst54DirectSchedule.sourceCount)
    (round : Fin First54.candidateCount) (slot : Fin First54Step.slotCount) :
    (First54.priorPosition 0 round.val slot).eval (decodedEnv inputs assignment source) =
      (priorPositionForm inputs ⟨source, round⟩ slot).eval assignment := by
  cases current : round.val with
  | zero =>
      by_cases first : slot.val = 0
      all_goals simp [First54.priorPosition, current, First54.initialPosition,
        priorPositionForm, initialPositionForm, oneForm, one, first, Expr.eval]
  | succ previous =>
      have previousLt : previous < First54.candidateCount := by
        have bound := round.isLt
        omega
      simp only [First54.priorPosition]
      rw [position_eval inputs assignment source ⟨previous, previousLt⟩ slot]
      simp [priorPositionForm, current, previousCandidate]
      rfl

theorem prior_value_eval (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (source : Fin PiRLCFirst54DirectSchedule.sourceCount)
    (round : Fin First54.candidateCount) (slot : Fin First54ValueStep.outputCount) :
    (First54.priorOutput 0 round.val slot).eval (decodedEnv inputs assignment source) =
      (priorValueForm inputs ⟨⟨source, round⟩, slot⟩).eval assignment := by
  cases current : round.val with
  | zero => simp [First54.priorOutput, current, priorValueForm, Expr.eval]
  | succ previous =>
      have previousLt : previous < First54.candidateCount := by
        have bound := round.isLt
        omega
      simp only [First54.priorOutput]
      rw [value_eval inputs assignment source ⟨previous, previousLt⟩ slot]
      simp [priorValueForm, current, previousCandidate]
      rfl

private def priorPrevious (prior : Fin First54Step.slotCount → F)
    (slot : Fin First54Step.slotCount) : F :=
  if first : slot.val = 0 then 0 else prior (First54Step.previousSlot slot (by omega))

private theorem position_update_of_equation
    (accepted output : F) (prior : Fin First54Step.slotCount → F)
    (slot : Fin First54Step.slotCount)
    (equation : accepted * (if slot.val = First54Step.fullSlot then priorPrevious prior slot
      else priorPrevious prior slot - prior slot) = output - prior slot) :
    output = First54Step.update accepted prior slot := by
  have moved := congrArg (fun value : F => value + prior slot) equation
  have outputEquation : output = prior slot + accepted *
      (if slot.val = First54Step.fullSlot then priorPrevious prior slot
        else priorPrevious prior slot - prior slot) := by
    simpa [sub_eq_add_neg, add_assoc, add_comm, add_left_comm] using moved.symm
  by_cases first : slot.val = 0
  · simp [priorPrevious, First54Step.update, first, First54Step.fullSlot] at outputEquation ⊢
    simpa [sub_eq_add_neg, mul_add, add_mul, mul_neg, neg_mul, mul_comm,
      mul_left_comm, mul_assoc] using outputEquation
  · by_cases full : slot.val = First54Step.fullSlot
    · simpa [priorPrevious, First54Step.update, first, full, mul_comm] using outputEquation
    · simp [priorPrevious, First54Step.update, first, full] at outputEquation ⊢
      simpa [sub_eq_add_neg, mul_add, add_mul, mul_neg, neg_mul, mul_comm,
        mul_left_comm, mul_assoc, add_assoc, add_comm, add_left_comm] using outputEquation

private theorem prior_previous_eval (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (candidate : PiRLCFirst54DirectSchedule.Candidate) (slot : Fin First54Step.slotCount) :
    (previousPositionForm inputs ⟨candidate, slot⟩).eval assignment =
      priorPrevious (fun current => (priorPositionForm inputs candidate current).eval assignment) slot := by
  unfold previousPositionForm priorPrevious
  split <;> simp

/-- Accepted position rows impose the exact position transition on actual values. -/
theorem position_rows_imply_update (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth) (one : assignment inputs.oneColumn = 1)
    (equations : PositionEquations inputs assignment)
    (source : Fin PiRLCFirst54DirectSchedule.sourceCount)
    (round : Fin First54.candidateCount) (slot : Fin First54Step.slotCount) :
    (inputs.position ⟨⟨source, round⟩, slot⟩).eval assignment =
      First54Step.update (1 - (inputs.reject ⟨source, round⟩).eval assignment)
        (fun current => (priorPositionForm inputs ⟨source, round⟩ current).eval assignment) slot := by
  have equation := equations
    (PiRLCFirst54DirectSchedule.positionIndex ⟨⟨source, round⟩, slot⟩)
  simp only [positionInterface, PiRLCFirst54DirectSchedule.position_positionIndex] at equation
  rw [accepted_eval inputs assignment one] at equation
  simp only [positionDifferenceForm, positionOutputForm, subtract_eval, positionDeltaForm] at equation
  split at equation
  · rename_i full
    rw [prior_previous_eval] at equation
    apply position_update_of_equation
    simpa only [if_pos full] using equation
  · rename_i notFull
    rw [subtract_eval, prior_previous_eval] at equation
    apply position_update_of_equation
    simpa only [if_neg notFull] using equation

/-- Accepted value and product rows impose the exact selected-value transition. -/
theorem value_rows_imply_update (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth) (one : assignment inputs.oneColumn = 1)
    (products : AcceptedProductEquations inputs assignment)
    (values : ValueEquations inputs assignment)
    (source : Fin PiRLCFirst54DirectSchedule.sourceCount)
    (round : Fin First54.candidateCount) (slot : Fin First54ValueStep.outputCount) :
    (inputs.value ⟨⟨source, round⟩, slot⟩).eval assignment =
      First54ValueStep.update (1 - (inputs.reject ⟨source, round⟩).eval assignment)
        ((inputs.symbol ⟨source, round⟩).eval assignment)
        (fun current => (priorPositionForm inputs ⟨source, round⟩ current).eval assignment)
        (fun current => (priorValueForm inputs ⟨⟨source, round⟩, current⟩).eval assignment) slot := by
  have product := products (PiRLCFirst54DirectSchedule.candidateIndex ⟨source, round⟩)
  simp only [acceptedProductInterface,
    PiRLCFirst54DirectSchedule.candidate_candidateIndex, symbolForm, productForm] at product
  rw [accepted_eval inputs assignment one] at product
  have equation := values (PiRLCFirst54DirectSchedule.valueIndex ⟨⟨source, round⟩, slot⟩)
  simp only [valueInterface, PiRLCFirst54DirectSchedule.value_valueIndex,
    productForm, valueDifferenceForm, valueOutputForm, subtract_eval] at equation
  rw [← product] at equation
  have moved := congrArg
    (fun value : F => value + (priorValueForm inputs ⟨⟨source, round⟩, slot⟩).eval assignment)
    equation
  simpa [First54ValueStep.update, sub_eq_add_neg, add_assoc, add_comm, add_left_comm,
    mul_assoc] using moved.symm

private theorem round_index_eq (round : Fin First54.candidateCount) :
    First54.candidateIndex round.val = round := by
  apply Fin.ext
  exact Nat.mod_eq_of_lt round.isLt

theorem final_eval (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (source : Fin PiRLCFirst54DirectSchedule.sourceCount) :
    (First54.finalFull 0).eval (decodedEnv inputs assignment source) =
      (finalForm inputs source).eval assignment := by
  exact position_eval inputs assignment source ⟨63, by decide⟩ First54.fullSlot

/-- Arbitrary accepted First54 rows satisfy the existing selector child in
values decoded from that assignment. The one cell is the only value premise. -/
theorem rowsZero_implies_specHolds (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth) (one : assignment inputs.oneColumn = 1)
    (rows : (plan inputs).RowsZero assignment)
    (source : Fin PiRLCFirst54DirectSchedule.sourceCount) :
    First54.SpecHolds (interface inputs assignment source) 0
      (decodedEnv inputs assignment source) := by
  have equations := (planRowsZero_iff inputs assignment one).mp rows
  refine ⟨?_, ?_, ?_⟩
  · intro round slot
    change (First54Step.output (First54.positionOffset 0 round.val) slot).eval
        (decodedEnv inputs assignment source) =
      First54Step.update
        (1 - (inputs.reject ⟨source, First54.candidateIndex round.val⟩).eval assignment)
        (fun current => (First54.priorPosition 0 round.val current).eval
          (decodedEnv inputs assignment source)) slot
    rw [round_index_eq, position_eval]
    have prior := funext (prior_position_eval inputs assignment one source round)
    rw [prior]
    exact position_rows_imply_update inputs assignment one equations.1 source round slot
  · intro round slot
    change (First54ValueStep.output (First54.valueOffset 0 round.val) slot).eval
        (decodedEnv inputs assignment source) =
      First54ValueStep.update
        (1 - (inputs.reject ⟨source, First54.candidateIndex round.val⟩).eval assignment)
        ((inputs.symbol ⟨source, First54.candidateIndex round.val⟩).eval assignment)
        (fun current => (First54.priorPosition 0 round.val current).eval
          (decodedEnv inputs assignment source))
        (fun current => (First54.priorOutput 0 round.val current).eval
          (decodedEnv inputs assignment source)) slot
    rw [round_index_eq, value_eval]
    have priorPosition := funext (prior_position_eval inputs assignment one source round)
    have priorOutput := funext (prior_value_eval inputs assignment source round)
    rw [priorPosition, priorOutput]
    exact value_rows_imply_update inputs assignment one equations.2.1 equations.2.2.1
      source round slot
  · rw [final_eval]
    have equation := equations.2.2.2 source
    simp only [finalInterface, subtract_eval] at equation
    have oneEval : (oneForm inputs).eval assignment = 1 := by simp [oneForm, one]
    rw [oneEval] at equation
    exact Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp equation

def outputValues (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (source : Fin PiRLCFirst54DirectSchedule.sourceCount) : List F :=
  List.ofFn fun slot : Fin First54.outputCount =>
    (inputs.value ⟨⟨source, ⟨63, by decide⟩⟩, slot⟩).eval assignment

theorem outputValues_eq (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (source : Fin PiRLCFirst54DirectSchedule.sourceCount) :
    First54.evalOutput (decodedEnv inputs assignment source) 0 =
      outputValues inputs assignment source := by
  apply congrArg List.ofFn
  funext slot
  exact value_eval inputs assignment source ⟨63, by decide⟩ slot

/-- The decoder supplies the Boolean reject flags. Given those flags, the
actual selector rows enforce bounded first-54 selection with no default output. -/
theorem rowsZero_and_rejects_imply_boundedSample (inputs : Inputs program logicalWidth)
    (assignment : Assignment F logicalWidth) (one : assignment inputs.oneColumn = 1)
    (rows : (plan inputs).RowsZero assignment)
    (source : Fin PiRLCFirst54DirectSchedule.sourceCount)
    (rejects : ∀ round : Fin First54.candidateCount,
      (inputs.reject ⟨source, round⟩).eval assignment = 0 ∨
        (inputs.reject ⟨source, round⟩).eval assignment = 1) :
    Sampling.FirstAccepted.boundedSample
        (First54.semanticVerifier (interface inputs assignment source) 0
          (decodedEnv inputs assignment source))
        First54.outputCount (First54.semanticCandidates First54.candidateCount) =
      some (outputValues inputs assignment source) := by
  have assumptions : First54.Assumptions (interface inputs assignment source) 0
      (decodedEnv inputs assignment source) := by
    refine ⟨?_, ?_, ?_⟩
    · intro round
      trivial
    · intro round
      trivial
    · intro round
      rcases rejects round with rejected | rejected
      · right
        simp [interface, rejected]
      · left
        simp [interface, rejected]
  rw [← outputValues_eq]
  exact First54.parentCoverage (interface inputs assignment source) 0
    (decodedEnv inputs assignment source) assumptions
    (rowsZero_implies_specHolds inputs assignment one rows source)

end NightstreamFPrime.Export.Stage1.ActualPiRLCSelector
