import NightstreamFPrime.Lifecycle.PiRLC.Wide.DigitWords

/-! Wide sampler with temporary digit words for the existing ring recipes.
The compact CCS assignment still retains only the checked bits and fields. -/

namespace NightstreamFPrime.Lifecycle.PiRLC.Wide.ProjectedBatch

open NightstreamFPrime.Spec NightstreamFPrime.Circuit

abbrev Interface := Batch.Interface
abbrev EState := Batch.EState
abbrev sourceCount := Batch.sourceCount
abbrev Assumptions := Batch.Assumptions

def wordsOffset (offset : Nat) : Nat := offset + Batch.privateCount
def privateCount : Nat := Batch.privateCount + DigitWords.count
def logicalRowCount : Nat := Batch.logicalRowCount + DigitWords.count

def sampleOp (interface : Interface) (offset : Nat) : Op :=
  Sequence.childOp "pirlc.wide.batch" (Batch.circuit interface) offset

def wordsOp (offset : Nat) : Op :=
  Sequence.childOp "pirlc.wide.digit_words" (DigitWords.circuit offset) (wordsOffset offset)

def operations (interface : Interface) (offset : Nat) : List Op := [sampleOp interface offset, wordsOp offset]

def outputState (interface : Interface) (offset : Nat) : EState := Batch.outputState interface offset

def outputChallenge (offset : Nat) (source : Fin sourceCount) (position : Fin ringDegree) : Expr :=
  DigitWords.outputWord (wordsOffset offset) source position - 2

structure SpecHolds (interface : Interface) (offset : Nat) (env : Env) extends Batch.SpecHolds interface offset env where
  words : DigitWords.SpecHolds offset (wordsOffset offset) env

private theorem child_constraints (name : String) (child : FormalCircuit) (offset : Nat) :
    (Sequence.childOp name child offset).flatConstraints = flatConstraints (Circuit.ops child.main offset) := rfl

theorem sourceCount_eq : sourceCount = 17 := Batch.sourceCount_eq

theorem counts : privateCount = 55403 ∧ logicalRowCount = 32623 := by
  rw [privateCount, logicalRowCount, Batch.counts.1, Batch.counts.2, DigitWords.count_eq]
  decide

theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (operations interface offset) = privateCount := by
  simp only [operations, localLength, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
    sampleOp, wordsOp, Sequence.childOp_localLength]
  change localLength (Circuit.ops (Batch.circuit interface).main offset) +
    (localLength (Circuit.ops (DigitWords.circuit offset).main (wordsOffset offset)) + 0) = _
  rw [Batch.circuit_ops, DigitWords.circuit_ops, Batch.localLength_eq, DigitWords.localLength_eq]
  change Batch.privateCount + (DigitWords.count + 0) = _
  unfold privateCount
  omega

theorem rowCount_eq (interface : Interface) (offset : Nat) :
    (flatConstraints (operations interface offset)).length = logicalRowCount := by
  rw [flatConstraints_length_eq_rowCount]
  rfl

theorem soundness (interface : Interface) (offset : Nat) (env : Env)
    (inputs : Assumptions interface offset) (rows : holds env (operations interface offset)) :
    SpecHolds interface offset env := by
  have sample := rows (sampleOp interface offset) (by simp [operations])
  have words := rows (wordsOp offset) (by simp [operations])
  exact ⟨sample inputs, words (Nat.le_refl _)⟩

theorem outputChallenge_below (offset : Nat) (source : Fin sourceCount) (position : Fin ringDegree) :
    (outputChallenge offset source position).VarsBelow (offset + privateCount) := by
  have bounded := (Fin.encodeProd (source, position)).isLt
  change (wordsOffset offset + (Fin.encodeProd (source, position)).val < offset + privateCount) ∧ True ∧ True
  unfold wordsOffset privateCount
  exact ⟨by change (Fin.encodeProd (source, position)).val < DigitWords.count at bounded; omega, trivial, trivial⟩

theorem outputChallenge_eval (interface : Interface) (env : Env) (offset : Nat)
    (spec : SpecHolds interface offset env) (source : Fin sourceCount) :
    (fun position => (outputChallenge offset source position).eval env) =
      Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Transcript.challengeAt
        (Scalar.evalState env (interface.initialState offset)) source.val := by
  calc
    (fun position => (outputChallenge offset source position).eval env) =
        (fun position => (Batch.outputChallenge offset source position).eval env) := by
      funext position
      have words := DigitWords.outputWord_eq offset (wordsOffset offset) env spec.words source position
      change (DigitWords.outputWord (wordsOffset offset) source position - 2).eval env =
        (Gadgets.Sampling.WideReduction.Program.outputWord (Scalar.rangeOffset (Batch.sourceOffset offset source.val)) position - 2).eval env
      simp only [Expr.eval_sub]
      exact congrArg (fun value : F => value - 2) words
    _ = _ := Batch.outputChallenge_eval interface env offset spec.toSpecHolds source

theorem outputChallenge_member (interface : Interface) (env : Env) (offset : Nat)
    (spec : SpecHolds interface offset env) (source : Fin sourceCount) :
    Phi81StrongSet.ProductionMember (fun position => (outputChallenge offset source position).eval env) := by
  rw [outputChallenge_eval interface env offset spec source]
  exact Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Transcript.challengeAt_member _ _

theorem scope (interface : Interface) (offset : Nat) (inputs : Assumptions interface offset) :
    ∀ expression ∈ flatConstraints (operations interface offset), expression.VarsBelow (offset + privateCount) := by
  intro expression member
  obtain ⟨operation, member, inside⟩ := List.mem_flatMap.mp member
  simp only [operations, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · apply Expr.VarsBelow.mono _ (Batch.scope interface offset inputs expression inside)
    unfold privateCount
    omega
  · rw [wordsOp, child_constraints, DigitWords.circuit_ops] at inside
    have bounded := DigitWords.scope offset (wordsOffset offset) (Nat.le_refl _) expression inside
    apply Expr.VarsBelow.mono _ bounded
    unfold wordsOffset privateCount
    omega

theorem complete (interface : Interface) (env : Env) (offset : Nat) (inputs : Assumptions interface offset) :
    ∃ completed, AgreesOutside env completed offset (localLength (operations interface offset)) ∧
      holdsFlat completed (operations interface offset) := by
  obtain ⟨sampled, sampleAgreement, sampleRows⟩ := Batch.complete interface env offset inputs
  obtain ⟨before, beforeOps, _, _, _⟩ := Sequence.appendBuiltAt (Sequence.empty env offset)
    "pirlc.wide.batch" (Batch.circuit interface) offset (by rfl)
    (by rw [Batch.circuit_ops, Batch.localLength_eq]; exact Batch.scope interface offset inputs)
    sampled (by rw [Batch.circuit_ops]; exact sampleAgreement) (by rw [Batch.circuit_ops]; exact sampleRows)
  have beforeLength : localLength before.operations = Batch.privateCount := by
    rw [beforeOps]
    simp only [Sequence.empty, List.nil_append, localLength, List.map_cons, List.map_nil,
      List.sum_cons, List.sum_nil, Sequence.childOp_localLength, Nat.add_zero]
    change localLength (Circuit.ops (Batch.circuit interface).main offset) = _
    rw [Batch.circuit_ops, Batch.localLength_eq]
  obtain ⟨built, wordAgreement, wordRows⟩ := DigitWords.complete offset (wordsOffset offset) before.current (Nat.le_refl _)
  obtain ⟨done, doneOps, _, _, _⟩ := Sequence.appendBuiltAt before "pirlc.wide.digit_words"
    (DigitWords.circuit offset) (wordsOffset offset) (by rw [beforeLength]; rfl)
    (by rw [DigitWords.circuit_ops, DigitWords.localLength_eq]; exact DigitWords.scope offset (wordsOffset offset) (Nat.le_refl _))
    built (by rw [DigitWords.circuit_ops, DigitWords.localLength_eq]; exact wordAgreement)
    (by rw [DigitWords.circuit_ops]; exact wordRows)
  have same : done.operations = operations interface offset := by
    rw [doneOps, beforeOps]
    rfl
  exact ⟨done.current, by simpa only [same] using done.agrees, by simpa only [same] using done.rows⟩

def circuit (interface : Interface) : FormalCircuit where
  main := fun offset => ((), offset + privateCount, operations interface offset)
  assumptions := fun offset _ => Assumptions interface offset
  spec := SpecHolds interface
  privateCount := fun _ => privateCount
  rowCount := fun _ => logicalRowCount
  privateCount_eq := localLength_eq interface
  rowCount_eq := rowCount_eq interface
  soundness := fun env offset => soundness interface offset env
  completeness := fun env offset inputs _ => complete interface env offset inputs

theorem circuit_ops (interface : Interface) (offset : Nat) :
    Circuit.ops (circuit interface).main offset = operations interface offset := rfl

end NightstreamFPrime.Lifecycle.PiRLC.Wide.ProjectedBatch
