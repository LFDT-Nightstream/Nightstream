import NightstreamFPrime.Spec.Folding.PiDEC.OutputWitnessConsumer
import NightstreamFPrime.Spec.Folding.PiRLC.CoordinateForkLaw
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork

/-!
SuperNeo B.1/B.3/B.4: one resumed PiRLC/PiDEC suffix returns its own final
child messages and witnesses. The parent is computed from the fixed PiCCS
batch and that invocation's challenge vector. The checked output witness
opens this parent by the existing PiDEC reduction.

The checker and radix recomposition return their values and work together.
Abort, rejected output, and successful return are all charged. No complete
fork or intermediate parent witness is supplied to this program.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Nifs.PaperWeakSuffix

open NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Spec.Folding
open PiRLC.PaperForkExtraction PiRLC.CoordinateForkLaw
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)

/-- Raw public child messages and the final witness returned by this call. -/
structure Reply (Assignment Evaluation Commitment : Type*) (params : GlobalParams) where
  messages : Fin params.k → PiDEC.PaperVerifier.ChildMessage Evaluation Commitment
  assignments : Fin params.k → Assignment

variable {Structure Assignment PublicInput Point Evaluation Commitment Scalar : Type*}
  {semantics : RelationSemantics Structure Assignment PublicInput Point Evaluation Commitment}
  {params : GlobalParams} {arity : BatchArity params}
  (rlc : PiRLC.Algebra Structure Assignment PublicInput Point Evaluation Commitment Scalar semantics params)
  (batch : InputBatch Structure PublicInput Point Evaluation Commitment params arity)

/-- The verifier computes the public parent; it is absent from the reply. -/
def attempt (vector : Fin arity.total → Challenge rlc)
    (reply : Reply Assignment Evaluation Commitment params) :
    PiDEC.PaperVerifier.Attempt Structure PublicInput Point Evaluation Commitment params where
  parent := PiRLC.combinedOutput rlc batch.system batch.point batch.inputs (scalarVector rlc vector)
  messages := reply.messages

/-- Executed suffix checks and radix recomposition, with their actual clocks. -/
structure Program where
  check : (Fin arity.total → Challenge rlc) → Reply Assignment Evaluation Commitment params → Result Bool
  recompose : (Fin params.k → Assignment) → Result Assignment

variable (dec : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation Commitment semantics params)
  (publicSplit : PiDEC.PaperVerifier.PublicInputSplit dec)
  (evaluationArity : PiDEC.PaperVerifier.EvaluationArity semantics)

/-- Only the exact accepted final-output relation enables recomposition. The
primitive equality refers to this same returned assignment vector. -/
structure Correct (program : Program (arity := arity) rlc) : Prop where
  check : ∀ vector reply, (program.check vector reply).value = true ↔
    PiDEC.PaperVerifier.Accepted dec publicSplit evaluationArity (attempt rlc batch vector reply) ∧
      ∀ child, CE.Holds semantics params
        (PiDEC.PaperVerifier.children publicSplit (attempt rlc batch vector reply) child)
        (reply.assignments child)
  recompose : ∀ assignments, (program.recompose assignments).value = dec.recomposeAssignment assignments

/-- Execute the check once and return the computed parent only on success. -/
def finish (program : Program (arity := arity) rlc) (vector : Fin arity.total → Challenge rlc) :
    Option (Reply Assignment Evaluation Commitment params) → Result (Option Assignment)
  | none => ⟨none, 1⟩
  | some reply =>
      let checked := program.check vector reply
      if checked.value then
        let parent := program.recompose reply.assignments
        ⟨some parent.value, checked.work + parent.work + 2⟩
      else ⟨none, checked.work + 2⟩

/-- An abort has no returned witness on which to run the output checker. -/
def checkerWork (program : Program (arity := arity) rlc) (vector : Fin arity.total → Challenge rlc) :
    Option (Reply Assignment Evaluation Commitment params) → Nat
  | none => 0
  | some reply => (program.check vector reply).work

theorem finish_return_iff (program : Program (arity := arity) rlc) (vector : Fin arity.total → Challenge rlc)
    (outcome : Option (Reply Assignment Evaluation Commitment params)) (assignment : Assignment) :
    (finish rlc program vector outcome).value = some assignment ↔
      ∃ reply, outcome = some reply ∧ (program.check vector reply).value = true ∧
        (program.recompose reply.assignments).value = assignment := by
  cases outcome with
  | none => simp [finish]
  | some reply =>
      cases checked : (program.check vector reply).value <;> simp [finish, checked]

/-- This call's valid final-output witness yields this call's exact PiRLC
response. Rewound calls must run the same program on their own replies. -/
theorem finish_returns_parent (program : Program (arity := arity) rlc)
    (correct : Correct rlc batch dec publicSplit evaluationArity program)
    (kPositive : 0 < params.k) (vector : Fin arity.total → Challenge rlc)
    (outcome : Option (Reply Assignment Evaluation Commitment params)) (assignment : Assignment)
    (returned : (finish rlc program vector outcome).value = some assignment) :
    (response rlc vector assignment).Success semantics params rlc batch := by
  rcases (finish_return_iff rlc program vector outcome assignment).mp returned with
    ⟨reply, issued, checked, recomposed⟩
  have successful := (correct.check vector reply).mp checked
  have parent := PiDEC.PaperVerifier.reduce_knowledge semantics params dec publicSplit evaluationArity
    (attempt rlc batch vector reply) reply.assignments kPositive successful.1 successful.2
  change CE.Holds semantics params (attempt rlc batch vector reply).parent assignment
  rw [← recomposed, correct.recompose]
  exact parent

/-- The observed checked-call clock plus a proved recomposition bound covers
every branch. The check clock remains part of the actual suffix-call mean. -/
theorem finish_work_le (program : Program (arity := arity) rlc) (recomposeBound : Nat)
    (bounded : ∀ assignments, (program.recompose assignments).work ≤ recomposeBound)
    (vector : Fin arity.total → Challenge rlc)
    (outcome : Option (Reply Assignment Evaluation Commitment params)) :
    (finish rlc program vector outcome).work ≤ checkerWork rlc program vector outcome + recomposeBound + 2 := by
  cases outcome with
  | none => simp only [finish, checkerWork, Nat.zero_add]; omega
  | some reply =>
      have work := bounded reply.assignments
      cases checked : (program.check vector reply).value <;>
        simp only [finish, checkerWork, checked, Bool.false_eq_true, ↓reduceIte] <;> omega

end NightstreamFPrime.Spec.Folding.Nifs.PaperWeakSuffix
