import NightstreamFPrime.Spec.Folding.PiRLC.CoordinateOracleStar
import NightstreamFPrime.Spec.Folding.PiRLC.CoordinateForkLaw

/-!
The successful stopped-oracle output event is a typed `CompleteFork` with the
same base vector, base assignment, changed challenges, and returned assignments.
Its mass is the joint trace law, so the interactive loss bound applies directly
to the existing inverse-difference extractor's input.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiRLC.CoordinateForkEvent

open scoped BigOperators
open NightstreamFPrime.Spec
open PaperForkExtraction CoordinateForkLaw CoordinateOracle CoordinateOracleStar

variable {Structure Assignment PublicInput Point Evaluation Commitment Scalar : Type*}
  {semantics : RelationSemantics
    Structure Assignment PublicInput Point Evaluation Commitment}
  {params : GlobalParams} {arity : BatchArity params}
  (algebra : Algebra Structure Assignment PublicInput Point Evaluation Commitment
    Scalar semantics params)
  (batch : InputBatch Structure PublicInput Point Evaluation Commitment params arity)

/-- The fork preserves every response value selected by the oracle experiment. -/
def Event (vector : Fin arity.total → Challenge algebra) (initial : Option Assignment)
    (outputs : Fin arity.total → Outcome
      (Challenge := Challenge algebra) (Assignment := Assignment)) : Prop :=
  ∃ fork : CompleteFork semantics params algebra batch,
    fork.base.challenges = scalarVector algebra vector ∧
    some fork.base.assignment = initial ∧
    ∀ coordinate,
      (fork.forks coordinate).challenges = Function.update (scalarVector algebra vector)
        coordinate (outputs coordinate).1.val ∧
      some (fork.forks coordinate).assignment = (outputs coordinate).2

variable [DecidableEq Scalar] [Fintype (Challenge algebra)]
  [Nonempty (Challenge algebra)] [Fintype Assignment]

omit [Nonempty (Challenge algebra)] in
private theorem returningMass_checked
    (oracle : Oracle (Fin arity.total) (Challenge algebra) Assignment)
    (check : Response Assignment Scalar params arity → Bool)
    (vector : Fin arity.total → Challenge algebra) (coordinate : Fin arity.total)
    (output : Outcome (Challenge := Challenge algebra) (Assignment := Assignment))
    (positive : 0 < returningMass oracle (oracleCheck algebra check) vector coordinate output) :
    output.1 ≠ vector coordinate ∧
      callAccepted (oracleCheck algebra check) coordinate
        (Equiv.funSplitAt coordinate (Challenge algebra) vector).2 output = true := by
  unfold returningMass at positive
  split at positive
  next changed =>
    refine ⟨changed, ?_⟩
    by_cases accepted : callAccepted (oracleCheck algebra check) coordinate
        (Equiv.funSplitAt coordinate (Challenge algebra) vector).2 output = true
    · exact accepted
    · have numeratorZero : acceptedCallMass oracle (oracleCheck algebra check) coordinate
          (Equiv.funSplitAt coordinate (Challenge algebra) vector).2 output = 0 := by
        unfold acceptedCallMass
        exact if_neg accepted
      have noMass : stoppedMass oracle (oracleCheck algebra check) vector coordinate output = 0 := by
        unfold stoppedMass
        rw [numeratorZero, zero_div]
      rw [noMass] at positive
      exact False.elim ((lt_irrefl 0) positive)
  next unchanged => simp at positive

private theorem outcomeMass_nonnegative
    (oracle : Oracle (Fin arity.total) (Challenge algebra) Assignment)
    (check : Response Assignment Scalar params arity → Bool)
    (vector : Fin arity.total → Challenge algebra) (initial : Option Assignment)
    (outputs : Fin arity.total → Outcome
      (Challenge := Challenge algebra) (Assignment := Assignment)) :
    0 ≤ outcomeMass oracle (oracleCheck algebra check) vector initial outputs := by
  apply mul_nonneg _ (Finset.prod_nonneg fun coordinate _ =>
    returningMass_nonnegative oracle (oracleCheck algebra check) vector coordinate _)
  unfold responseAcceptedMass
  split
  · exact oracle.nonnegative vector initial
  · exact le_rfl

/-- A positive atom of the actual stopped-output law supplies a complete fork
of the exact verifier-computed CE(B) responses. No source opening is supplied. -/
theorem positive_outcome_implies_event
    (oracle : Oracle (Fin arity.total) (Challenge algebra) Assignment)
    (check : Response Assignment Scalar params arity → Bool)
    (check_spec : ∀ found, check found = true ↔
      found.Success semantics params algebra batch)
    (vector : Fin arity.total → Challenge algebra) (initial : Option Assignment)
    (outputs : Fin arity.total → Outcome
      (Challenge := Challenge algebra) (Assignment := Assignment))
    (positive : 0 < outcomeMass oracle (oracleCheck algebra check) vector initial outputs) :
    Event algebra batch vector initial outputs := by
  have baseAccepted : accepted (oracleCheck algebra check) vector initial = true := by
    by_cases hit : accepted (oracleCheck algebra check) vector initial = true
    · exact hit
    · simp [outcomeMass, responseAcceptedMass, hit] at positive
  have checked : ∀ coordinate,
      (outputs coordinate).1 ≠ vector coordinate ∧
        callAccepted (oracleCheck algebra check) coordinate
          (Equiv.funSplitAt coordinate (Challenge algebra) vector).2 (outputs coordinate) = true := by
    intro coordinate
    have nonzero : returningMass oracle (oracleCheck algebra check) vector coordinate
        (outputs coordinate) ≠ 0 := by
      intro zero
      have productZero :
          (∏ index, returningMass oracle (oracleCheck algebra check) vector index (outputs index)) = 0 :=
        Finset.prod_eq_zero (Finset.mem_univ coordinate) zero
      simp [outcomeMass, productZero] at positive
    exact returningMass_checked algebra oracle check vector coordinate _
      (lt_of_le_of_ne (returningMass_nonnegative oracle (oracleCheck algebra check)
        vector coordinate _) (Ne.symm nonzero))
  cases initial with
  | none => simp [accepted] at baseAccepted
  | some baseAssignment =>
      have assignmentsExist : ∀ coordinate, ∃ assignment,
          (outputs coordinate).2 = some assignment := by
        intro coordinate
        have hit := (checked coordinate).2
        cases result : (outputs coordinate).2 with
        | none => simp [callAccepted, accepted, result] at hit
        | some assignment => exact ⟨assignment, rfl⟩
      choose assignments assigned using assignmentsExist
      let base : Response Assignment Scalar params arity := response algebra vector baseAssignment
      let forks (coordinate : Fin arity.total) : Response Assignment Scalar params arity :=
        CoordinateForkSampler.response algebra (scalarVector algebra vector) coordinate
          ((outputs coordinate).1, assignments coordinate)
      let candidates (coordinate : Fin arity.total) : List (CoordinateForkSampler.Candidate algebra) :=
        [((outputs coordinate).1, assignments coordinate)]
      have baseSuccess : base.Success semantics params algebra batch := by
        apply (check_spec base).mp
        simpa [accepted, oracleCheck, base] using baseAccepted
      have baseStrong : ∀ index, algebra.challengeValid (base.challenges index) :=
        fun index => (vector index).property
      have sampled : ∀ coordinate,
          CoordinateForkSampler.forkResponse algebra check base.challenges coordinate
            (candidates coordinate) = some (forks coordinate) := by
        intro coordinate
        have hit : check (forks coordinate) = true := by
          have checkedCall := (checked coordinate).2
          rw [show outputs coordinate = ((outputs coordinate).1, some (assignments coordinate)) from
            Prod.ext rfl (assigned coordinate)] at checkedCall
          simpa only [callAccepted_some, forks] using checkedCall
        change check (CoordinateForkSampler.response algebra base.challenges coordinate
          ((outputs coordinate).1, assignments coordinate)) = true at hit
        have changed : base.challenges coordinate ≠ (forks coordinate).challenges coordinate := by
          change (vector coordinate).val ≠
            Function.update (scalarVector algebra vector) coordinate (outputs coordinate).1.val coordinate
          rw [Function.update_self]
          intro same
          exact (checked coordinate).1 (Subtype.ext same.symm)
        simp only [CoordinateForkSampler.forkResponse, CoordinateForkSampler.firstResponse,
          candidates, List.find?_cons, hit, Option.map_some, Option.bind_some]
        exact if_pos changed
      let fork := CoordinateForkSampler.completeForkOfSamples algebra batch check check_spec
        base baseSuccess baseStrong candidates forks sampled
      refine ⟨fork, rfl, rfl, ?_⟩
      intro coordinate
      exact ⟨rfl, (assigned coordinate).symm⟩

/-- Probability mass restricted to the typed complete-fork event. -/
noncomputable def eventMass
    (oracle : Oracle (Fin arity.total) (Challenge algebra) Assignment)
    (check : Response Assignment Scalar params arity → Bool) : ℝ := by
  classical
  exact 𝔼 vector, ∑ initial, ∑ outputs,
    if Event algebra batch vector initial outputs then
      outcomeMass oracle (oracleCheck algebra check) vector initial outputs else 0

/-- The event restriction removes no positive mass from the actual output law. -/
theorem eventMass_eq_successMass
    (oracle : Oracle (Fin arity.total) (Challenge algebra) Assignment)
    (check : Response Assignment Scalar params arity → Bool)
    (check_spec : ∀ found, check found = true ↔
      found.Success semantics params algebra batch) :
    eventMass algebra batch oracle check = successMass oracle (oracleCheck algebra check) := by
  classical
  unfold eventMass successMass
  apply Finset.expect_congr rfl
  intro vector _
  apply Finset.sum_congr rfl
  intro initial _
  apply Finset.sum_congr rfl
  intro outputs _
  by_cases event : Event algebra batch vector initial outputs
  · simp [event]
  · have noPositive : ¬ 0 < outcomeMass oracle (oracleCheck algebra check) vector initial outputs :=
      fun positive => event (positive_outcome_implies_event algebra batch oracle check check_spec
        vector initial outputs positive)
    have zero : outcomeMass oracle (oracleCheck algebra check) vector initial outputs = 0 :=
      le_antisymm (le_of_not_gt noPositive) (outcomeMass_nonnegative algebra oracle check vector initial outputs)
    simp [event, zero]

/-- The paper loss bound now names the exact successful `CompleteFork` event. -/
theorem completeFork_probability_lower_bound
    (oracle : Oracle (Fin arity.total) (Challenge algebra) Assignment)
    (check : Response Assignment Scalar params arity → Bool)
    (check_spec : ∀ found, check found = true ↔
      found.Success semantics params algebra batch) :
    (line oracle (oracleCheck algebra check)).rate -
        (arity.total : ℝ) / Fintype.card (Challenge algebra) ≤
      eventMass algebra batch oracle check := by
  rw [eventMass_eq_successMass algebra batch oracle check check_spec]
  simpa only [Fintype.card_fin] using
    successMass_lower_bound oracle (oracleCheck algebra check)

end NightstreamFPrime.Spec.Folding.PiRLC.CoordinateForkEvent
