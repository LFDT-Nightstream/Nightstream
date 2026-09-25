import NightstreamFPrime.Spec.Folding.PiRLC.CoordinateOracle
import NightstreamFPrime.Spec.Folding.PiRLC.CoordinateForkSampler

/-!
Connects the response-trace probability law to the actual typed PiRLC search.
The challenge carrier contains only verifier-valid scalars. Oracle failures
remain in the probability trace and are omitted only when passing candidate
assignments to the existing first-response selector.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiRLC.CoordinateForkLaw

open NightstreamFPrime.Spec
open PaperForkExtraction

variable {Structure Assignment PublicInput Point Evaluation Commitment Scalar : Type*}
  {semantics : RelationSemantics
    Structure Assignment PublicInput Point Evaluation Commitment}
  {params : GlobalParams} {arity : BatchArity params}
  (algebra : Algebra Structure Assignment PublicInput Point Evaluation Commitment
    Scalar semantics params)

abbrev Challenge := {challenge : Scalar // algebra.challengeValid challenge}

def scalarVector (vector : Fin arity.total → Challenge algebra) : Fin arity.total → Scalar :=
  fun index => (vector index).val

def response (vector : Fin arity.total → Challenge algebra) (assignment : Assignment) :
    Response Assignment Scalar params arity where
  challenges := scalarVector algebra vector
  assignment := assignment

def oracleCheck (check : Response Assignment Scalar params arity → Bool)
    (vector : Fin arity.total → Challenge algebra) (assignment : Assignment) : Bool :=
  check (response algebra vector assignment)

def outcomeCandidate
    (outcome : CoordinateOracle.Outcome
      (Challenge := Challenge algebra) (Assignment := Assignment)) :
    Option (CoordinateForkSampler.Candidate algebra) :=
  outcome.2.map fun assignment => (outcome.1, assignment)

def observedCandidates
    (outcomes : List (CoordinateOracle.Outcome
      (Challenge := Challenge algebra) (Assignment := Assignment))) :
    List (CoordinateForkSampler.Candidate algebra) :=
  outcomes.filterMap (outcomeCandidate algebra)

private theorem callVector_eq_update
    (vector : Fin arity.total → Challenge algebra) (coordinate : Fin arity.total)
    (challenge : Challenge algebra) :
    CoordinateOracle.callVector coordinate
      (Equiv.funSplitAt coordinate (Challenge algebra) vector).2 challenge =
      Function.update vector coordinate challenge := by
  funext index
  by_cases atCoordinate : index = coordinate
  · subst index
    simp [CoordinateOracle.callVector, Equiv.funSplitAt, Equiv.piSplitAt]
  · simp [CoordinateOracle.callVector, Equiv.funSplitAt, Equiv.piSplitAt, atCoordinate]

theorem response_atCoordinate
    (vector : Fin arity.total → Challenge algebra) (coordinate : Fin arity.total)
    (challenge : Challenge algebra) (assignment : Assignment) :
    CoordinateForkSampler.response algebra (scalarVector algebra vector) coordinate
      (challenge, assignment) =
      response algebra (CoordinateOracle.callVector coordinate
        (Equiv.funSplitAt coordinate (Challenge algebra) vector).2 challenge) assignment := by
  rw [callVector_eq_update]
  unfold CoordinateForkSampler.response response
  apply congrArg (fun coefficients : Fin arity.total → Scalar =>
    ({ challenges := coefficients, assignment := assignment } :
      Response Assignment Scalar params arity))
  funext index
  by_cases atCoordinate : index = coordinate
  · subst index
    simp [scalarVector]
  · simp [scalarVector,
      Function.update_of_ne atCoordinate]

theorem callAccepted_some
    (check : Response Assignment Scalar params arity → Bool)
    (vector : Fin arity.total → Challenge algebra) (coordinate : Fin arity.total)
    (challenge : Challenge algebra) (assignment : Assignment) :
    CoordinateOracle.callAccepted (oracleCheck algebra check) coordinate
      (Equiv.funSplitAt coordinate (Challenge algebra) vector).2 (challenge, some assignment) =
      check (CoordinateForkSampler.response algebra (scalarVector algebra vector) coordinate
        (challenge, assignment)) := by
  simp only [CoordinateOracle.callAccepted, CoordinateOracle.accepted,
    Option.map_some, Option.getD_some, oracleCheck]
  rw [response_atCoordinate]

private theorem observedCandidates_rejected
    (check : Response Assignment Scalar params arity → Bool)
    (vector : Fin arity.total → Challenge algebra) (coordinate : Fin arity.total)
    (outcomes : List (CoordinateOracle.Outcome
      (Challenge := Challenge algebra) (Assignment := Assignment)))
    (rejected : ∀ outcome ∈ outcomes,
      CoordinateOracle.callAccepted (oracleCheck algebra check) coordinate
        (Equiv.funSplitAt coordinate (Challenge algebra) vector).2 outcome = false) :
    ∀ candidate ∈ observedCandidates algebra outcomes,
      check (CoordinateForkSampler.response algebra (scalarVector algebra vector)
        coordinate candidate) = false := by
  intro candidate member
  change candidate ∈ outcomes.filterMap (outcomeCandidate algebra) at member
  rcases List.mem_filterMap.mp member with ⟨⟨challenge, result⟩, rawMember, equal⟩
  cases result with
  | none => simp [outcomeCandidate] at equal
  | some assignment =>
      simp only [outcomeCandidate, Option.map_some, Option.some.injEq] at equal
      subst candidate
      simpa only [callAccepted_some] using rejected (challenge, some assignment) rawMember

/-- Every positive first-hit trace returns its actual final response through
the typed selector. All preceding aborting or rejected calls are accounted for. -/
theorem positive_trace_firstResponse
    [Fintype (Challenge algebra)] [Nonempty (Challenge algebra)] [Fintype Assignment]
    (oracle : CoordinateOracle.Oracle (Fin arity.total) (Challenge algebra) Assignment)
    (check : Response Assignment Scalar params arity → Bool)
    (vector : Fin arity.total → Challenge algebra) (coordinate : Fin arity.total)
    {rejections : Nat}
    (before : Fin rejections → CoordinateOracle.Outcome
      (Challenge := Challenge algebra) (Assignment := Assignment))
    (challenge : Challenge algebra) (assignment : Assignment)
    (positive : 0 < CoordinateOracle.traceMass oracle (oracleCheck algebra check) coordinate
      (Equiv.funSplitAt coordinate (Challenge algebra) vector).2 before (challenge, some assignment)) :
    CoordinateForkSampler.firstResponse algebra check (scalarVector algebra vector) coordinate
      (observedCandidates algebra (List.ofFn before ++ [(challenge, some assignment)])) =
      some (CoordinateForkSampler.response algebra (scalarVector algebra vector) coordinate
        (challenge, assignment)) := by
  have checks := CoordinateOracle.traceMass_positive_implies_checks oracle
    (oracleCheck algebra check) coordinate
    (Equiv.funSplitAt coordinate (Challenge algebra) vector).2 before
    (challenge, some assignment) positive
  have earlier := observedCandidates_rejected algebra check vector coordinate (List.ofFn before)
    (by
      intro outcome member
      rcases List.mem_ofFn.mp member with ⟨position, equal⟩
      rw [← equal]
      exact checks.1 position)
  have noEarlier :
      (observedCandidates algebra (List.ofFn before)).find?
        (fun candidate => check (CoordinateForkSampler.response algebra
          (scalarVector algebra vector) coordinate candidate)) = none := by
    apply List.find?_eq_none.mpr
    intro candidate member
    rw [earlier candidate member]
    decide
  have lastAccepted :
      check (CoordinateForkSampler.response algebra (scalarVector algebra vector) coordinate
        (challenge, assignment)) = true := by
    simpa only [callAccepted_some] using checks.2
  simp only [observedCandidates, List.filterMap_append, List.filterMap_cons,
    List.filterMap_nil, outcomeCandidate, Option.map_some]
  change ((observedCandidates algebra (List.ofFn before) ++ [(challenge, assignment)]).find?
    (fun candidate => check (CoordinateForkSampler.response algebra
      (scalarVector algebra vector) coordinate candidate))).map
        (CoordinateForkSampler.response algebra (scalarVector algebra vector) coordinate) = _
  simp [List.find?_append, noEarlier, lastAccepted]

end NightstreamFPrime.Spec.Folding.PiRLC.CoordinateForkLaw
