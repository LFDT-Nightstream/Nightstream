import NightstreamFPrime.Spec.Folding.PiRLC.CoordinateTerminalProgram
import NightstreamFPrime.Spec.Folding.PiRLC.CoordinateForkEvent

/-!
The deterministic terminal program consumes the exact observed responses.
A CompleteFork is used only in proofs to identify those data and their source
openings. Successful execution also forces all endpoint presence and distinct
challenge checks; it cannot bypass a failed endpoint.
-/

namespace NightstreamFPrime.Spec.Folding.PiRLC.CoordinateTerminalSemantics

open NightstreamFPrime.Spec
open PaperForkExtraction PaperForkExtractionWork CoordinateTerminalProgram
open CoordinateForkLaw CoordinateForkEvent CoordinateOracle

universe uScalar uAssignment uStructure uPublicInput uPoint uEvaluation uCommitment

private theorem gather_returns_implies_endpoints
    {Scalar : Type uScalar} {Assignment : Type uAssignment} {member : Scalar → Prop}
    [DecidableEq Scalar] (program : Primitives Scalar Assignment) : ∀ {count : Nat}
    (vector : Fin count → {scalar // member scalar}) (base : Assignment)
    (outputs : Fin count → ({scalar // member scalar} × Option Assignment))
    (values : List Assignment),
    (gather program vector base outputs).value = some values →
      (∀ index, (outputs index).2.isSome = true) ∧
      (∀ index, (vector index).val ≠ (outputs index).1.val)
  | 0, _, _, _, _, _ => ⟨fun index => Fin.elim0 index, fun index => Fin.elim0 index⟩
  | _ + 1, vector, base, outputs, values, returned => by
      cases headResponse : (outputs 0).2 with
      | none =>
          simp only [gather, headResponse] at returned
          cases returned
      | some assignment =>
          by_cases same : (vector 0).val = (outputs 0).1.val
          · simp only [gather, headResponse, same, ↓reduceIte] at returned
            cases returned
          · cases tailResponse : (gather program (fun index => vector index.succ) base
                (fun index => outputs index.succ)).value with
            | none =>
                simp only [gather, headResponse, same, ↓reduceIte, tailResponse,
                  Option.map_none] at returned
                cases returned
            | some tailValues =>
                have tail := gather_returns_implies_endpoints program
                  (fun index => vector index.succ) base (fun index => outputs index.succ)
                  tailValues tailResponse
                constructor
                · intro index
                  refine Fin.cases ?_ (fun prior => tail.1 prior) index
                  simp only [headResponse, Option.isSome_some]
                · intro index
                  exact Fin.cases same (fun prior => tail.2 prior) index

/-- Every successful terminal return passed the actual presence and distinctness
checks. This direction does not require a CompleteFork or valid openings. -/
theorem finish_returns_implies_endpoints
    {Scalar : Type uScalar} {Assignment : Type uAssignment} {member : Scalar → Prop}
    [DecidableEq Scalar] (program : Primitives Scalar Assignment) {count : Nat}
    (vector : Fin count → {scalar // member scalar}) (initial : Option Assignment)
    (outputs : Fin count → ({scalar // member scalar} × Option Assignment))
    (values : List Assignment)
    (returned : (finish program vector initial outputs).value = some values) :
    ∃ base, initial = some base ∧
      (∀ index, (outputs index).2.isSome = true) ∧
      (∀ index, (vector index).val ≠ (outputs index).1.val) := by
  cases initial with
  | none =>
      simp only [finish] at returned
      cases returned
  | some base =>
      exact ⟨base, rfl, gather_returns_implies_endpoints program vector base outputs values returned⟩

variable {Structure : Type uStructure} {Assignment : Type uAssignment}
  {PublicInput : Type uPublicInput} {Point : Type uPoint}
  {Evaluation : Type uEvaluation} {Commitment : Type uCommitment} {Scalar : Type uScalar}
  {semantics : RelationSemantics
    Structure Assignment PublicInput Point Evaluation Commitment}
  {params : GlobalParams} {arity : BatchArity params}
  (algebra : Algebra Structure Assignment PublicInput Point Evaluation Commitment
    Scalar semantics params)
  (batch : InputBatch Structure PublicInput Point Evaluation Commitment params arity)
  (laws : ExtractionAlgebra semantics params algebra)
  (strongSet : StrongSetUnits laws.ring algebra.challengeValid)
  (program : Primitives Scalar Assignment)
  (correct : Correct laws.ring laws.assignmentModule program)
  [DecidableEq Scalar]

include strongSet correct

/-- The actual endpoint program returns the exact existing extracted assignments,
preserving the fork's base and coordinate response values. -/
theorem event_implies_result
    (vector : Fin arity.total → Challenge algebra) (initial : Option Assignment)
    (outputs : Fin arity.total → Outcome
      (Challenge := Challenge algebra) (Assignment := Assignment))
    (event : Event algebra batch vector initial outputs) :
    ∃ fork : CompleteFork semantics params algebra batch,
      fork.base.challenges = scalarVector algebra vector ∧
      some fork.base.assignment = initial ∧
      (∀ coordinate,
        (fork.forks coordinate).challenges = Function.update (scalarVector algebra vector)
          coordinate (outputs coordinate).1.val ∧
        some (fork.forks coordinate).assignment = (outputs coordinate).2) ∧
      (finish program vector initial outputs).value =
        some (List.ofFn (extractedAssignment laws strongSet fork)) ∧
      (∀ coordinate, PaperCorrections.CorrectedAmbientHolds semantics params
        (batch.inputs coordinate) (extractedAssignment laws strongSet fork coordinate)) := by
  rcases event with ⟨fork, baseVector, baseAssignment, responses⟩
  refine ⟨fork, baseVector, baseAssignment, responses, ?_, ?_⟩
  · have baseAt (coordinate : Fin arity.total) :
        fork.base.challenges coordinate = (vector coordinate).val :=
      congrFun baseVector coordinate
    have forkAt (coordinate : Fin arity.total) :
        (fork.forks coordinate).challenges coordinate = (outputs coordinate).1.val := by
      rw [(responses coordinate).1, Function.update_self]
    have different (coordinate : Fin arity.total) :
        (vector coordinate).val ≠ (outputs coordinate).1.val := by
      simpa only [baseAt coordinate, forkAt coordinate] using fork.changed coordinate
    have result := gather_values program vector fork.base.assignment outputs
      (fun coordinate => (fork.forks coordinate).assignment)
      (fun coordinate => (responses coordinate).2.symm) different
    rw [← baseAssignment]
    change (gather program vector fork.base.assignment outputs).value = _
    rw [result]
    apply congrArg some
    apply congrArg List.ofFn
    funext coordinate
    have equal := coordinate_value laws strongSet program correct fork coordinate
    simpa only [baseAt coordinate, forkAt coordinate] using equal
  · exact completeFork_implies_correctedAmbientHolds semantics params arity algebra laws strongSet
      batch fork

/-- The successful fork event causes the actual program to return a value. -/
theorem event_implies_returns
    (vector : Fin arity.total → Challenge algebra) (initial : Option Assignment)
    (outputs : Fin arity.total → Outcome
      (Challenge := Challenge algebra) (Assignment := Assignment))
    (event : Event algebra batch vector initial outputs) :
    (finish program vector initial outputs).value.isSome = true := by
  obtain ⟨fork, _baseVector, _baseAssignment, _responses, returned, _openings⟩ :=
    event_implies_result algebra batch laws strongSet program correct vector initial outputs event
  rw [returned]
  rfl

end NightstreamFPrime.Spec.Folding.PiRLC.CoordinateTerminalSemantics
