import Nightstream.SuperNeo.Folding.PiCCS

/-!
Output-evaluation authority gap in the abstract `PiCCS` phase model.

Protocol: SuperNeo `PiCCS`.
Phase: output CE product after the two abstract SumCheck chains.
Constraint family: semantic authority only; this file emits no rows.

Owns: evaluation-only and common-point-only replacements of the accepted
output product; proof that `PiCCS.Shape` and `PiCCS.Accepted` cannot determine
those fields; and necessity theorems exhibiting the resulting ambiguity.

Does not own: the paper-joint terminal message, production SplitNc, delayed-NC
`y_zcol`, Fiat--Shamir, Rust, R1CS, output hashing, or row removal.

Emits constraints: no.

Authority boundary: this is a fail-closed result about the current abstract
interface. It proves that `PiCCS.Accepted` alone cannot establish semantic
truth for the evaluations carried into an output digest or `PiRLC`. Exact
message hashing and downstream NIFS validation are separate obligations. A
future verifier refinement must prove those links; it may not assume this gap
away through a conversion function.

| Protocol | Phase | Family | Mathematical result |
|---|---|---|---|
| `PiCCS` | output product | evaluation replacement | only `CE.Instance.evaluations` changes |
| `PiCCS` | shape | inherited public fields | `Shape` is preserved exactly |
| `PiCCS` | acceptance | FE / NC chains | `Accepted` is preserved exactly |
| necessity | output authority | later `PiRLC` input | distinct evaluations remain possible under the same accepted abstract attempt context |
| necessity | output authority | output point | any caller-chosen common point preserves abstract acceptance |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.OutputEvaluationAuthority

universe uStructure uPublicInput uPoint uEvaluation uCommitment uChallenge
  uValue

/-- Replace only the CE evaluation arrays in one abstract `PiCCS` output
product. Inputs, both SumCheck instances, and every other output field are
definitionally unchanged. -/
def replaceEvaluations
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (attempt : Attempt Structure PublicInput Point Evaluation Commitment
      Challenge Value params arity)
    (evaluations : Fin arity.total -> Array Evaluation) :
    Attempt Structure PublicInput Point Evaluation Commitment Challenge Value
      params arity :=
  { attempt with
    outputs := fun index =>
      { attempt.outputs index with evaluations := evaluations index } }

@[simp] theorem replaceEvaluations_inputs
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (attempt : Attempt Structure PublicInput Point Evaluation Commitment
      Challenge Value params arity)
    (evaluations : Fin arity.total -> Array Evaluation) :
    (replaceEvaluations attempt evaluations).inputs = attempt.inputs := by
  rfl

@[simp] theorem replaceEvaluations_fe
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (attempt : Attempt Structure PublicInput Point Evaluation Commitment
      Challenge Value params arity)
    (evaluations : Fin arity.total -> Array Evaluation) :
    (replaceEvaluations attempt evaluations).fe = attempt.fe := by
  rfl

@[simp] theorem replaceEvaluations_nc
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (attempt : Attempt Structure PublicInput Point Evaluation Commitment
      Challenge Value params arity)
    (evaluations : Fin arity.total -> Array Evaluation) :
    (replaceEvaluations attempt evaluations).nc = attempt.nc := by
  rfl

@[simp] theorem replaceEvaluations_output_evaluations
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (attempt : Attempt Structure PublicInput Point Evaluation Commitment
      Challenge Value params arity)
    (evaluations : Fin arity.total -> Array Evaluation)
    (index : Fin arity.total) :
    ((replaceEvaluations attempt evaluations).outputs index).evaluations =
      evaluations index := by
  rfl

/-- The abstract shape predicate never inspects output evaluations. -/
theorem shape_replaceEvaluations
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    {attempt : Attempt Structure PublicInput Point Evaluation Commitment
      Challenge Value params arity}
    (evaluations : Fin arity.total -> Array Evaluation)
    (shape : Shape attempt) :
    Shape (replaceEvaluations attempt evaluations) where
  sourceFresh := shape.sourceFresh
  outputFresh := shape.outputFresh
  sameStructure := shape.sameStructure
  sameCommitment := shape.sameCommitment
  samePublicInput := shape.samePublicInput
  sharedOutputPoint := shape.sharedOutputPoint

/-- Replacing evaluations cannot create shape validity either. -/
theorem shape_of_replaceEvaluations
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    {attempt : Attempt Structure PublicInput Point Evaluation Commitment
      Challenge Value params arity}
    (evaluations : Fin arity.total -> Array Evaluation)
    (shape : Shape (replaceEvaluations attempt evaluations)) :
    Shape attempt where
  sourceFresh := shape.sourceFresh
  outputFresh := shape.outputFresh
  sameStructure := shape.sameStructure
  sameCommitment := shape.sameCommitment
  samePublicInput := shape.samePublicInput
  sharedOutputPoint := shape.sharedOutputPoint

theorem shape_replaceEvaluations_iff
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    {attempt : Attempt Structure PublicInput Point Evaluation Commitment
      Challenge Value params arity}
    (evaluations : Fin arity.total -> Array Evaluation) :
    Shape (replaceEvaluations attempt evaluations) ↔ Shape attempt := by
  exact ⟨shape_of_replaceEvaluations evaluations,
    shape_replaceEvaluations evaluations⟩

/-- Consequently the complete abstract acceptance predicate is invariant
under an arbitrary replacement of every output evaluation array. -/
theorem accepted_replaceEvaluations
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (ops : SumCheck.Ops Challenge Value)
    {attempt : Attempt Structure PublicInput Point Evaluation Commitment
      Challenge Value params arity}
    (evaluations : Fin arity.total -> Array Evaluation)
    (accepted : Accepted ops attempt) :
    Accepted ops (replaceEvaluations attempt evaluations) := by
  exact ⟨shape_replaceEvaluations evaluations accepted.1,
    accepted.2.1, accepted.2.2⟩

/-- Abstract acceptance is exactly invariant under arbitrary output-evaluation
replacement. This is stronger than an honest-completeness observation: the
accepted predicate has no way to distinguish the two products. -/
theorem accepted_replaceEvaluations_iff
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (ops : SumCheck.Ops Challenge Value)
    {attempt : Attempt Structure PublicInput Point Evaluation Commitment
      Challenge Value params arity}
    (evaluations : Fin arity.total -> Array Evaluation) :
    Accepted ops (replaceEvaluations attempt evaluations) ↔
      Accepted ops attempt := by
  constructor
  · intro accepted
    exact ⟨shape_of_replaceEvaluations evaluations accepted.1,
      accepted.2.1, accepted.2.2⟩
  · exact accepted_replaceEvaluations ops evaluations

/-- Protocol-obligation necessity: whenever one accepted attempt exists and a
different evaluation product is available, the same inputs, FE transcript,
NC transcript, and all non-evaluation output fields permit a second accepted
attempt with different outputs.

This is the exact reason a later output-authority theorem cannot be derived
from `PiCCS.Accepted` alone. -/
theorem accepted_does_not_determine_output_evaluations
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (ops : SumCheck.Ops Challenge Value)
    {attempt : Attempt Structure PublicInput Point Evaluation Commitment
      Challenge Value params arity}
    (evaluations : Fin arity.total -> Array Evaluation)
    (accepted : Accepted ops attempt)
    (different : exists index,
      evaluations index ≠ (attempt.outputs index).evaluations) :
    Accepted ops attempt /\
      Accepted ops (replaceEvaluations attempt evaluations) /\
      (replaceEvaluations attempt evaluations).inputs = attempt.inputs /\
      (replaceEvaluations attempt evaluations).fe = attempt.fe /\
      (replaceEvaluations attempt evaluations).nc = attempt.nc /\
      (replaceEvaluations attempt evaluations).outputs ≠ attempt.outputs := by
  refine ⟨accepted, accepted_replaceEvaluations ops evaluations accepted,
    rfl, rfl, rfl, ?_⟩
  intro outputsEqual
  rcases different with ⟨index, evaluationsDifferent⟩
  apply evaluationsDifferent
  have fieldEqual := congrArg
    (fun outputs => (outputs index).evaluations) outputsEqual
  exact fieldEqual

/-! ## Common output-point ambiguity -/

/-- Replace every output point by one caller-selected common point. This
operation leaves the inputs and both SumCheck chains definitionally fixed. -/
def replaceOutputPoint
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (attempt : Attempt Structure PublicInput Point Evaluation Commitment
      Challenge Value params arity)
    (point : Point) :
    Attempt Structure PublicInput Point Evaluation Commitment Challenge Value
      params arity :=
  { attempt with
    outputs := fun index => { attempt.outputs index with point := point } }

@[simp] theorem replaceOutputPoint_output_point
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (attempt : Attempt Structure PublicInput Point Evaluation Commitment
      Challenge Value params arity)
    (point : Point) (index : Fin arity.total) :
    ((replaceOutputPoint attempt point).outputs index).point = point := by
  rfl

/-- The current shape predicate requires only equality among outputs; it does
not derive the common point from verifier transcript state. -/
theorem shape_replaceOutputPoint
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    {attempt : Attempt Structure PublicInput Point Evaluation Commitment
      Challenge Value params arity}
    (point : Point) (shape : Shape attempt) :
    Shape (replaceOutputPoint attempt point) where
  sourceFresh := shape.sourceFresh
  outputFresh := shape.outputFresh
  sameStructure := shape.sameStructure
  sameCommitment := shape.sameCommitment
  samePublicInput := shape.samePublicInput
  sharedOutputPoint := fun _ _ => rfl

theorem accepted_replaceOutputPoint
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (ops : SumCheck.Ops Challenge Value)
    {attempt : Attempt Structure PublicInput Point Evaluation Commitment
      Challenge Value params arity}
    (point : Point) (accepted : Accepted ops attempt) :
    Accepted ops (replaceOutputPoint attempt point) := by
  exact ⟨shape_replaceOutputPoint point accepted.1,
    accepted.2.1, accepted.2.2⟩

/-- A caller-chosen common output point different from any existing output
point yields another accepted abstract attempt with the same inputs and the
same FE/NC transcripts. A concrete verifier model must derive this point from
its transcript rather than merely assert cross-output equality. -/
theorem accepted_does_not_determine_common_output_point
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (ops : SumCheck.Ops Challenge Value)
    {attempt : Attempt Structure PublicInput Point Evaluation Commitment
      Challenge Value params arity}
    (point : Point) (accepted : Accepted ops attempt)
    (different : exists index, point ≠ (attempt.outputs index).point) :
    Accepted ops attempt /\
      Accepted ops (replaceOutputPoint attempt point) /\
      (replaceOutputPoint attempt point).inputs = attempt.inputs /\
      (replaceOutputPoint attempt point).fe = attempt.fe /\
      (replaceOutputPoint attempt point).nc = attempt.nc /\
      (replaceOutputPoint attempt point).outputs ≠ attempt.outputs := by
  refine ⟨accepted, accepted_replaceOutputPoint ops point accepted,
    rfl, rfl, rfl, ?_⟩
  intro outputsEqual
  rcases different with ⟨index, pointDifferent⟩
  apply pointDifferent
  have fieldEqual := congrArg (fun outputs => (outputs index).point) outputsEqual
  exact fieldEqual

end Nightstream.SuperNeo.Folding.PiCCS.OutputEvaluationAuthority
