import Nightstream.SuperNeo.Folding.BatchArity

/-!
Paper-level source and output products for one SuperNeo fold.

Owns: the ordered fresh/running source product, source membership, the common
verifier-selected output point, and honest CE output completeness.

Does not own: a SumCheck instance, transcript events, Fiat--Shamir, PiRLC,
PiDEC, Rust, R1CS, or constraint counts.

Assurance tier: model-level.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperProduct

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment

/-- One source in paper order: fresh CCS sources first, then running CE
sources. -/
inductive Source
    (Structure : Type uStructure)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment) where
  | ccs (statement : CCS.Instance Structure PublicInput Commitment)
  | ce (statement : CE.Instance Structure PublicInput Point Evaluation Commitment)

namespace Source

def constraintSystem
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment} :
    Source Structure PublicInput Point Evaluation Commitment -> Structure
  | .ccs statement => statement.constraintSystem
  | .ce statement => statement.constraintSystem

def commitment
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment} :
    Source Structure PublicInput Point Evaluation Commitment -> Commitment
  | .ccs statement => statement.commitment
  | .ce statement => statement.commitment

def publicInput
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment} :
    Source Structure PublicInput Point Evaluation Commitment -> PublicInput
  | .ccs statement => statement.publicInput
  | .ce statement => statement.publicInput

def stage
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment} :
    Source Structure PublicInput Point Evaluation Commitment -> NormStage
  | .ccs statement => statement.stage
  | .ce statement => statement.stage

def Holds
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    (source : Source Structure PublicInput Point Evaluation Commitment)
    (assignment : Assignment) : Prop :=
  match source with
  | .ccs statement => CCS.Holds semantics params statement assignment
  | .ce statement => CE.Holds semantics params statement assignment

end Source

/-- Exact fresh/running source product selected by one batch arity. -/
structure InputProduct
    (Structure : Type uStructure)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (params : GlobalParams)
    (arity : BatchArity params) where
  fresh : Fin arity.freshCount -> CCS.Instance Structure PublicInput Commitment
  running : Fin (arity.mode.count params) ->
    CE.Instance Structure PublicInput Point Evaluation Commitment

namespace InputProduct

/-- Canonical source order used by the paper polynomial and PiRLC. -/
def source
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {params : GlobalParams}
    {arity : BatchArity params}
    (input : InputProduct
      Structure PublicInput Point Evaluation Commitment params arity) :
    Fin arity.total -> Source Structure PublicInput Point Evaluation Commitment :=
  Fin.addCases (fun index => .ccs (input.fresh index))
    (fun index => .ce (input.running index))

end InputProduct

/-- Canonical output for one source at the verifier-selected point. -/
def honestOutput
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (source : Source Structure PublicInput Point Evaluation Commitment)
    (assignment : Assignment)
    (point : Point) :
    CE.Instance Structure PublicInput Point Evaluation Commitment where
  constraintSystem := source.constraintSystem
  commitment := source.commitment
  publicInput := source.publicInput
  point := point
  evaluations := semantics.evaluations source.constraintSystem assignment point
  stage := .fresh

/-- Canonical output product at one shared evaluation point. -/
def honestOutputs
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {params : GlobalParams}
    {arity : BatchArity params}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (input : InputProduct
      Structure PublicInput Point Evaluation Commitment params arity)
    (assignments : Fin arity.total -> Assignment)
    (point : Point) :
    Fin arity.total -> CE.Instance Structure PublicInput Point Evaluation Commitment :=
  fun index => honestOutput semantics (input.source index)
    (assignments index) point

/-- Honest source membership is preserved at the new evaluation point. -/
theorem honestOutput_holds
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    (source : Source Structure PublicInput Point Evaluation Commitment)
    (assignment : Assignment)
    (point : Point)
    (sourceFresh : source.stage = .fresh)
    (sourceValid : source.Holds semantics params assignment)
    (pointValid : semantics.evaluationPointValid source.constraintSystem point) :
    CE.Holds semantics params (honestOutput semantics source assignment point)
      assignment := by
  cases source with
  | ccs statement =>
      rcases sourceValid with ⟨opening, relation⟩
      have statementFresh : statement.stage = .fresh := by
        simpa [Source.stage] using sourceFresh
      exact ⟨⟨opening.1, opening.2.1, by
        simpa [honestOutput, statementFresh] using opening.2.2⟩,
        pointValid, rfl⟩
  | ce statement =>
      rcases sourceValid with ⟨opening, oldPointValid, evaluations⟩
      have statementFresh : statement.stage = .fresh := by
        simpa [Source.stage] using sourceFresh
      exact ⟨⟨opening.1, opening.2.1, by
        simpa [honestOutput, statementFresh] using opening.2.2⟩,
        pointValid, rfl⟩

/-- Honest completeness for the complete source product. -/
theorem product_complete
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    (arity : BatchArity params)
    (input : InputProduct
      Structure PublicInput Point Evaluation Commitment params arity)
    (assignments : Fin arity.total -> Assignment)
    (point : Point)
    (sourceFresh : forall index, (input.source index).stage = .fresh)
    (sourceValid : forall index,
      (input.source index).Holds semantics params (assignments index))
    (pointValid : forall index,
      semantics.evaluationPointValid
        (input.source index).constraintSystem point) :
    forall index, CE.Holds semantics params
      (honestOutputs semantics input assignments point index)
      (assignments index) := by
  intro index
  exact honestOutput_holds semantics params (input.source index)
    (assignments index) point (sourceFresh index) (sourceValid index)
    (pointValid index)

end Nightstream.SuperNeo.Folding.PiCCS.PaperProduct
