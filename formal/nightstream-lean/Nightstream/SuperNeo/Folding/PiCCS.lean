import Nightstream.SuperNeo.Folding.BatchArity
import Nightstream.SuperNeo.SumCheck

/-!
Model-level Π_CCS reduction (SuperNeo Lemma 3) at the production batch shape.

One execution owns the whole fresh/running input product, the whole CE output
product, and the two joint FE/NC SumCheck transcripts used by the implementation.
The independent truth paths establish payload truth and fresh-norm truth for
every coordinate.  Acceptance never carries either conclusion.
-/

namespace Nightstream.SuperNeo.Folding.PiCCS

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment
  uChallenge uValue

/-- One member of the Π_CCS input product: either fresh CCS or running CE. -/
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
    Source Structure PublicInput Point Evaluation Commitment → Structure
  | .ccs statement => statement.constraintSystem
  | .ce statement => statement.constraintSystem

def commitment
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment} :
    Source Structure PublicInput Point Evaluation Commitment → Commitment
  | .ccs statement => statement.commitment
  | .ce statement => statement.commitment

def publicInput
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment} :
    Source Structure PublicInput Point Evaluation Commitment → PublicInput
  | .ccs statement => statement.publicInput
  | .ce statement => statement.publicInput

def stage
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment} :
    Source Structure PublicInput Point Evaluation Commitment → NormStage
  | .ccs statement => statement.stage
  | .ce statement => statement.stage

/-- Relation truth checked by joint FE, excluding opening authority and norm. -/
def PayloadTruth
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (source : Source Structure PublicInput Point Evaluation Commitment)
    (assignment : Assignment) : Prop :=
  match source with
  | .ccs statement =>
      semantics.ccsSatisfied statement.constraintSystem assignment
  | .ce statement =>
      semantics.evaluationPointValid statement.constraintSystem statement.point ∧
      semantics.evaluations statement.constraintSystem assignment statement.point =
        statement.evaluations

/-- Actual membership in one coordinate of the input product. -/
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

/-- Production-shaped input product: `K` fresh CCS and either zero or `k`
running CE statements. -/
structure InputProduct
    (Structure : Type uStructure)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (params : GlobalParams)
    (arity : BatchArity params) where
  fresh : Fin arity.freshCount → CCS.Instance Structure PublicInput Commitment
  running : Fin (arity.mode.count params) →
    CE.Instance Structure PublicInput Point Evaluation Commitment

namespace InputProduct

/-- Unified source order consumed jointly by FE, NC, and Π_RLC. -/
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
    Fin arity.total → Source Structure PublicInput Point Evaluation Commitment :=
  Fin.addCases (fun i => .ccs (input.fresh i)) (fun i => .ce (input.running i))

/-- Prove a property of every unified source by proving it on the two owned
input families. -/
theorem sourceCases
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {params : GlobalParams}
    {arity : BatchArity params}
    (input : InputProduct
      Structure PublicInput Point Evaluation Commitment params arity)
    (motive : Source Structure PublicInput Point Evaluation Commitment → Prop)
    (fresh : ∀ i, motive (.ccs (input.fresh i)))
    (running : ∀ i, motive (.ce (input.running i))) :
    ∀ i, motive (input.source i) := by
  intro i
  refine Fin.addCases ?_ ?_ i
  · intro j
    simpa [source] using fresh j
  · intro j
    simpa [source] using running j

end InputProduct

/-- One joint production Π_CCS execution. -/
structure Attempt
    (Structure : Type uStructure)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (Challenge : Type uChallenge)
    (Value : Type uValue)
    (params : GlobalParams)
    (arity : BatchArity params) where
  inputs : InputProduct
    Structure PublicInput Point Evaluation Commitment params arity
  outputs : Fin arity.total →
    CE.Instance Structure PublicInput Point Evaluation Commitment
  /-- One joint constraint/evaluation (FE) transcript for the whole batch. -/
  fe : SumCheck.Instance Challenge Value
  /-- One joint norm-check (NC) transcript for the whole batch. -/
  nc : SumCheck.Instance Challenge Value

/-- Exact public-data preservation and the single shared output point. -/
structure Shape
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (attempt : Attempt
      Structure PublicInput Point Evaluation Commitment Challenge Value params arity) : Prop where
  sourceFresh : ∀ i, (attempt.inputs.source i).stage = .fresh
  outputFresh : ∀ i, (attempt.outputs i).stage = .fresh
  sameStructure : ∀ i,
    (attempt.outputs i).constraintSystem = (attempt.inputs.source i).constraintSystem
  sameCommitment : ∀ i,
    (attempt.outputs i).commitment = (attempt.inputs.source i).commitment
  samePublicInput : ∀ i,
    (attempt.outputs i).publicInput = (attempt.inputs.source i).publicInput
  sharedOutputPoint : ∀ i j, (attempt.outputs i).point = (attempt.outputs j).point

/-- Verifier acceptance contains shape plus exactly the two claimed chains. -/
def Accepted
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
    (attempt : Attempt
      Structure PublicInput Point Evaluation Commitment Challenge Value params arity) : Prop :=
  Shape attempt ∧ SumCheck.Accepted ops attempt.fe ∧ SumCheck.Accepted ops attempt.nc

/-- Interpret one Π_CCS output in the relaxed target relation `CE(q/2)`. -/
def relaxedOutput
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (statement : CE.Instance Structure PublicInput Point Evaluation Commitment) :
    CE.Instance Structure PublicInput Point Evaluation Commitment :=
  { statement with stage := .ambient }

/-- Semantic obligations for the two joint SumChecks.  Both are equivalences
to independently defined truth, never acceptance-implies-validity fields. -/
structure Arithmetization
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    (ops : SumCheck.Ops Challenge Value)
    {arity : BatchArity params}
    (attempt : Attempt
      Structure PublicInput Point Evaluation Commitment Challenge Value params arity)
    (assignments : Fin arity.total → Assignment) : Prop where
  feTruthPath : SumCheck.TruthPath ops attempt.fe
  ncTruthPath : SumCheck.TruthPath ops attempt.nc
  feClaimTrue_iff :
    SumCheck.Claim.True attempt.fe ↔
      ∀ i, Source.PayloadTruth semantics (attempt.inputs.source i) (assignments i)
  ncClaimTrue_iff :
    SumCheck.Claim.True attempt.nc ↔
      ∀ i, semantics.normBounded params.b (assignments i)

/-- The phase-tagged bad event exposed by either false accepted joint claim. -/
inductive BadChallenge
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (attempt : Attempt
      Structure PublicInput Point Evaluation Commitment Challenge Value params arity) where
  | fe (round : SumCheck.Round Challenge Value)
      (evidence : SumCheck.BadChallenge attempt.fe round)
  | nc (round : SumCheck.Round Challenge Value)
      (evidence : SumCheck.BadChallenge attempt.nc round)

/-- Canonical output for one source at the verifier's new shared point. -/
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

/-- Canonical joint output product at one shared evaluation point. -/
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
    (assignments : Fin arity.total → Assignment)
    (point : Point) :
    Fin arity.total → CE.Instance Structure PublicInput Point Evaluation Commitment :=
  fun i => honestOutput semantics (input.source i) (assignments i) point

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
    CE.Holds semantics params (honestOutput semantics source assignment point) assignment := by
  cases source with
  | ccs statement =>
      rcases sourceValid with ⟨opening, _⟩
      have statementFresh : statement.stage = .fresh := by
        simpa [Source.stage] using sourceFresh
      exact ⟨⟨opening.1, opening.2.1, by
        simpa [honestOutput, statementFresh] using opening.2.2⟩, pointValid, rfl⟩
  | ce statement =>
      rcases sourceValid with ⟨opening, _, _⟩
      have statementFresh : statement.stage = .fresh := by
        simpa [Source.stage] using sourceFresh
      exact ⟨⟨opening.1, opening.2.1, by
        simpa [honestOutput, statementFresh] using opening.2.2⟩, pointValid, rfl⟩

/-- Product completeness at both bootstrap and active production arities. -/
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
    (assignments : Fin arity.total → Assignment)
    (point : Point)
    (sourceFresh : ∀ i, (input.source i).stage = .fresh)
    (sourceValid : ∀ i, (input.source i).Holds semantics params (assignments i))
    (pointValid : ∀ i,
      semantics.evaluationPointValid (input.source i).constraintSystem point) :
    ∀ i, CE.Holds semantics params
      (honestOutputs semantics input assignments point i) (assignments i) := by
  intro i
  exact honestOutput_holds semantics params (input.source i) (assignments i) point
    (sourceFresh i) (sourceValid i) (pointValid i)

/-- Perfect completeness of the one-joint-attempt production model. -/
theorem complete
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    (ops : SumCheck.Ops Challenge Value)
    (arity : BatchArity params)
    (input : InputProduct
      Structure PublicInput Point Evaluation Commitment params arity)
    (assignments : Fin arity.total → Assignment)
    (point : Point)
    (fe nc : SumCheck.Instance Challenge Value)
    (sourceFresh : ∀ i, (input.source i).stage = .fresh)
    (sourceValid : ∀ i, (input.source i).Holds semantics params (assignments i))
    (pointValid : ∀ i,
      semantics.evaluationPointValid (input.source i).constraintSystem point)
    (feTruth : SumCheck.TruthPath ops fe)
    (ncTruth : SumCheck.TruthPath ops nc)
    (feHonest : SumCheck.Honest fe)
    (ncHonest : SumCheck.Honest nc) :
    let attempt : Attempt
        Structure PublicInput Point Evaluation Commitment Challenge Value params arity := {
      inputs := input
      outputs := honestOutputs semantics input assignments point
      fe := fe
      nc := nc
    }
    Accepted ops attempt ∧
      ∀ i, CE.Holds semantics params (attempt.outputs i) (assignments i) := by
  dsimp only
  constructor
  · exact ⟨{
      sourceFresh := sourceFresh
      outputFresh := fun _ => rfl
      sameStructure := fun _ => rfl
      sameCommitment := fun _ => rfl
      samePublicInput := fun _ => rfl
      sharedOutputPoint := fun _ _ => rfl
    }, SumCheck.complete ops fe feTruth feHonest,
      SumCheck.complete ops nc ncTruth ncHonest⟩
  · exact product_complete semantics params arity input assignments point
      sourceFresh sourceValid pointValid

private theorem source_holds_of_relaxed_output
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
    (output : CE.Instance Structure PublicInput Point Evaluation Commitment)
    (assignment : Assignment)
    (sourceFresh : source.stage = .fresh)
    (sameCommitment : output.commitment = source.commitment)
    (samePublicInput : output.publicInput = source.publicInput)
    (outputValid : CE.Holds semantics params (relaxedOutput output) assignment)
    (freshNorm : semantics.normBounded params.b assignment)
    (payloadTruth : Source.PayloadTruth semantics source assignment) :
    source.Holds semantics params assignment := by
  rcases outputValid with ⟨outputOpening, _, _⟩
  cases sourceEq : source with
  | ccs statement =>
      have statementFresh : statement.stage = .fresh := by
        simpa [Source.stage, sourceEq] using sourceFresh
      have commitmentEq : (relaxedOutput output).commitment = statement.commitment := by
        change output.commitment = statement.commitment
        exact sameCommitment.trans (by simp [sourceEq, Source.commitment])
      have publicInputEq : (relaxedOutput output).publicInput = statement.publicInput := by
        change output.publicInput = statement.publicInput
        exact samePublicInput.trans (by simp [sourceEq, Source.publicInput])
      exact ⟨⟨outputOpening.1.trans commitmentEq,
        outputOpening.2.1.trans publicInputEq, by
          simpa [statementFresh] using freshNorm⟩, by
            simpa [Source.PayloadTruth, sourceEq] using payloadTruth⟩
  | ce statement =>
      have statementFresh : statement.stage = .fresh := by
        simpa [Source.stage, sourceEq] using sourceFresh
      have commitmentEq : (relaxedOutput output).commitment = statement.commitment := by
        change output.commitment = statement.commitment
        exact sameCommitment.trans (by simp [sourceEq, Source.commitment])
      have publicInputEq : (relaxedOutput output).publicInput = statement.publicInput := by
        change output.publicInput = statement.publicInput
        exact samePublicInput.trans (by simp [sourceEq, Source.publicInput])
      have payload :
          semantics.evaluationPointValid statement.constraintSystem statement.point ∧
          semantics.evaluations statement.constraintSystem assignment statement.point =
            statement.evaluations := by
        simpa [Source.PayloadTruth, sourceEq] using payloadTruth
      exact ⟨⟨outputOpening.1.trans commitmentEq,
        outputOpening.2.1.trans publicInputEq, by
          simpa [statementFresh] using freshNorm⟩, payload.1, payload.2⟩

/-- Strong joint Π_CCS extraction.  Either every source opens, or one of the
two accepted joint SumChecks exposes a concrete bad sampled challenge. -/
theorem strong_extract_or_bad_challenge
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    (ops : SumCheck.Ops Challenge Value)
    {arity : BatchArity params}
    (attempt : Attempt
      Structure PublicInput Point Evaluation Commitment Challenge Value params arity)
    (assignments : Fin arity.total → Assignment)
    (accepted : Accepted ops attempt)
    (arithmetization : Arithmetization semantics params ops attempt assignments)
    (outputValid : ∀ i,
      CE.Holds semantics params (relaxedOutput (attempt.outputs i)) (assignments i)) :
    (∀ i, (attempt.inputs.source i).Holds semantics params (assignments i)) ∨
      Nonempty (BadChallenge attempt) := by
  rcases accepted with ⟨shape, feAccepted, ncAccepted⟩
  by_cases feTrue : SumCheck.Claim.True attempt.fe
  · by_cases ncTrue : SumCheck.Claim.True attempt.nc
    · left
      intro i
      exact source_holds_of_relaxed_output semantics params
        (attempt.inputs.source i) (attempt.outputs i) (assignments i)
        (shape.sourceFresh i) (shape.sameCommitment i) (shape.samePublicInput i)
        (outputValid i) (arithmetization.ncClaimTrue_iff.mp ncTrue i)
        (arithmetization.feClaimTrue_iff.mp feTrue i)
    · right
      rcases SumCheck.false_acceptance_implies_bad_challenge ops attempt.nc
          ncAccepted arithmetization.ncTruthPath ncTrue with ⟨round, evidence⟩
      exact ⟨BadChallenge.nc round evidence⟩
  · right
    rcases SumCheck.false_acceptance_implies_bad_challenge ops attempt.fe
        feAccepted arithmetization.feTruthPath feTrue with ⟨round, evidence⟩
    exact ⟨BadChallenge.fe round evidence⟩

/-- The strong reduction projects the complete vector of output commitments. -/
def phi
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (attempt : Attempt
      Structure PublicInput Point Evaluation Commitment Challenge Value params arity) :
    Fin arity.total → Commitment :=
  fun i => (attempt.outputs i).commitment

/-- Repeated joint executions for one input product preserve the same `φ`. -/
theorem repeated_outputs_same_phi
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (left right : Attempt
      Structure PublicInput Point Evaluation Commitment Challenge Value params arity)
    (sameInputs : left.inputs = right.inputs)
    (leftShape : Shape left)
    (rightShape : Shape right) :
    phi left = phi right := by
  funext i
  calc
    phi left i = (left.inputs.source i).commitment := leftShape.sameCommitment i
    _ = (right.inputs.source i).commitment := by rw [sameInputs]
    _ = phi right i := (rightShape.sameCommitment i).symm

end Nightstream.SuperNeo.Folding.PiCCS
