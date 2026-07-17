import Nightstream.SuperNeo.Concrete.Phi81Relation.Semantics
import Nightstream.SuperNeo.Folding.PiCCS
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.ProductAlignment
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Types

/-!
Canonical model-level `Pi_CCS` CE output materialization.

Protocol: SuperNeo `Pi_CCS`.
Phase: verifier-visible output product after FE has derived the shared row point.
Constraint family: semantic materialization only; this file emits no rows.

Owns: the fresh/running source-order alignment; the unique CE product whose
structure, commitment, and public input are copied from the corresponding
public source, whose point is the explicit FE-derived `rPrime`, and whose
evaluations are exactly `Array.ofFn message.yRing`.

Does not own: FE derivation of `rPrime`, NC derivation of `sPrime`, `yZcol`,
commitment binding, input membership, transcripts, NIFS composition, Rust,
R1CS, row removal, or constraint counts.

Emits constraints: no.

Authority boundary: `message.yRing` is prover-shaped data until
`yRingBoundToSources_iff_outputEvaluationsBound` binds every matrix and every
Phi81 lane to the sole source assignments and matrices. `message.yZcol` is
deliberately excluded from CE. It remains a delayed-NC payload owned by the
separate output-authority branch.

| Stage path | Output field | Mathematical obligation | Authority class |
|---|---|---|---|
| `nifs.pi_ccs.output.source_order` | source index | preserve fresh-then-running order, not merely total cardinality | computed |
| `nifs.pi_ccs.output.structure` | `constraintSystem` | copy the corresponding public input-source structure | direct dataflow |
| `nifs.pi_ccs.output.opening` | `commitment`, `publicInput` | copy the corresponding authoritative input-source fields | direct dataflow |
| `nifs.pi_ccs.output.point` | `point` | use the explicit FE-derived shared `rPrime` | computed upstream |
| `nifs.pi_ccs.output.y_ring` | `evaluations` | `Array.ofFn` over every message matrix and all 54 lanes | checked payload |
| `nifs.pi_ccs.output.stage` | `stage` | every output is at `.fresh` | computed |
| `nifs.pi_ccs.output.y_zcol` | excluded | delayed NC projection; never a CE field | checked elsewhere |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputProduct

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

universe uCommitment uChallenge uValue

/-- Exact matrix-indexed CE array carried by one raw output message. `yZcol`
is not read here. -/
def claimedEvaluations
    {shape : SemanticShape}
    (message : OutputMessage shape)
    (source : Fin shape.sourceCount) : Array Phi81Relation.Evaluation :=
  Array.ofFn fun matrix => message.yRing source matrix

@[simp] theorem claimedEvaluations_size
    {shape : SemanticShape}
    (message : OutputMessage shape)
    (source : Fin shape.sourceCount) :
    (claimedEvaluations message source).size = shape.matrixCount := by
  simp [claimedEvaluations]

@[simp] theorem claimedEvaluations_get
    {shape : SemanticShape}
    (message : OutputMessage shape)
    (source : Fin shape.sourceCount)
    (matrix : Fin shape.matrixCount) :
    (claimedEvaluations message source)[matrix.val]'(by
      simpa only [claimedEvaluations, Array.size_ofFn] using matrix.isLt) =
      message.yRing source matrix := by
  simp [claimedEvaluations]

/-- Canonical CE output product. All derivable fields are computed or copied;
the only raw CE payload is the complete `yRing` array. -/
def materialize
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (alignment : SourceAlignment shape params arity)
    (input : SourceProduct shape publicRingColumns publicFits Commitment params arity)
    (rPrime : CubePoint K shape.rowVariables)
    (message : OutputMessage shape) :
    Product shape publicRingColumns publicFits Commitment params arity :=
  fun source => {
    constraintSystem := (input.source source).constraintSystem
    commitment := (input.source source).commitment
    publicInput := (input.source source).publicInput
    point := rPrime
    evaluations :=
      claimedEvaluations message (alignment.semanticIndex source)
    stage := .fresh
  }

@[simp] theorem materialize_constraintSystem
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (alignment : SourceAlignment shape params arity)
    (input : SourceProduct shape publicRingColumns publicFits Commitment params arity)
    (rPrime : CubePoint K shape.rowVariables)
    (message : OutputMessage shape)
    (source : Fin arity.total) :
    (materialize publicRingColumns publicFits alignment input rPrime message
      source).constraintSystem =
      (input.source source).constraintSystem := by
  rfl

@[simp] theorem materialize_commitment
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (alignment : SourceAlignment shape params arity)
    (input : SourceProduct shape publicRingColumns publicFits Commitment params arity)
    (rPrime : CubePoint K shape.rowVariables)
    (message : OutputMessage shape)
    (source : Fin arity.total) :
    (materialize publicRingColumns publicFits alignment input rPrime message
      source).commitment =
      (input.source source).commitment := by
  rfl

@[simp] theorem materialize_publicInput
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (alignment : SourceAlignment shape params arity)
    (input : SourceProduct shape publicRingColumns publicFits Commitment params arity)
    (rPrime : CubePoint K shape.rowVariables)
    (message : OutputMessage shape)
    (source : Fin arity.total) :
    (materialize publicRingColumns publicFits alignment input rPrime message
      source).publicInput =
      (input.source source).publicInput := by
  rfl

@[simp] theorem materialize_point
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (alignment : SourceAlignment shape params arity)
    (input : SourceProduct shape publicRingColumns publicFits Commitment params arity)
    (rPrime : CubePoint K shape.rowVariables)
    (message : OutputMessage shape)
    (source : Fin arity.total) :
    (materialize publicRingColumns publicFits alignment input rPrime message
      source).point =
      rPrime := by
  rfl

@[simp] theorem materialize_evaluations
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (alignment : SourceAlignment shape params arity)
    (input : SourceProduct shape publicRingColumns publicFits Commitment params arity)
    (rPrime : CubePoint K shape.rowVariables)
    (message : OutputMessage shape)
    (source : Fin arity.total) :
    (materialize publicRingColumns publicFits alignment input rPrime message
      source).evaluations =
      claimedEvaluations message (alignment.semanticIndex source) := by
  rfl

@[simp] theorem materialize_stage
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (alignment : SourceAlignment shape params arity)
    (input : SourceProduct shape publicRingColumns publicFits Commitment params arity)
    (rPrime : CubePoint K shape.rowVariables)
    (message : OutputMessage shape)
    (source : Fin arity.total) :
    (materialize publicRingColumns publicFits alignment input rPrime message
      source).stage =
      .fresh := by
  rfl

/-- The canonical product satisfies the public `PiCCS.Shape` obligations once
the input product is known to be fresh. FE/NC messages are parameters only
because the existing shape predicate is a field of a complete `PiCCS.Attempt`;
materialization never reads them. -/
theorem outputProduct_shape
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (alignment : SourceAlignment shape params arity)
    (input : SourceProduct shape publicRingColumns publicFits Commitment params arity)
    (rPrime : CubePoint K shape.rowVariables)
    (message : OutputMessage shape)
    (fe nc : SumCheck.Instance Challenge Value)
    (sourceFresh : forall source, (input.source source).stage = .fresh) :
    PiCCS.Shape
      ({ inputs := input
         outputs :=
           materialize publicRingColumns publicFits alignment input rPrime
             message
         fe := fe
         nc := nc } :
        PiCCS.Attempt
          (Phi81Relation.Structure
            (RelationShape shape publicRingColumns publicFits))
          (Phi81Relation.PublicInput
            (RelationShape shape publicRingColumns publicFits))
          (Phi81Relation.Point
            (RelationShape shape publicRingColumns publicFits))
          Phi81Relation.Evaluation Commitment Challenge Value params arity) := by
  refine {
    sourceFresh := sourceFresh
    outputFresh := ?_
    sameStructure := ?_
    sameCommitment := ?_
    samePublicInput := ?_
    sharedOutputPoint := ?_
  }
  · intro source
    rfl
  · intro source
    rfl
  · intro source
    rfl
  · intro source
    rfl
  · intro left right
    rfl

/-- Field completeness: a candidate CE product agreeing with every owned field
is the canonical product. In particular there is no independent point, stage,
array shape, or hidden `yZcol` field left to choose. -/
theorem outputProduct_unique
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (alignment : SourceAlignment shape params arity)
    (input : SourceProduct shape publicRingColumns publicFits Commitment params arity)
    (rPrime : CubePoint K shape.rowVariables)
    (message : OutputMessage shape)
    (candidate : Product shape publicRingColumns publicFits Commitment params arity)
    (structure_eq : forall source,
      (candidate source).constraintSystem =
        (input.source source).constraintSystem)
    (commitment_eq : forall source,
      (candidate source).commitment = (input.source source).commitment)
    (publicInput_eq : forall source,
      (candidate source).publicInput = (input.source source).publicInput)
    (point_eq : forall source, (candidate source).point = rPrime)
    (evaluations_eq : forall source,
      (candidate source).evaluations =
        claimedEvaluations message (alignment.semanticIndex source))
    (stage_eq : forall source, (candidate source).stage = .fresh) :
    candidate =
      materialize publicRingColumns publicFits alignment input rPrime message := by
  apply funext
  intro source
  have hSystem := structure_eq source
  have hCommitment := commitment_eq source
  have hPublicInput := publicInput_eq source
  have hPoint := point_eq source
  have hEvaluations := evaluations_eq source
  have hStage := stage_eq source
  cases candidateValue : candidate source with
  | mk constraintSystem commitment publicInput point evaluations stage =>
      simp only [candidateValue] at hSystem
      simp only [candidateValue] at hCommitment
      simp only [candidateValue] at hPublicInput
      simp only [candidateValue] at hPoint
      simp only [candidateValue] at hEvaluations
      simp only [candidateValue] at hStage
      simp only [materialize]
      rw [hSystem, hCommitment, hPublicInput, hPoint, hEvaluations, hStage]

/-- `yZcol` is observably absent from CE materialization: two messages with
the same complete `yRing` family produce the same CE product even when their
delayed-NC payloads differ. -/
theorem materialize_eq_of_yRing_eq
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (alignment : SourceAlignment shape params arity)
    (input : SourceProduct shape publicRingColumns publicFits Commitment params arity)
    (rPrime : CubePoint K shape.rowVariables)
    (left right : OutputMessage shape)
    (yRing_eq : left.yRing = right.yRing) :
    materialize publicRingColumns publicFits alignment input rPrime left =
      materialize publicRingColumns publicFits alignment input rPrime right := by
  apply outputProduct_unique publicRingColumns publicFits alignment input
    rPrime right
      (candidate :=
        materialize publicRingColumns publicFits alignment input rPrime left)
  · intro source
    rfl
  · intro source
    rfl
  · intro source
    rfl
  · intro source
    rfl
  · intro source
    simp only [materialize_evaluations]
    unfold claimedEvaluations
    rw [yRing_eq]
  · intro source
    rfl

/-- Exact model-level authority equivalence for a shape-indexed `yRing`
product. The proof intentionally factors through the canonical relation array
using both `evaluationsBound_iff_eq` and `evaluations_get_ofSourceData`. -/
theorem yRingBoundToSources_iff_claimedEvaluationsBound
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (data : Sources.Data shape)
    (points : VerifierPoints shape domain)
    (message : OutputMessage shape) :
    YRingBoundToSources data points message <->
      forall source,
        Phi81Relation.EvaluationsBound
          (Phi81Relation.Structure.ofSourceData
            publicRingColumns publicFits data)
          (data.assignment source) points.rPrime
          (claimedEvaluations message source) := by
  constructor
  · intro bound source
    apply (Phi81Relation.evaluationsBound_iff_eq _ _ _ _).2
    apply Array.ext
    · simp [claimedEvaluations, Phi81Relation.evaluations,
        Phi81Relation.Shape.ofSemantic]
    · intro index claimedLt canonicalLt
      have indexLt : index < shape.matrixCount := by
        simpa only [Phi81Relation.evaluations, Array.size_ofFn] using canonicalLt
      let matrix : Fin shape.matrixCount := ⟨index, indexLt⟩
      funext lane
      calc
        ((claimedEvaluations message source)[index]'claimedLt) lane =
            message.yRing source matrix lane := by
          simp [claimedEvaluations, matrix]
        _ = canonicalYRing data points source matrix lane :=
          bound source matrix lane
        _ =
            ((Phi81Relation.evaluations
              (Phi81Relation.Structure.ofSourceData
                publicRingColumns publicFits data)
              (data.assignment source) points.rPrime)[index]'canonicalLt) lane := by
          symm
          exact congrFun
            (Phi81Relation.evaluations_get_ofSourceData
              publicRingColumns publicFits data points source matrix) lane
  · intro allBound source matrix lane
    calc
      message.yRing source matrix lane =
          ((claimedEvaluations message source)[matrix.val]'(by
            simpa only [claimedEvaluations, Array.size_ofFn] using matrix.isLt)) lane := by
        simp [claimedEvaluations]
      _ = Phi81Relation.matrixEvaluation
            (Phi81Relation.Structure.ofSourceData
              publicRingColumns publicFits data)
            (data.assignment source) points.rPrime matrix lane :=
        (allBound source).lane_eq matrix lane
      _ = canonicalYRing data points source matrix lane :=
        Phi81Relation.matrixEvaluation_apply_ofSourceData
          publicRingColumns publicFits data points source matrix lane

/-- Exact authority equivalence for the actual protocol-indexed CE product.
The reverse direction uses the proved partition-preserving index inverse, so
checking every output is checking every semantic source exactly once. -/
theorem yRingBoundToSources_iff_outputEvaluationsBound
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (data : Sources.Data shape)
    (alignment : SourceAlignment shape params arity)
    (input : SourceProduct shape publicRingColumns publicFits Commitment params arity)
    (points : VerifierPoints shape domain)
    (message : OutputMessage shape) :
    YRingBoundToSources data points message <->
      forall source,
        Phi81Relation.EvaluationsBound
          (Phi81Relation.Structure.ofSourceData
            publicRingColumns publicFits data)
          (data.assignment (alignment.semanticIndex source)) points.rPrime
          (materialize publicRingColumns publicFits alignment input
            points.rPrime message source).evaluations := by
  rw [yRingBoundToSources_iff_claimedEvaluationsBound
    publicRingColumns publicFits data points message]
  constructor
  · intro bound source
    simpa only [materialize_evaluations] using
      bound (alignment.semanticIndex source)
  · intro bound source
    simpa only [materialize_evaluations,
      SourceAlignment.semanticIndex_productIndex] using
      bound (alignment.productIndex source)

/-- Exact CE-evaluation authority stated only with the row-point equality
actually consumed by output materialization. The delayed-NC column or block
point is intentionally absent. -/
theorem yRing_eq_sourceYRingAt_iff_outputEvaluationsBound
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (data : Sources.Data shape)
    (alignment : SourceAlignment shape params arity)
    (input :
      SourceProduct shape publicRingColumns publicFits Commitment params arity)
    (rPrime : CubePoint K shape.rowVariables)
    (message : OutputMessage shape) :
    message.yRing = Polynomial.Fe.sourceYRingAt data rPrime <->
      forall source,
        Phi81Relation.EvaluationsBound
          (Phi81Relation.Structure.ofSourceData
            publicRingColumns publicFits data)
          (data.assignment (alignment.semanticIndex source)) rPrime
          (materialize publicRingColumns publicFits alignment input
            rPrime message source).evaluations := by
  let inertDomain : FlatNcDomain := {
    columnVariables := 0
    laneVariables := 0
  }
  let points : VerifierPoints shape inertDomain := {
    rPrime := rPrime
    sPrime := {
      coordinates := []
      dimension := rfl
    }
  }
  have authority :=
    yRingBoundToSources_iff_outputEvaluationsBound
      publicRingColumns publicFits data alignment input points message
  constructor
  · intro equal
    apply authority.mp
    intro source matrix lane
    have coordinate := congrFun (congrFun (congrFun equal source) matrix) lane
    simpa [points, canonicalYRing, Polynomial.Fe.sourceYRingAt] using coordinate
  · intro evaluationsBound
    have bound : YRingBoundToSources data points message :=
      authority.mpr evaluationsBound
    funext source matrix lane
    simpa [points, canonicalYRing, Polynomial.Fe.sourceYRingAt] using
      bound source matrix lane

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputProduct
