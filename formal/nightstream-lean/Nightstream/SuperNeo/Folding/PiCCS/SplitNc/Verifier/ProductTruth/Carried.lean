import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.ProductTruth

/-!
Carried-claim half of concrete Split-NC payload truth transport.

Assurance tier: model-level.

The proof reads the adversarial public running evaluation array at every
matrix and Phi81 lane and transports that exact value to the independent
carried-evaluation equation. It does not replace a public claim with an
honestly recomputed claim.

Owns: exact transport of public running evaluation claims to carried truth.

Does not own: honest replacement of carried claims, transcript acceptance,
commitments, extraction, probability, Rust, R1CS, costs, or rows.

Emits constraints: no.

| Stage path | Owned equation | Authority |
|---|---|---|
| `piccs.split_nc.product_truth.carried` | public claimed coefficient equals the corresponding computed coefficient | derived from payload binding |
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.ProductTruth

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uCommitment

/-- One adversarial running payload equation transports through the public
source binding without changing its claimed evaluation array. -/
theorem runningEvaluationsEqual_of_payloads
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (commit : SourceAssignment shape -> Commitment)
    (data : Data shape)
    (alignment : SourceAlignment shape params arity)
    (input :
      SourceProduct shape publicRingColumns publicFits Commitment params arity)
    (inputAuthority :
      InputAuthority.BoundToSources publicRingColumns publicFits commit data
        alignment input)
    (payloads :
      PayloadsHold publicRingColumns publicFits commit data alignment input)
    (running : Fin shape.runningCount) :
    Phi81Relation.evaluations
        (Phi81Relation.Structure.ofSourceData
          publicRingColumns publicFits data)
        (data.runningAssignments running) data.priorPoint =
      InputAuthority.priorEvaluations data running := by
  let productRunning :=
    alignment.productRunningIndex running
  have payload :=
    payloads (Fin.natAdd arity.freshCount productRunning)
  simp only [PiCCS.InputProduct.source, PiCCS.Source.PayloadTruth,
    InputAuthority.productAssignments_running] at payload
  simp [productRunning] at payload
  have evaluationsEqual := payload.2
  change
    Phi81Relation.evaluations
        (input.running productRunning).constraintSystem
        (data.runningAssignments
          (alignment.semanticRunningIndex productRunning))
        (input.running productRunning).point =
      (input.running productRunning).evaluations at evaluationsEqual
  rw [(inputAuthority.running productRunning).constraintSystem,
    (inputAuthority.running productRunning).point,
    (inputAuthority.running productRunning).evaluations] at evaluationsEqual
  simpa only [SourceAlignment.semanticRunningIndex_productRunningIndex,
    productRunning] using evaluationsEqual

/-- Concrete running-source payloads imply the independent carried-claim
statement without replacing the public claims by honest recomputation. -/
theorem carriedTruth_of_payloads
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (commit : SourceAssignment shape -> Commitment)
    (data : Data shape)
    (alignment : SourceAlignment shape params arity)
    (input :
      SourceProduct shape publicRingColumns publicFits Commitment params arity)
    (inputAuthority :
      InputAuthority.BoundToSources publicRingColumns publicFits commit data
        alignment input)
    (payloads :
      PayloadsHold publicRingColumns publicFits commit data alignment input) :
    SplitNc.Semantics.Fe.CarriedTruth data := by
  rintro ⟨running, matrix, coefficient⟩
  have evaluationsEqual' :
      Phi81Relation.evaluations
          (Phi81Relation.Structure.ofSourceData
            publicRingColumns publicFits data)
          (data.runningAssignments running) data.priorPoint =
        InputAuthority.priorEvaluations data running :=
    runningEvaluationsEqual_of_payloads publicRingColumns publicFits commit
      data alignment input inputAuthority payloads running
  have atMatrix := congrArg
    (fun values =>
      values.getD matrix.val (fun _ => K.zero))
    evaluationsEqual'
  have atLane := congrFun atMatrix coefficient
  have evaluationAtLane := congrFun
    (Phi81Relation.evaluations_get_ofSourceData_atRow
      publicRingColumns publicFits data data.priorPoint
      (Data.runningIndex running) matrix)
    coefficient
  rw [data.assignment_runningIndex] at evaluationAtLane
  have evaluationIndexLt :
      matrix.val <
        (Phi81Relation.evaluations
          (Phi81Relation.Structure.ofSourceData
            publicRingColumns publicFits data)
          (data.runningAssignments running) data.priorPoint).size := by
    simpa only [Phi81Relation.evaluations, Array.size_ofFn] using matrix.isLt
  have evaluationGetD :
      (Phi81Relation.evaluations
          (Phi81Relation.Structure.ofSourceData
            publicRingColumns publicFits data)
          (data.runningAssignments running) data.priorPoint).getD
            matrix.val (fun _ => K.zero) coefficient =
        OutputClaims.yRingForAssignment data
          (data.runningAssignments running) data.priorPoint
          matrix coefficient := by
    rw [← Array.getElem_eq_getD (h := evaluationIndexLt)
      (fun _ => K.zero)]
    exact evaluationAtLane
  have priorAtLane := congrFun
    (InputAuthority.priorEvaluations_get data running matrix)
    coefficient
  have priorIndexLt :
      matrix.val <
        (InputAuthority.priorEvaluations data running).size := by
    simpa only [InputAuthority.priorEvaluations, Array.size_ofFn] using
      matrix.isLt
  have priorGetD :
      (InputAuthority.priorEvaluations data running).getD
            matrix.val (fun _ => K.zero) coefficient =
        data.claimedCoefficient
          { running := running, matrix := matrix,
            coefficient := coefficient } := by
    rw [← Array.getElem_eq_getD (h := priorIndexLt) (fun _ => K.zero)]
    rw [PublicInput.ofSources_claimedYRing] at priorAtLane
    exact priorAtLane
  unfold CarriedEvaluationResidual.EvaluationClaimHolds
  symm
  calc
    CarriedEvaluationResidual.computedCoefficient
        ConcreteCarrier.baseOps ConcreteCarrier.extensionOps K.embed
        data.carriedData
        { running := running, matrix := matrix, coefficient := coefficient } =
        Polynomial.Fe.sourceYRingAt data data.priorPoint
          (Data.runningIndex running) matrix coefficient :=
      (Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.SourceRefinement.sourceYRingAt_running_eq_computedCoefficient
          data running matrix coefficient).symm
    _ =
        OutputClaims.yRingForAssignment data
          (data.runningAssignments running) data.priorPoint
          matrix coefficient := by
      rw [Polynomial.Fe.sourceYRingAt, data.assignment_runningIndex]
    _ =
        (Phi81Relation.evaluations
            (Phi81Relation.Structure.ofSourceData
              publicRingColumns publicFits data)
            (data.runningAssignments running)
            data.priorPoint).getD matrix.val (fun _ => K.zero)
          coefficient :=
      evaluationGetD.symm
    _ =
        (InputAuthority.priorEvaluations data running).getD
          matrix.val (fun _ => K.zero) coefficient :=
      atLane
    _ = data.claimedCoefficient
        { running := running, matrix := matrix, coefficient := coefficient } :=
      priorGetD

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.ProductTruth
