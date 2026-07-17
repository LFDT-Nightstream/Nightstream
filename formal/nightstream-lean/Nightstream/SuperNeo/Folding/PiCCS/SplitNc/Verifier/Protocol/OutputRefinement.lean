import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.InputAuthority
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputProduct
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.HonestProver

/-!
Protocol-level refinement from accepted, source-authorized Split-NC `Pi_CCS`
to the canonical CE product consumed by `Pi_RLC`.

Protocol: SuperNeo `Pi_CCS`.
Phase: exact FE/NC acceptance and public output handoff.
Constraint family: transcript acceptance, `yRing`/`yZcol` output authority,
and CE opening membership; this file emits no rows.

Owns: proof that a paper-valid, input-bound execution with source-bound
`yRing` materializes genuine CE statements; and deterministic composition of
separate transcript acceptance and full output authority into either that
complete CE product or a named FE/NC bad event.

Does not own: transcript coin derivation, Poseidon2 refinement, PiRLC
challenges or algebra, PiDEC, Fiat--Shamir probability bounds, Rust, R1CS,
rows, costs, or row removal.

Emits constraints: no.

Authority boundary: transcript acceptance does not authorize commitments,
public inputs, `yRing`, or `yZcol`. Input authority is supplied separately by
`InputAuthority.BoundToSources`. CE materialization requires only
`YRingBoundToSources`; the delayed-NC `yZcol` branch remains a separate NIFS
security boundary. This module deliberately defines no second `Accepted`
wrapper that could be mistaken for executable verifier acceptance.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.handoff.acceptance` | exact FE then NC transcript accepts | physical premise | `Protocol.Accepted` argument |
| `nifs.pi_ccs.handoff.output_authority.y_ring` | complete `yRing` binds to the same sources | semantic premise | `YRingBoundToSources` argument |
| `nifs.pi_ccs.handoff.output_authority.y_zcol` | delayed-NC sidecar authority is excluded from CE materialization | separate security boundary | not consumed by `materializedOutputsHold_of_yRingEq` |
| `nifs.pi_ccs.handoff.opening` | output commitment/public input/norm come from the input-bound assignment | derived | `materializedOutputsHold_of_yRingEq` |
| `nifs.pi_ccs.handoff.evaluations` | every matrix and every Phi81 lane equals the source-derived evaluation | derived | `materializedOutputsHold_of_yRingEq` |
| `nifs.pi_ccs.handoff.soundness` | separate physical acceptance and semantic authority yield the complete CE product or a named bad event | derived | `accepted_and_outputBound_implies_outputsHold_or_badEvent` |
| `nifs.pi_ccs.handoff.completeness` | every paper-valid authorized input has an accepted certificate and complete CE product | derived | `complete_of_paperObligations` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.OutputRefinement

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uCommitment uState

/-- The canonical materialized output is genuine CE membership once its
`yRing` family equals the independent source evaluator at the sole row point
consumed by CE. No delayed-NC point or `yZcol` premise occurs here. -/
theorem materializedOutputsHold_of_yRingEq
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
    (rPrime : CubePoint K shape.rowVariables)
    (message : OutputMessage shape)
    (freshBound_eq_two : NormStage.bound params .fresh = 2)
    (paper : SplitNc.Semantics.Paper.Holds data)
    (inputBound :
      InputAuthority.BoundToSources publicRingColumns publicFits commit data
        alignment input)
    (yRingEq : message.yRing = Polynomial.Fe.sourceYRingAt data rPrime) :
    ProductHolds publicRingColumns publicFits commit
      (OutputProduct.materialize publicRingColumns publicFits alignment input
        rPrime message)
      (InputAuthority.productAssignments data alignment) := by
  unfold ProductHolds
  intro source
  let output :=
    OutputProduct.materialize publicRingColumns publicFits alignment input
      rPrime message source
  refine ⟨⟨?_, ?_, ?_⟩, ?_, ?_⟩
  · change
      commit (data.assignment (alignment.semanticIndex source)) =
        output.commitment
    rw [show output.commitment = (input.source source).commitment by rfl]
    exact InputAuthority.BoundToSources.sourceCommitment
      publicRingColumns publicFits commit data alignment input inputBound source
  · change
      sourcePublicInput publicRingColumns publicFits
          (data.assignment (alignment.semanticIndex source)) =
        output.publicInput
    rw [show output.publicInput = (input.source source).publicInput by rfl]
    exact InputAuthority.BoundToSources.sourcePublicInput
      publicRingColumns publicFits commit data alignment input inputBound source
  · change ∀ column,
      centeredMagnitude
          (data.assignment (alignment.semanticIndex source) column) <
        NormStage.bound params output.stage
    rw [show output.stage = NormStage.fresh by rfl]
    exact InputAuthority.productAssignments_normFresh data alignment
      freshBound_eq_two paper source
  · exact output.point.dimension
  · have evaluationsBound :=
      (OutputProduct.yRing_eq_sourceYRingAt_iff_outputEvaluationsBound
        publicRingColumns publicFits data alignment input rPrime message).mp
        yRingEq source
    have outputStructure :
        output.constraintSystem =
          Phi81Relation.Structure.ofSourceData
            publicRingColumns publicFits data := by
      simpa [output, OutputProduct.materialize] using
        InputAuthority.BoundToSources.sourceStructure
          publicRingColumns publicFits commit data alignment input inputBound
          source
    have outputPoint : output.point = rPrime := by
      rfl
    have evaluationsEq :
        output.evaluations =
          Phi81Relation.evaluations
            (Phi81Relation.Structure.ofSourceData
              publicRingColumns publicFits data)
            (data.assignment (alignment.semanticIndex source))
            rPrime :=
      (Phi81Relation.evaluationsBound_iff_eq
        (Phi81Relation.Structure.ofSourceData
          publicRingColumns publicFits data)
        (data.assignment (alignment.semanticIndex source))
        rPrime output.evaluations).mp evaluationsBound
    rw [outputStructure, outputPoint]
    exact evaluationsEq.symm

/-- Flat-point adapter retained until the active concrete NIFS transition is
cut over to canonical block×lane NC. It adds no column-point premise to the
shared CE handoff. -/
theorem materializedOutputsHold_of_yRingBound
    {shape : SemanticShape}
    {domain : FlatNcDomain}
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
    (points : VerifierPoints shape domain)
    (message : OutputMessage shape)
    (freshBound_eq_two : NormStage.bound params .fresh = 2)
    (paper : SplitNc.Semantics.Paper.Holds data)
    (inputBound :
      InputAuthority.BoundToSources publicRingColumns publicFits commit data
        alignment input)
    (yRingBound : YRingBoundToSources data points message) :
    ProductHolds publicRingColumns publicFits commit
      (OutputProduct.materialize publicRingColumns publicFits alignment input
        points.rPrime message)
      (InputAuthority.productAssignments data alignment) := by
  apply materializedOutputsHold_of_yRingEq publicRingColumns publicFits commit
    data alignment input points.rPrime message freshBound_eq_two paper
    inputBound
  funext source matrix lane
  simpa [canonicalYRing, Polynomial.Fe.sourceYRingAt] using
    yRingBound source matrix lane

/-- Deterministic protocol soundness after the separate output-authority
check: the handoff is a complete CE product or one named FE/NC bad event. -/
theorem accepted_and_outputBound_implies_outputsHold_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (covers : domain.Covers shape)
    (feMachine : Transcript.Fe.Machine State)
    (ncMachine : Transcript.Nc.Machine State)
    (initialState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (data : Data shape)
    (feCoins : Polynomial.Fe.Coins shape domain)
    (ncCoins : Polynomial.Nc.Mixing.Coins domain)
    (certificate :
      Protocol.Certificate (PublicInput.ofSources data) domain)
    (challengeSetSize : Nat)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (commit : SourceAssignment shape -> Commitment)
    (alignment : SourceAlignment shape params arity)
    (input :
      SourceProduct shape publicRingColumns publicFits Commitment params arity)
    (freshBound_eq_two : NormStage.bound params .fresh = 2)
    (inputBound :
      InputAuthority.BoundToSources publicRingColumns publicFits commit data
        alignment input)
    (accepted :
      Protocol.Accepted feMachine ncMachine initialState profile
        (PublicInput.ofSources data) feCoins ncCoins certificate)
    (outputBound :
      OutputClaims.BoundToSources covers data
        (Protocol.derive feMachine ncMachine initialState
          certificate).outputPoints
        certificate.output) :
    let execution :=
      Protocol.derive feMachine ncMachine initialState certificate
    ProductHolds publicRingColumns publicFits commit
      (OutputProduct.materialize publicRingColumns publicFits alignment input
        execution.outputPoints.rPrime certificate.output)
      (InputAuthority.productAssignments data alignment) ∨
      Protocol.BadEvent profile covers data feCoins ncCoins execution
        certificate challengeSetSize := by
  let execution :=
    Protocol.derive feMachine ncMachine initialState certificate
  change
    ProductHolds publicRingColumns publicFits commit
      (OutputProduct.materialize publicRingColumns publicFits alignment input
        execution.outputPoints.rPrime certificate.output)
      (InputAuthority.productAssignments data alignment) ∨
      Protocol.BadEvent profile covers data feCoins ncCoins execution
        certificate challengeSetSize
  have phaseSoundness :=
    Protocol.accepted_implies_paperObligations_or_unbound_or_badEvent
      noZeroDivisors covers feMachine ncMachine initialState profile data
      feCoins ncCoins certificate challengeSetSize accepted
  change
    SplitNc.Semantics.Paper.Holds data ∨
      ¬ OutputClaims.BoundToSources covers data execution.outputPoints
          certificate.output ∨
      Protocol.BadEvent profile covers data feCoins ncCoins execution
        certificate challengeSetSize at phaseSoundness
  rcases phaseSoundness with paper | unbound | bad
  · exact Or.inl <|
      materializedOutputsHold_of_yRingBound publicRingColumns publicFits commit data
        alignment input execution.outputPoints certificate.output
        freshBound_eq_two paper inputBound outputBound.yRing
  · exact False.elim (unbound outputBound)
  · exact Or.inr bad

/-- Honest completeness at the same public authority boundary: an independent
paper-valid source family with correctly bound public inputs has a physical
sequential certificate whose canonical materialization is a complete CE
product. The explicit coin records remain verifier inputs until the concrete
Poseidon2 transcript schedule is composed. -/
theorem complete_of_paperObligations
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (covers : domain.Covers shape)
    (feMachine : Transcript.Fe.Machine State)
    (ncMachine : Transcript.Nc.Machine State)
    (initialState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (data : Data shape)
    (feCoins : Polynomial.Fe.Coins shape domain)
    (ncCoins : Polynomial.Nc.Mixing.Coins domain)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (commit : SourceAssignment shape -> Commitment)
    (alignment : SourceAlignment shape params arity)
    (input :
      SourceProduct shape publicRingColumns publicFits Commitment params arity)
    (freshBound_eq_two : NormStage.bound params .fresh = 2)
    (paper : SplitNc.Semantics.Paper.Holds data)
    (inputBound :
      InputAuthority.BoundToSources publicRingColumns publicFits commit data
        alignment input) :
    ∃ certificate :
        Protocol.Certificate (PublicInput.ofSources data) domain,
      Protocol.Accepted feMachine ncMachine initialState profile
          (PublicInput.ofSources data) feCoins ncCoins certificate ∧
        OutputClaims.BoundToSources covers data
          (Protocol.derive feMachine ncMachine initialState
            certificate).outputPoints
          certificate.output ∧
        ProductHolds publicRingColumns publicFits commit
          (OutputProduct.materialize publicRingColumns publicFits alignment input
            ((Protocol.derive feMachine ncMachine initialState certificate).outputPoints.rPrime)
            certificate.output)
          (InputAuthority.productAssignments data alignment) := by
  rcases Protocol.HonestProver.complete_of_paperObligations
      covers feMachine ncMachine initialState profile data feCoins ncCoins
      paper with
    ⟨certificate, transcriptAccepted, outputAuthority⟩
  refine ⟨certificate, transcriptAccepted, outputAuthority, ?_⟩
  exact materializedOutputsHold_of_yRingBound publicRingColumns publicFits commit data
    alignment input
    (Protocol.derive feMachine ncMachine initialState certificate).outputPoints
    certificate.output freshBound_eq_two paper inputBound outputAuthority.yRing

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.OutputRefinement
