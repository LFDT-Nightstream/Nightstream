import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.SemanticAttempt

/-!
Assignment-indexed semantic view of the actual production Split-NC prefix.

Assurance tier: model-level registered-deviation refinement.

Owns: a generic `PiCCS.Attempt` whose FE view is reconstructed from the FE
polynomial and whose NC view is reconstructed independently from the actual
base-or-delayed production polynomial. The delayed view consumes the registered
full-vector pending-state authority.

Does not own: probability, Fiat--Shamir, commitment security, extraction,
Rust, R1CS, costs, or rows.

Emits constraints: no.

| Stage path | Owned equation | Authority |
|---|---|---|
| `fprime.piccs.production.semantic_attempt` | pending tag selects the exact ordinary or delayed NC semantic polynomial | transcript-derived |
-/

set_option autoImplicit false
set_option maxRecDepth 2048

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.SemanticAttempt

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane
open Nightstream.SuperNeo.SumCheck.Finite

universe uState

private abbrev ops := ConcreteCarrier.extensionOps
private abbrev laws := ConcreteCarrier.extensionLaws

variable
  {shape : SemanticShape}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-- The actual production NC semantic view. Its polynomial is selected by the
transcript-bound pending-state tag; it is not the ordinary NC polynomial in
the delayed branch. -/
def ncInstance
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (certificate : Certificate input) :
    Nightstream.SuperNeo.SumCheck.Instance K K :=
  FixedPhase.symbolicInstance ops.toOps
    (ProductionPiCcs.rawPolynomial input.full input.data)
    Polynomial.Nc.Degree.ncSumcheckDegreeBound input.full.challengeSetSize
    (ProductionPiCcs.rawInitial input.full)
    (ProductionPiCcs.ncPoint input.full certificate.materialize).coordinates
    certificate.nc.toSumCheck

/-- One proof-only attempt over the exact production certificate. FE and NC
share the physical schedule but no semantic polynomial or truth equation. -/
def attempt
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (certificate : Certificate input) :=
  { (Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.SemanticAttempt.attempt
      input.full.covers StatementInput.polynomial input.full.piCcsSchedule
      input.full.priorState input.full.profile input.full.piCcsStatement
      input.data input.publicInput_eq_sources certificate.materialize.piCcs
      input.full.challengeSetSize publicRingColumns publicFits
      input.full.alignment input.full.input) with
    nc := ncInstance input certificate }

private theorem feAccepted_atSources
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (certificate : Certificate input)
    (accepted : ProductionPiCcs.Accepted input.full input.data
      certificate.materialize) :
    SumCheck.Fe.Accepted
      (Polynomial.Fe.initial input.full.profile
        (PublicInput.ofSources input.data) input.full.feCoins)
      (Polynomial.Fe.terminalFromMessage input.full.profile
        (PublicInput.ofSources input.data) input.full.feCoins
        (ProductionPiCcs.fePoint input.full certificate.materialize)
        certificate.output)
      (ProductionPiCcs.fePoint input.full certificate.materialize)
      (Protocol.BlockLane.certificateAtSources input.data
        certificate.materialize.piCcs input.publicInput_eq_sources).fe := by
  simpa [ProductionPiCcs.fePoint] using accepted.fe

/-- Exact message acceptance plus verifier-owned packed output materialization
transports to generic acceptance of the two independently reconstructed
semantic instances. -/
theorem accepted
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (certificate : Certificate input)
    (accepted : ProductionVerifierAccepts input certificate) :
    PiCCS.Accepted ops.toOps.toSymbolic (attempt input certificate) := by
  have raw : ProductionPiCcs.Accepted input.full input.data
      certificate.materialize :=
    ProductionPiCcs.accepted_of_messageAccepted_and_packed input.full
      input.data certificate.materialize accepted certificate.output_bound.2
  have sourceFe := feAccepted_atSources input certificate raw
  refine ⟨?_, ?_, ?_⟩
  · exact OutputProduct.outputProduct_shape publicRingColumns publicFits
      input.full.alignment input.full.input
      (ProductionPiCcs.fePoint input.full certificate.materialize).row
      certificate.output
      (SumCheck.SemanticAdapter.feInstance input.full.profile input.data
        input.full.feCoins
        (ProductionPiCcs.fePoint input.full certificate.materialize)
        certificate.output
        (Protocol.BlockLane.certificateAtSources input.data
          certificate.materialize.piCcs input.publicInput_eq_sources).fe
        input.full.challengeSetSize)
      (ncInstance input certificate)
      (InputAuthority.BoundToSources.sourceFresh publicRingColumns publicFits
        (commit input.full.key) input.data input.full.alignment input.full.input
        input.sourceProduct_bound)
  · exact SumCheck.SemanticAdapter.feAccepted_implies_genericAccepted
      input.full.profile input.data input.full.feCoins
      (ProductionPiCcs.fePoint input.full certificate.materialize)
      certificate.output
      (Protocol.BlockLane.certificateAtSources input.data
        certificate.materialize.piCcs input.publicInput_eq_sources).fe
      input.full.challengeSetSize sourceFe certificate.output_bound.1
  · exact FixedPhase.accepted_implies_symbolicAccepted ops.toOps
      (ProductionPiCcs.rawPolynomial input.full input.data)
      input.full.challengeSetSize (ProductionPiCcs.rawInitial input.full)
      (ProductionPiCcs.ncPoint input.full certificate.materialize).coordinates
      certificate.nc.toSumCheck raw.nc

private theorem rawInitial_eq_semanticInitial
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (certificate : Certificate input)
    (paper : Semantics.Paper.Holds input.data)
    (pendingBound : ProductionPiCcs.PendingBound input.full input.data) :
    ProductionPiCcs.rawInitial input.full =
      FixedPhase.semanticInitial ops.toOps
        (ProductionPiCcs.rawPolynomial input.full input.data)
        (ProductionPiCcs.ncPoint input.full
          certificate.materialize).coordinates.length := by
  cases pendingEq : input.full.pending with
  | none =>
      simp only [ProductionPiCcs.rawInitial, ProductionPiCcs.rawPolynomial,
        pendingEq]
      rw [Polynomial.Nc.BlockLane.InitialSum.claimedInitial_eq_sumcheckHypercubeSum_of_truth
        input.full.covers input.data input.full.ncCoins paper.2.1]
      unfold Polynomial.Nc.BlockLane.InitialSum.sumcheckHypercubeSum
        FixedPhase.semanticInitial
      rw [(ProductionPiCcs.ncPoint input.full
        certificate.materialize).coordinates_length]
  | some pending =>
      simp only [ProductionPiCcs.rawInitial, ProductionPiCcs.rawPolynomial,
        pendingEq]
      rw [Polynomial.Nc.BlockLane.DelayedCombinedNc.Acceptance.semanticInitial_eq_ordinary_add_weightedProjection
        input.full.covers input.data input.full.ncCoins
        (ProductionProjection.productionWeights input.full)
        input.full.producerBeta input.full.batchWeight pending.oldBlock
        (ProductionPiCcs.ncPoint input.full certificate.materialize)]
      rw [Polynomial.Nc.BlockLane.InitialSum.mixedResidualAtBeta_eq_zero_of_truth
        input.full.covers input.data input.full.ncCoins paper.2.1]
      change
        ops.mul input.full.batchWeight
            (DelayedPackedProjection.projectedValue pending.parentYZcol
              input.full.producerBeta) =
          ops.add ops.zero
            (ops.mul input.full.batchWeight
              (Polynomial.Nc.BlockLane.DelayedCombinedNc.authoritativeRunningProjection
                input.full.covers input.data
                (ProductionProjection.productionWeights input.full)
                input.full.producerBeta pending.oldBlock))
      rw [laws.zero_add]
      rw [Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionPiCcs.HonestProver.parentProjectionBound
        input.full input.data pending pendingEq pendingBound]

/-- The production attempt has a concrete assignment-indexed
arithmetization once the paper branch and the registered delayed-state
authority have been established. No rewind callback or semantic ghost is a
premise. -/
theorem arithmetization
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (certificate : Certificate input)
    (accepted : ProductionVerifierAccepts input certificate)
    (paper : Semantics.Paper.Holds input.data)
    (pendingBound : ProductionPiCcs.PendingBound input.full input.data) :
    PiCCS.Arithmetization (semantics input.full.key) productionGlobalParams
      ops.toOps.toSymbolic (attempt input certificate)
      (InputAuthority.productAssignments input.data input.full.alignment) := by
  have raw : ProductionPiCcs.Accepted input.full input.data
      certificate.materialize :=
    ProductionPiCcs.accepted_of_messageAccepted_and_packed input.full
      input.data certificate.materialize accepted certificate.output_bound.2
  have sourceFe := feAccepted_atSources input certificate raw
  refine {
    feTruthPath := ?_
    ncTruthPath := ?_
    feClaimTrue_of_payloads := ?_
    ncClaimTrue_of_norms := ?_
  }
  · exact SumCheck.SemanticAdapter.feAccepted_implies_truthPath
      input.full.profile input.data input.full.feCoins
      (ProductionPiCcs.fePoint input.full certificate.materialize)
      certificate.output
      (Protocol.BlockLane.certificateAtSources input.data
        certificate.materialize.piCcs input.publicInput_eq_sources).fe
      input.full.challengeSetSize sourceFe certificate.output_bound.1
  · exact FixedPhase.symbolicTruthPath ops.toOps
      (ProductionPiCcs.rawPolynomial input.full input.data)
      input.full.challengeSetSize (ProductionPiCcs.rawInitial input.full)
      (ProductionPiCcs.ncPoint input.full certificate.materialize).coordinates
      certificate.nc.toSumCheck raw.nc
  · intro payloads
    change
      ProductTruth.PayloadsHold publicRingColumns publicFits
        (commit input.full.key) input.data input.full.alignment input.full.input
        at payloads
    have feTruth : Semantics.Fe.Truth input.data := ⟨
      ProductTruth.freshTruth_of_payloads publicRingColumns publicFits
        (commit input.full.key) input.data input.full.alignment input.full.input
        input.sourceProduct_bound payloads,
      ProductTruth.carriedTruth_of_payloads publicRingColumns publicFits
        (commit input.full.key) input.data input.full.alignment input.full.input
        input.sourceProduct_bound payloads⟩
    exact SumCheck.SemanticAdapter.feClaimTrue_of_truth input.full.profile
      input.data input.full.feCoins
      (ProductionPiCcs.fePoint input.full certificate.materialize)
      certificate.output
      (Protocol.BlockLane.certificateAtSources input.data
        certificate.materialize.piCcs input.publicInput_eq_sources).fe
      input.full.challengeSetSize feTruth
  · intro _
    change ProductionPiCcs.rawInitial input.full =
      FixedPhase.semanticInitial ops.toOps
        (ProductionPiCcs.rawPolynomial input.full input.data)
        (ProductionPiCcs.ncPoint input.full
          certificate.materialize).coordinates.length
    exact rawInitial_eq_semanticInitial input certificate paper pendingBound

end Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.SemanticAttempt
