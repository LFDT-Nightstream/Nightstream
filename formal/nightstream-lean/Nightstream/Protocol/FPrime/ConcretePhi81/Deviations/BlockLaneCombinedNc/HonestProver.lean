import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionPiCcs
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.CanonicalOpening.SourceInput
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Fe.HonestProver
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.BlockLane.DelayedHonestProver
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.HonestProver

/-!
Honest construction for the production two-SumCheck Split-NC prefix.

Assurance tier: model-level registered-deviation refinement.

Owns: authoritative opening-derived public input, causal FE construction, the
exact FE successor entering NC, base or delayed combined block/lane NC
construction, source-derived `yRing`/`yZcol`, and claims-level acceptance.

Does not own: PiRLC sampling, PiDEC payload construction, terminal lifecycle
closure, concrete Poseidon2/Ajtai internals, Rust/R1CS, costs, or rows.

Emits constraints: none.

Authority boundary: `PendingBound` binds the complete pending 54-lane vector
to the radix recomposition of the current authoritative raw running
assignments. It is not a digest, output-sidecar copy, or paper conclusion.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `pi_ccs.production.pending` | bind the complete delayed vector to raw-running recomposition | checked | `PendingBound`, `parentProjectionBound` |
| `pi_ccs.production.fe` | construct the causal FE certificate from paper FE truth | computed | `complete_of_paper` |
| `pi_ccs.production.nc` | construct ordinary or delayed block/lane NC from paper NC truth | computed | `complete_of_paper` |
| `pi_ccs.production.output` | compute and bind both output products at verifier-derived points | computed/derived | `canonicalOutput`, `canonicalOutput_bound` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionPiCcs.HonestProver

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane
open Nightstream.SuperNeo.SumCheck.Finite

private abbrev ops := ConcreteCarrier.extensionOps

universe uState

variable
  {shape : SemanticShape}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Honest authority for the delayed input owned by this iteration. At the base
boundary there is no predecessor. At a recursive boundary the complete pending
vector is recomputed from all fourteen authoritative running assignments in
semantic order. -/
def PendingBound
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape) : Prop :=
  match context.pending with
  | none => True
  | some pending =>
      pending.parentYZcol =
        PackedBlockAction.packedYZcol context.covers
          (PiDEC.Raw.recomposeAssignment
            (DelayedRawChildren.rawRunningAssignments context data))
          pending.oldBlock

/-- The production output is computed only after both transcript points exist.
Every coordinate is read from the authoritative source table. -/
def canonicalOutput
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (row : CubePoint K shape.rowVariables)
    (block : CubePoint K PiCcsDomains.production.nc.blockVariables) :
    OutputMessage shape :=
  Verifier.Protocol.BlockLane.HonestProver.canonicalOutput context.covers data
    row block

/-- Canonical output binds both independently materialized products by
computation. -/
theorem canonicalOutput_bound
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (row : CubePoint K shape.rowVariables)
    (block : CubePoint K PiCcsDomains.production.nc.blockVariables) :
    (canonicalOutput context data row block).yRing =
        Polynomial.Fe.sourceYRingAt data row ∧
      Terminal.PackedYZcolBoundAtBlock context.covers data block
        (canonicalOutput context data row block) := by
  constructor
  · rfl
  · intro source lane
    rfl

/-- Full-vector pending authority implies the exact scalar identity needed by
the delayed combined-NC honest prover. -/
theorem parentProjectionBound
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (pending : ProductionDelayedBlockLane)
    (pendingEq : context.pending = some pending)
    (bound : PendingBound context data) :
    DelayedPackedProjection.projectedValue pending.parentYZcol
        context.producerBeta =
      DelayedCombinedNc.authoritativeRunningProjection context.covers data
        (ProductionProjection.productionWeights context) context.producerBeta
        pending.oldBlock := by
  unfold PendingBound at bound
  rw [pendingEq] at bound
  simp only at bound
  rw [bound]
  exact
    (ProductionProjection.authoritativeRunningProjection_eq_projectedRawRecomposition
      context data context.producerBeta pending.oldBlock).symm

/-- Paper-valid opening-derived sources and an honestly materialized pending
vector construct one accepted production Split-NC certificate. The PiRLC and
PiDEC tail payloads are explicit inert parameters because this theorem owns
only the complete production PiCCS prefix. -/
theorem complete_of_paper
    (carrier : FixedActive.CanonicalOpening.SourceInput.Carrier shape
      publicRingColumns publicFits)
    (context : FixedActive.CanonicalOpening.Context shape State
      publicRingColumns publicFits verifierRows)
    (paper : Semantics.Paper.Holds carrier.data)
    (pendingBound : PendingBound (carrier.install context).full carrier.data)
    (piRlcChallenges : Fin FixedActive.arity.total -> RingF)
    (piDecPayloads : Fin productionGlobalParams.k ->
      PiDecChildPayload (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) :
    ∃ certificate : FixedActive.Certificate (carrier.install context).full,
      MessageAccepted (carrier.install context).full certificate ∧
        YRingBound (carrier.install context).full carrier.data certificate ∧
        Terminal.PackedYZcolBoundAtBlock
          (carrier.install context).full.covers carrier.data
          (ncPoint (carrier.install context).full certificate).block
          certificate.piCcs.output := by
  let full := (carrier.install context).full
  have feTruth : Semantics.Fe.Truth carrier.data :=
    ⟨paper.1, paper.2.2⟩
  have ncTruth : Semantics.Nc.Truth carrier.data := paper.2.1
  rcases
      Fe.HonestProver.exists_honest_certificate full.profile carrier.data
        full.feMachine full.initialState full.feCoins with
    ⟨feCertificate, feHonest⟩
  let feExecution :=
    Transcript.Fe.derive full.feMachine full.initialState feCertificate
  have ncExists :
      ∃ ncCertificate : Transcript.Nc.BlockLane.Certificate
          PiCcsDomains.production.nc,
        FixedPhase.Accepted ops.toOps (rawPolynomial full carrier.data)
          (rawInitial full)
          (Transcript.Nc.BlockLane.derive full.ncMachine
            feExecution.finalState ncCertificate).challengePoint.coordinates
          ncCertificate.toSumCheck := by
    cases pendingEq : full.pending with
    | none =>
        rcases
            Nc.BlockLane.HonestProver.complete_of_truth full.covers carrier.data
              full.ncMachine feExecution.finalState full.ncCoins ncTruth with
          ⟨ncCertificate, accepted⟩
        refine ⟨ncCertificate, ?_⟩
        simpa [rawPolynomial, rawInitial, pendingEq,
          Transcript.Nc.BlockLane.Accepted, FixedPhase.Accepted,
          SumCheck.Nc.Accepted] using accepted
    | some pending =>
        have scalarBound := parentProjectionBound full carrier.data pending
          pendingEq pendingBound
        rcases
            Nc.BlockLane.DelayedHonestProver.complete_of_truth_and_parentProjection
              full.covers carrier.data full.ncMachine feExecution.finalState
              full.ncCoins (ProductionProjection.productionWeights full)
              full.producerBeta full.batchWeight
              (DelayedPackedProjection.projectedValue pending.parentYZcol
                full.producerBeta)
              pending.oldBlock ncTruth scalarBound with
          ⟨ncCertificate, accepted⟩
        refine ⟨ncCertificate, ?_⟩
        simpa [rawPolynomial, rawInitial, pendingEq] using accepted
  rcases ncExists with ⟨ncCertificate, ncRawAccepted⟩
  let ncExecution :=
    Transcript.Nc.BlockLane.derive full.ncMachine feExecution.finalState
      ncCertificate
  let output := canonicalOutput full carrier.data feExecution.challengePoint.row
    ncExecution.challengePoint.block
  let piCcsCertificate : Protocol.BlockLane.Certificate full.piCcsInput
      PiCcsDomains.production := {
    fe := feCertificate
    nc := ncCertificate
    output := output
  }
  let certificate : FixedActive.Certificate full := {
    piCcs := piCcsCertificate
    piRlcChallenges := piRlcChallenges
    piDecPayloads := piDecPayloads
  }
  have outputAuthority :
      output.yRing = Polynomial.Fe.sourceYRingAt carrier.data
          feExecution.challengePoint.row ∧
        Terminal.PackedYZcolBoundAtBlock full.covers carrier.data
          ncExecution.challengePoint.block output := by
    exact canonicalOutput_bound full carrier.data feExecution.challengePoint.row
      ncExecution.challengePoint.block
  have feAccepted : Fe.Accepted full.feMachine full.initialState full.profile
      full.piCcsInput full.feCoins output feCertificate := by
    simpa [full] using
      (Fe.accepted_of_truth_and_honestAt full.feMachine full.initialState
        full.profile carrier.data full.feCoins output feCertificate feTruth
        outputAuthority.1 feHonest)
  have rawAccepted : NcAccepted full carrier.data certificate := by
    simpa [NcAccepted, certificate, piCcsCertificate, ncPoint, ncExecution,
      ncTranscriptState, feExecution] using ncRawAccepted
  have packed : Terminal.PackedYZcolBoundAtBlock full.covers carrier.data
      (ncPoint full certificate).block certificate.piCcs.output := by
    simpa [certificate, piCcsCertificate, ncPoint, ncExecution, feExecution]
      using outputAuthority.2
  have terminalEq :
      messageTerminal full certificate =
        rawPolynomial full carrier.data (ncPoint full certificate).coordinates := by
    cases pendingEq : full.pending with
    | none =>
        simp only [messageTerminal, rawPolynomial, pendingEq]
        calc
          Terminal.terminalFromMessage certificate.piCcs.output full.ncCoins
              (ncPoint full certificate) =
              Mixing.qAtPoint full.covers carrier.data full.ncCoins
                (ncPoint full certificate) :=
            Terminal.terminal_eq_qAtPoint_of_bound full.covers carrier.data
              full.ncCoins (ncPoint full certificate) certificate.piCcs.output
              packed
          _ = InitialSum.sumcheckPolynomial full.covers carrier.data
                full.ncCoins (ncPoint full certificate).coordinates :=
            (InitialSum.sumcheckPolynomial_coordinates_eq_qAtPoint full.covers
              carrier.data full.ncCoins (ncPoint full certificate)).symm
    | some pending =>
        simp only [messageTerminal, rawPolynomial, pendingEq]
        exact MessageTerminal.verifierTerminal_eq_sumcheckPolynomial_of_bound
          full.covers carrier.data full.ncCoins
          (ProductionProjection.productionWeights full) full.producerBeta
          full.batchWeight pending.oldBlock (ncPoint full certificate)
          certificate.piCcs.output packed
  have ncAccepted : NcMessageAccepted full certificate := by
    unfold NcMessageAccepted Transcript.Nc.BlockLane.Accepted
    unfold NcAccepted FixedPhase.Accepted at rawAccepted
    rw [terminalEq]
    simpa [ncTranscriptState, ncPoint_eq_transcriptPoint] using rawAccepted
  refine ⟨certificate, ?_, ?_, ?_⟩
  · exact ⟨by simpa [certificate, piCcsCertificate] using feAccepted, ncAccepted⟩
  · simpa [YRingBound, fePoint, certificate, piCcsCertificate, output,
      canonicalOutput, feExecution] using outputAuthority.1
  · exact packed

end Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionPiCcs.HonestProver
