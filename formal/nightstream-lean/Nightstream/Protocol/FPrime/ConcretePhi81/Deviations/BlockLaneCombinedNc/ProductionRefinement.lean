import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.HonestProver
import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.PaperStep
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc.BlockLane
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.OutputRefinement

/-!
Typed, artifact-independent refinement of the production Split-NC prefix to
SuperNeo Section 7.3.

Assurance tier: model-level registered-deviation refinement.

Owns: the typed production input/certificate boundary, verifier-derived
transcript handoff and output materialization, deterministic reduction to the
paper relation or exact algebraic events, honest completeness, block/lane and
delayed-projection refinement, and the PiRLC output handoff.

Does not own: probability bounds, Fiat--Shamir, Poseidon2/Ajtai internals,
concrete Goldilocks primality, Rust/R1CS, generated artifacts, costs, or rows.

Emits constraints: none.

The only caller data in `AuthoritativeInput` are the opening-derived source
carrier and the verifier context.  Public input, commitments, evaluations,
challenge records, transcript states, block/lane layout, and output values are
all projections or computations from those two values and from message-only
SumCheck certificates.  `Certificate` deliberately has no output field.
`materializeCertificate` creates the complete `yRing`/`yZcol` message from the
FE and NC transcript points, and `AcceptedOutput` exposes it only together with
acceptance.

This module leaves the paper-owned `SplitNc.Semantics.Paper.Holds` proposition
unchanged.  The block/lane and delayed-output paths remain explicitly named as
`FPR-DEV-BLOCK-LANE-COMBINED-NC` and `FPR-DEV-DELAYED-PACKED-YZCOL`.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `pi_ccs.production.input` | derive public input, statement, and source product from canonical openings | computed | `AuthoritativeInput` |
| `pi_ccs.production.transcript` | start NC from FE final state and share verifier-derived mixing coins | computed | `Certificate.feExecution`, `Certificate.ncExecution` |
| `pi_ccs.production.output` | materialize both output products only at verifier-derived points | computed | `Certificate.output`, `Certificate.materialize` |
| `pi_ccs.production.refinement` | reduce acceptance to paper truth or exact FE/NC algebraic events | derived | `accepted_implies_paper_or_algebraic_failure` |
| `pi_ccs.production.deviations` | reduce block/lane and delayed projection to paper-owned relations | derived | `blockLaneCombinedNc_refines_paperNc`, `delayedProjection_refines_rawRecomposition` |
| `pi_ccs.production.complete` | construct accepted messages and the exact downstream PiRLC product | derived | `honest_complete_with_output`, `accepted_output_suitable_for_piRlc` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

set_option maxRecDepth 2048

namespace ProductionPiCcs

export Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionPiCcs
  (fePoint ncPoint ncTranscriptState OutputBindingFailure NcMessageAccepted
    NcAccepted MessageAccepted Accepted BadEvent YRingUnbound YRingBound
    accepted_of_messageAccepted_and_packed
    accepted_implies_paper_or_yRingUnbound_or_badEvent)

export Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionPiCcs.HonestProver
  (PendingBound canonicalOutput canonicalOutput_bound complete_of_paper)

end ProductionPiCcs

universe uState

variable
  {shape : SemanticShape}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Complete typed authority for one production Split-NC invocation.  No
prover-carried public-input, commitment, evaluation, transcript-state, layout,
or output copy occurs in this carrier. -/
structure AuthoritativeInput where
  carrier : FixedActive.CanonicalOpening.SourceInput.Carrier shape
    publicRingColumns publicFits
  context : FixedActive.CanonicalOpening.Context shape State
    publicRingColumns publicFits verifierRows

namespace AuthoritativeInput

/-- Existing complete fixed-active verifier context after installing the sole
authoritative source carrier. -/
def full
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits)) :
    FixedActive.Context shape State publicRingColumns publicFits verifierRows :=
  (input.carrier.install input.context).full

/-- Paper source data computed from the authoritative matrices, fresh
assignment, and canonical opening. -/
def data
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits)) : Data shape :=
  input.carrier.data

/-- Statement-derived Split-NC public input. -/
def publicInput
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits)) : Verifier.PublicInput shape :=
  input.full.piCcsInput

/-- Commitment/public-input/evaluation product reconstructed from the same
source carrier and verifier key. -/
def sourceProduct
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits)) :=
  input.carrier.sourceProduct input.context.key

/-- The complete typed statement bound before challenge sampling. -/
def statement
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits)) :=
  input.full.piCcsStatement

/-- Sole statement-bound pre-SumCheck challenge record and FE entry state. -/
def preSumcheck
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits)) :=
  input.full.piCcsPreSumcheck

/-- The verifier public input is source-derived by construction, not a
prover-supplied equality premise. -/
@[simp] theorem publicInput_eq_sources
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits)) :
    input.publicInput = Verifier.PublicInput.ofSources input.data := by
  rfl

/-- The complete public source product is likewise definitionally the one
materialized from the authoritative opening-derived carrier. -/
theorem sourceProduct_bound
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits)) :
    SemanticFold.InputBound input.full input.data := by
  exact input.carrier.inputBound input.context

/-- FE and NC read the same lane challenge field from one challenge record. -/
@[simp] theorem shared_betaA
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits)) :
    input.full.ncCoins.betaA = input.full.feCoins.betaA := by
  exact input.full.ncCoins_betaA_eq_feCoins_betaA

/-- FE and NC read the same source-mixing challenge field. -/
@[simp] theorem shared_gamma
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits)) :
    input.full.ncCoins.gamma = input.full.feCoins.gamma := by
  exact input.full.ncCoins_gamma_eq_feCoins_gamma

end AuthoritativeInput

/-- Prover-visible production certificate.  It contains exactly FE and NC
round messages plus the later PiRLC/PiDEC payloads.  In particular, there is
no prover field for public input, challenges, transcript states, `yRing`, or
`yZcol`. -/
structure Certificate
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits)) where
  fe : SumCheck.Fe.Certificate input.full.piCcsInput
    PiCcsDomains.production.fe
  nc : Transcript.Nc.BlockLane.Certificate PiCcsDomains.production.nc
  piRlcChallenges : Fin FixedActive.arity.total -> RingF
  piDecPayloads : Fin productionGlobalParams.k ->
    PiDecChildPayload (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows)

namespace Certificate

/-- FE replay is causal: each challenge is derived only after its message is
absorbed by `Transcript.Fe.derive`. -/
def feExecution
    {input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits)}
    (certificate : Certificate input) :=
  Transcript.Fe.derive input.full.feMachine input.full.initialState
    certificate.fe

/-- NC starts from the exact accepted FE successor state.  No restart state or
parallel challenge record is representable. -/
def ncExecution
    {input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits)}
    (certificate : Certificate input) :=
  Transcript.Nc.BlockLane.derive input.full.ncMachine
    certificate.feExecution.finalState certificate.nc

/-- Verifier-side output materialization at the two transcript-derived points.
This value is absent from `Certificate` and is computed only after the NC
replay has produced its block point. -/
def output
    {input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits)}
    (certificate : Certificate input) : OutputMessage shape :=
  ProductionPiCcs.canonicalOutput input.full input.data
    certificate.feExecution.challengePoint.row
    certificate.ncExecution.challengePoint.block

/-- Complete existing certificate obtained by installing the verifier-owned
output. -/
def materialize
    {input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits)}
    (certificate : Certificate input) : FixedActive.Certificate input.full := {
  piCcs := {
    fe := certificate.fe
    nc := certificate.nc
    output := certificate.output
  }
  piRlcChallenges := certificate.piRlcChallenges
  piDecPayloads := certificate.piDecPayloads
}

/-- FE-to-NC handoff is exact by construction. -/
@[simp] theorem nc_initial_state_eq_fe_final
    {input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits)}
    (certificate : Certificate input) :
    ProductionPiCcs.ncTranscriptState input.full certificate.materialize =
      certificate.feExecution.finalState := by
  rfl

/-- The production FE point is exactly the point derived from the message-only
FE certificate. -/
@[simp] theorem fePoint_materialize
    {input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits)}
    (certificate : Certificate input) :
    ProductionPiCcs.fePoint input.full certificate.materialize =
      certificate.feExecution.challengePoint := by
  rfl

/-- The production NC point is exactly the block-prefix/lane-suffix replay
from FE's final state. -/
@[simp] theorem ncPoint_materialize
    {input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits)}
    (certificate : Certificate input) :
    ProductionPiCcs.ncPoint input.full certificate.materialize =
      certificate.ncExecution.challengePoint := by
  rfl

/-- Both output products are authoritative computations. -/
theorem output_bound
    {input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits)}
    (certificate : Certificate input) :
    ProductionPiCcs.YRingBound input.full input.data certificate.materialize ∧
      Polynomial.Nc.BlockLane.Terminal.PackedYZcolBoundAtBlock
        input.full.covers input.data
        (ProductionPiCcs.ncPoint input.full certificate.materialize).block
        certificate.materialize.piCcs.output := by
  simpa [ProductionPiCcs.YRingBound] using
    (ProductionPiCcs.canonicalOutput_bound input.full input.data
      certificate.feExecution.challengePoint.row
      certificate.ncExecution.challengePoint.block)

/-- Output absorption is structurally after the full NC replay. -/
@[simp] theorem finalState_eq_absorbOutput_after_nc
    {input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits)}
    (certificate : Certificate input) :
    (derive input.full certificate.materialize).piCcs.finalState =
      input.full.piCcsSchedule.absorbOutput
        certificate.ncExecution.finalState certificate.output := by
  rfl

end Certificate

/-- Executable production acceptance over a certificate with no output copy.
The verifier first derives both transcript points, materializes the output,
and then checks the existing claims-only relation. -/
def ProductionVerifierAccepts
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (certificate : Certificate input) : Prop :=
  ProductionPiCcs.MessageAccepted input.full certificate.materialize

/-- Authoritative outputs are exposed only with an accepted certificate. -/
structure AcceptedOutput
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits)) where
  certificate : Certificate input
  accepted : ProductionVerifierAccepts input certificate

namespace AcceptedOutput

/-- The sole authoritative output sidecar. -/
def output
    {input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits)}
    (result : AcceptedOutput input) : OutputMessage shape :=
  result.certificate.output

end AcceptedOutput

/-- Exact FE failure branch. -/
inductive FeFailure
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (certificate : Certificate input) : Prop where
  | sumcheck
      (bound : input.full.piCcsInput = Verifier.PublicInput.ofSources input.data)
      (bad : SumCheck.Fe.BadEvent input.full.profile input.data
        input.full.feCoins (ProductionPiCcs.fePoint input.full certificate.materialize)
        (Protocol.BlockLane.certificateAtSources input.data
          certificate.materialize.piCcs bound).fe
        input.full.challengeSetSize) :
      FeFailure input certificate

/-- Exact NC failure branches, preserving the frozen loss ordering. -/
inductive NcFailure
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (certificate : Certificate input) : Prop where
  | laneSelectorRoot
      (root : Polynomial.Nc.BlockLane.MixingSoundness.LaneSelectorRoot
        input.full.covers input.data input.full.ncCoins) : NcFailure input certificate
  | blockSelectorRoot
      (root : Polynomial.Nc.BlockLane.MixingSoundness.BlockSelectorRoot
        input.full.covers input.data input.full.ncCoins) : NcFailure input certificate
  | gammaPolynomialRoot
      (root : Polynomial.Nc.BlockLane.MixingSoundness.GammaPolynomialRoot
        input.full.covers input.data input.full.ncCoins) : NcFailure input certificate
  | residualWeightRoot
      (pending : ProductionDelayedBlockLane)
      (pendingEq : input.full.pending = some pending)
      (root : Polynomial.Nc.BlockLane.DelayedCombinedNc.Acceptance.ResidualWeightRoot
        input.full.covers input.data input.full.ncCoins
        (ProductionProjection.productionWeights input.full)
        input.full.producerBeta input.full.batchWeight
        (DelayedPackedProjection.projectedValue pending.parentYZcol
          input.full.producerBeta) pending.oldBlock) : NcFailure input certificate
  | roundCollision
      (round : Nightstream.SuperNeo.SumCheck.Round K K)
      (collision : Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.BadChallenge
        ConcreteCarrier.extensionOps.toOps
        (ProductionPiCcs.rawPolynomial input.full input.data)
        Polynomial.Nc.Degree.ncSumcheckDegreeBound input.full.challengeSetSize
        (ProductionPiCcs.rawInitial input.full)
        (ProductionPiCcs.ncPoint input.full certificate.materialize).coordinates
        certificate.nc.toSumCheck round) : NcFailure input certificate

/-- Typed transcript failures requested by the production boundary.  Every
constructor contradicts a definitional or structural equality of the typed
schedule; a concrete Poseidon2/Fiat--Shamir refinement may map a primitive
collision into this named branch. -/
inductive TranscriptFailure
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (certificate : Certificate input) : Prop where
  | feToNcReset
      (different : ProductionPiCcs.ncTranscriptState input.full
          certificate.materialize ≠ certificate.feExecution.finalState) :
      TranscriptFailure input certificate
  | betaAFork
      (different : input.full.ncCoins.betaA ≠ input.full.feCoins.betaA) :
      TranscriptFailure input certificate
  | gammaFork
      (different : input.full.ncCoins.gamma ≠ input.full.feCoins.gamma) :
      TranscriptFailure input certificate

/-- Output/commitment binding failures remain separately named.  The
verifier-materialized output makes this branch unreachable at the Split-NC
prefix, while the delayed trace owns the later commitment-opening events. -/
inductive BindingFailure
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (certificate : Certificate input) : Prop where
  | packedOutput
      (failure : ProductionPiCcs.OutputBindingFailure input.full input.data
        certificate.materialize) : BindingFailure input certificate

/-- The two frozen deviations, kept explicit rather than folded into a generic
failure. -/
inductive RegisteredDeviationObligation
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (certificate : Certificate input) : Prop where
  | blockLaneCombinedNc
      (failure :
        ¬ (Semantics.Nc.BlockLane.ResidualsZero input.data ↔
          Semantics.Nc.Truth input.data)) : RegisteredDeviationObligation input certificate
  | delayedPackedYZcol
      (pending : ProductionDelayedBlockLane)
      (pendingEq : input.full.pending = some pending)
      (unbound : ¬ ProductionPiCcs.PendingBound input.full input.data) :
      RegisteredDeviationObligation input certificate

private theorem classifyBadEvent
    {input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits)}
    {certificate : Certificate input}
    (bad : ProductionPiCcs.BadEvent input.full input.data
      certificate.materialize) :
    FeFailure input certificate ∨ NcFailure input certificate := by
  cases bad with
  | fe bound event => exact Or.inl (.sumcheck bound event)
  | laneSelectorRoot root => exact Or.inr (.laneSelectorRoot root)
  | blockSelectorRoot root => exact Or.inr (.blockSelectorRoot root)
  | gammaPolynomialRoot root => exact Or.inr (.gammaPolynomialRoot root)
  | residualWeightRoot pending pendingEq root =>
      exact Or.inr (.residualWeightRoot pending pendingEq root)
  | roundCollision round collision => exact Or.inr (.roundCollision round collision)

/-- Production Split-NC soundness at the complete source-derived boundary.
There is no opaque `BoundToSources` or output-authority premise: both are
computed by `AuthoritativeInput` and `Certificate.materialize`. -/
theorem accepted_implies_paper_or_named_failure
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (certificate : Certificate input)
    (accepted : ProductionVerifierAccepts input certificate) :
    Semantics.Paper.Holds input.data ∨
      FeFailure input certificate ∨
      NcFailure input certificate ∨
      TranscriptFailure input certificate ∨
      BindingFailure input certificate ∨
      RegisteredDeviationObligation input certificate := by
  have raw : ProductionPiCcs.Accepted input.full input.data
      certificate.materialize :=
    ProductionPiCcs.accepted_of_messageAccepted_and_packed input.full input.data
      certificate.materialize accepted certificate.output_bound.2
  rcases ProductionPiCcs.accepted_implies_paper_or_yRingUnbound_or_badEvent
      noZeroDivisors input.full input.data certificate.materialize
      input.publicInput_eq_sources raw with paper | yRingUnbound | bad
  · exact Or.inl paper
  · exact False.elim (yRingUnbound certificate.output_bound.1)
  · rcases classifyBadEvent bad with fe | nc
    · exact Or.inr (Or.inl fe)
    · exact Or.inr (Or.inr (Or.inl nc))

/-- The exact deterministic Split-NC reduction needs only its algebraic FE
or NC events. Transcript, binding, and registered-deviation constructors are
not escape branches in the reduction proof. -/
theorem accepted_implies_paper_or_algebraic_failure
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (certificate : Certificate input)
    (accepted : ProductionVerifierAccepts input certificate) :
    Semantics.Paper.Holds input.data ∨
      FeFailure input certificate ∨ NcFailure input certificate := by
  have raw : ProductionPiCcs.Accepted input.full input.data
      certificate.materialize :=
    ProductionPiCcs.accepted_of_messageAccepted_and_packed input.full input.data
      certificate.materialize accepted certificate.output_bound.2
  rcases ProductionPiCcs.accepted_implies_paper_or_yRingUnbound_or_badEvent
      noZeroDivisors input.full input.data certificate.materialize
      input.publicInput_eq_sources raw with paper | yRingUnbound | bad
  · exact Or.inl paper
  · exact False.elim (yRingUnbound certificate.output_bound.1)
  · rcases classifyBadEvent bad with fe | nc
    · exact Or.inr (Or.inl fe)
    · exact Or.inr (Or.inr nc)

/-- Typed transcript reset/fork events are impossible in the model-level
schedule. -/
theorem not_transcriptFailure
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (certificate : Certificate input) :
    ¬ TranscriptFailure input certificate := by
  intro failure
  cases failure with
  | feToNcReset different => exact different certificate.nc_initial_state_eq_fe_final
  | betaAFork different => exact different input.shared_betaA
  | gammaFork different => exact different input.shared_gamma

/-- Canonical materialization makes the sole Split-NC output-binding failure
uninhabited. -/
theorem not_bindingFailure
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (certificate : Certificate input) :
    ¬ BindingFailure input certificate := by
  intro failure
  cases failure with
  | packedOutput unbound =>
      exact unbound certificate.output_bound.2

/-- FPR-DEV-BLOCK-LANE-COMBINED-NC refines exactly to the paper strict norm
relation.  The coordinate decoder is a two-sided ownership witness and the
residual family is equivalent to `Semantics.Nc.Truth`; no lane, block, or
carrier coordinate can be omitted or duplicated. -/
theorem blockLaneCombinedNc_refines_paperNc
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits)) :
    Semantics.Nc.BlockLane.ResidualsZero input.data ↔
      Semantics.Nc.Truth input.data :=
  Semantics.Nc.BlockLane.residualsZero_iff_truth noZeroDivisors input.data

/-- Every flat paper carrier coordinate has its unique canonical block/lane
owner, and flattening that owner returns the same coordinate. -/
theorem everyCoordinate_has_exact_owner
    (column : Fin shape.carrierWidth) :
    Semantics.Nc.BlockLane.carrierColumn
        (Phi81ColumnLayout.decode column).1
        (Phi81ColumnLayout.decode column).2 = column :=
  Semantics.Nc.BlockLane.carrierColumn_decode column

/-- Exact delayed production projection over all fourteen authoritative raw
running assignments.  This is the production-to-paper bridge; no message
identity with the paper's displayed polynomial is claimed. -/
theorem delayedProjection_refines_rawRecomposition
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (oldBlock : CubePoint K PiCcsDomains.production.nc.blockVariables) :
    Polynomial.Nc.BlockLane.DelayedCombinedNc.authoritativeRunningProjection
        input.full.covers input.data
        (ProductionProjection.productionWeights input.full)
        input.full.producerBeta oldBlock =
      DelayedPackedProjection.projectedValue
        (PackedBlockAction.packedYZcol input.full.covers
          (PiDEC.Raw.recomposeAssignment
            (DelayedRawChildren.rawRunningAssignments
              input.full input.data)) oldBlock)
        input.full.producerBeta :=
  ProductionProjection.authoritativeRunningProjection_eq_projectedRawRecomposition
    input.full input.data input.full.producerBeta oldBlock

/-- Honest completeness for the complete typed production Split-NC prefix.
The pending premise is the narrowly typed FPR-DEV-DELAYED-PACKED-YZCOL
lifecycle authority: at recursive steps the complete pending vector must be
the raw-running recomposition owned by that iteration; at base it is `True`. -/
theorem honest_complete
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (paper : Semantics.Paper.Holds input.data)
    (pendingBound : ProductionPiCcs.PendingBound input.full input.data)
    (piRlcChallenges : Fin FixedActive.arity.total -> RingF)
    (piDecPayloads : Fin productionGlobalParams.k ->
      PiDecChildPayload (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) :
    ∃ certificate : Certificate input,
      ProductionVerifierAccepts input certificate := by
  rcases ProductionPiCcs.complete_of_paper input.carrier input.context paper
      pendingBound piRlcChallenges piDecPayloads with
    ⟨fullCertificate, accepted, yRing, packed⟩
  let certificate : Certificate input := {
    fe := fullCertificate.piCcs.fe
    nc := fullCertificate.piCcs.nc
    piRlcChallenges := fullCertificate.piRlcChallenges
    piDecPayloads := fullCertificate.piDecPayloads
  }
  have outputEq : certificate.materialize.piCcs.output =
      fullCertificate.piCcs.output := by
    apply Claims.ext
    · intro source matrix lane
      have computed := certificate.output_bound.1
      unfold ProductionPiCcs.YRingBound at computed yRing
      exact congrFun (congrFun (congrFun (computed.trans yRing.symm) source)
        matrix) lane
    · intro source lane
      have computed := certificate.output_bound.2 source lane
      have original := packed source lane
      exact computed.trans original.symm
  have certificateEq : certificate.materialize = fullCertificate := by
    cases fullCertificate with
    | mk piCcs piRlc piDec =>
      cases piCcs with
      | mk fe nc output =>
        simp only [certificate, Certificate.materialize] at outputEq ⊢
        cases outputEq
        rfl
  refine ⟨certificate, ?_⟩
  simpa [ProductionVerifierAccepts, certificateEq] using accepted

/-- Explicit output presentation of honest completeness. The output is not a
second witness: it is definitionally the verifier computation attached to the
accepted message-only certificate. -/
theorem honest_complete_with_output
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (paper : Semantics.Paper.Holds input.data)
    (pendingBound : ProductionPiCcs.PendingBound input.full input.data)
    (piRlcChallenges : Fin FixedActive.arity.total -> RingF)
    (piDecPayloads : Fin productionGlobalParams.k ->
      PiDecChildPayload (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) :
    ∃ certificate : Certificate input, ∃ output : OutputMessage shape,
      ProductionVerifierAccepts input certificate ∧
        output = certificate.output := by
  rcases honest_complete input paper pendingBound piRlcChallenges
      piDecPayloads with ⟨certificate, accepted⟩
  exact ⟨certificate, certificate.output, accepted, rfl⟩

/-- Accepted materialized outputs are the exact honest PiCCS product consumed
by PiRLC.  Statement identity, the common relation `phi`, source order, and the
complete evaluation product are inherited from the same authoritative source
product. -/
theorem accepted_output_suitable_for_piRlc
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (certificate : Certificate input)
    (accepted : ProductionVerifierAccepts input certificate)
    (noFailure : ¬ FeFailure input certificate ∧ ¬ NcFailure input certificate) :
    (derive input.full certificate.materialize).piCcsOutputs =
      PiCCS.honestOutputs (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.semantics input.full.key)
        input.full.input
        (InputAuthority.productAssignments input.data input.full.alignment)
        certificate.feExecution.challengePoint.row := by
  have raw : ProductionPiCcs.Accepted input.full input.data
      certificate.materialize :=
    ProductionPiCcs.accepted_of_messageAccepted_and_packed input.full input.data
      certificate.materialize accepted certificate.output_bound.2
  have paper : Semantics.Paper.Holds input.data := by
    rcases ProductionPiCcs.accepted_implies_paper_or_yRingUnbound_or_badEvent
        noZeroDivisors input.full input.data certificate.materialize
        input.publicInput_eq_sources raw with paper | yRingUnbound | bad
    · exact paper
    · exact False.elim (yRingUnbound certificate.output_bound.1)
    · rcases classifyBadEvent bad with fe | nc
      · exact False.elim (noFailure.1 fe)
      · exact False.elim (noFailure.2 nc)
  change
    OutputProduct.materialize publicRingColumns publicFits input.full.alignment
        input.full.input certificate.feExecution.challengePoint.row
        certificate.output =
      PiCCS.honestOutputs (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.semantics input.full.key)
        input.full.input
        (InputAuthority.productAssignments input.data input.full.alignment)
        certificate.feExecution.challengePoint.row
  simpa [Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.semantics] using
    (Protocol.OutputRefinement.materializedOutputs_eq_honestOutputs_of_yRingEq
      publicRingColumns publicFits (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.commit input.full.key)
      input.data input.full.alignment input.full.input
      certificate.feExecution.challengePoint.row certificate.output
      production_norm_stages.1 paper input.sourceProduct_bound
      certificate.output_bound.1)

end Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement
