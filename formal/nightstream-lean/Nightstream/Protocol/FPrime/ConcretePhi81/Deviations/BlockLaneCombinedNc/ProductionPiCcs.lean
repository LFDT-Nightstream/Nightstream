import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.MessageTerminal
import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionProjection
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Transition

/-!
Production `Pi_CCS` acceptance with a raw-assignment NC terminal.

Assurance tier: model-level registered production deviation, with a separate
concrete Rust/R1CS decoder boundary.

Owns: reuse of the existing statement-derived FE transcript; selection of the
ordinary raw NC polynomial at the base boundary or the delayed combined-NC
polynomial when a pending parent is present; a terminal computed exclusively
from `Sources.Data`; and current-step paper soundness with `yRing` separated
from delayed `yZcol` authority.

Does not own: construction of `Sources.Data` from production assignment
columns, exact matrix rows, the successor-step continuity proof, terminal
closure, commitment binding, Rust conformance, costs, or row removal.

Emits constraints: none.

Authority boundary: `NcAccepted` evaluates the final claimed-chain value by
calling `rawPolynomial` on the decoded raw source table. It never reads
`certificate.piCcs.output.yZcol` or any child `CeClaim.y_zcol` sidecar. The
public-input equality used by the soundness theorem is intentionally visible;
the generated production decoder must derive it before final composition.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.production.fe` | retain the existing statement-bound FE phase | checked | `Accepted.fe` |
| `nifs.pi_ccs.production.nc.base` | evaluate the ordinary NC polynomial directly from raw assignments | checked/computed | `rawPolynomial`, `NcAccepted` |
| `nifs.pi_ccs.production.nc.delayed` | add the delayed old-parent residual to that same raw polynomial | checked/computed | `rawPolynomial`, `NcAccepted` |
| `nifs.pi_ccs.production.nc.initial` | use zero at base or the transcript-weighted pending projection | computed | `rawInitial` |
| `nifs.pi_ccs.production.soundness` | derive paper truth, `yRing` mismatch, or one named algebraic event | derived | `accepted_implies_paper_or_yRingUnbound_or_badEvent` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionPiCcs

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.DelayedCombinedNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.MixingSoundness
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Degree
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority
open Nightstream.SuperNeo.SumCheck.Finite

private abbrev ops := ConcreteCarrier.extensionOps

universe uState

variable
  {shape : SemanticShape}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-- The verifier-derived FE point. This is the same replay used by the
ordinary `Protocol.BlockLane` path; only the subsequent NC terminal changes. -/
def fePoint
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context) :
    Polynomial.Fe.Point shape PiCcsDomains.production.fe :=
  (derive context certificate).piCcs.fePoint

/-- The verifier-derived block/lane NC point from the same chained replay. -/
def ncPoint
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context) :
    Polynomial.Nc.BlockLane.Point PiCcsDomains.production.nc :=
  (derive context certificate).piCcs.ncPoint

/-- Exact FE successor state entering the one-entry block×lane NC replay.
This is computed by the canonical protocol schedule, not supplied by the
prover or caller. -/
def ncTranscriptState
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context) : State :=
  (Transcript.Fe.derive context.feMachine context.initialState
    certificate.piCcs.fe).finalState

/-- The production NC point is exactly the point obtained by replaying the NC
certificate from the FE successor state. -/
@[simp] theorem ncPoint_eq_transcriptPoint
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context) :
    ncPoint context certificate =
      (Transcript.Nc.BlockLane.derive context.ncMachine
        (ncTranscriptState context certificate) certificate.piCcs.nc
        ).challengePoint := by
  rfl

/-- The only delayed scalar permitted at the NC initial boundary: evaluate
the complete statement-bound pending vector at the statement-derived producer
challenge. Absence at the base boundary yields the ordinary zero claim. -/
def rawInitial
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows) : K :=
  match context.pending with
  | none => InitialSum.claimedInitial
  | some pending =>
      K.mul context.batchWeight
        (DelayedPackedProjection.projectedValue pending.parentYZcol
          context.producerBeta)

/-- Production-native NC polynomial. Both branches read the same authoritative
`Sources.Data`; the recursive branch adds exactly one delayed residual. -/
def rawPolynomial
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape) : List K -> K :=
  match context.pending with
  | none => InitialSum.sumcheckPolynomial context.covers data context.ncCoins
  | some pending =>
      sumcheckPolynomial context.covers data context.ncCoins
        (ProductionProjection.productionWeights context)
        context.producerBeta context.batchWeight pending.oldBlock

/-- Terminal computed by a claims-only production verifier. At base it is the
ordinary packed-output terminal. With a pending parent it adds the delayed
running-suffix evaluation, still computed solely from the current public
output message. -/
def messageTerminal
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context) : K :=
  match context.pending with
  | none =>
      Terminal.terminalFromMessage certificate.piCcs.output context.ncCoins
        (ncPoint context certificate)
  | some pending =>
      MessageTerminal.verifierTerminal certificate.piCcs.output
        context.ncCoins (ProductionProjection.productionWeights context)
        context.producerBeta context.batchWeight pending.oldBlock
        (ncPoint context certificate)

/-- Actual claims-level NC relation: replay the exact 21-block-plus-6-lane
certificate from FE's successor state and compare its final claim with the
verifier-computed public terminal. No private assignment is an input. -/
def NcMessageAccepted
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context) : Prop :=
  Transcript.Nc.BlockLane.Accepted context.ncMachine
    (ncTranscriptState context certificate) (rawInitial context)
    (messageTerminal context certificate) certificate.piCcs.nc

/-- Exact weak-relation/extraction failure for the current output product.
Unlike a generic `outputUnbound`, this names only the full active packed
`yZcol` opening at the transcript-derived block point. -/
def OutputBindingFailure
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context) : Prop :=
  ¬ Terminal.PackedYZcolBoundAtBlock context.covers data
      (ncPoint context certificate).block certificate.piCcs.output

/-- Physical NC claimed-chain acceptance against the raw-source terminal.
The certificate contributes only five-slot round messages. -/
def NcAccepted
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context) : Prop :=
  FixedPhase.Accepted ops.toOps (rawPolynomial context data)
    (rawInitial context) (ncPoint context certificate).coordinates
    certificate.piCcs.nc.toSumCheck

/-- Claims-level acceptance refines to the independent raw-witness NC
relation or exposes exactly the missing output opening. The recursive branch
uses `MessageTerminal`; the base branch is the ordinary block×lane terminal
rewrite. -/
theorem ncMessageAccepted_implies_ncAccepted_or_outputBindingFailure
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context)
    (accepted : NcMessageAccepted context certificate) :
    NcAccepted context data certificate ∨
      OutputBindingFailure context data certificate := by
  cases pendingEq : context.pending with
  | none =>
      by_cases bound : Terminal.PackedYZcolBoundAtBlock context.covers data
          (ncPoint context certificate).block certificate.piCcs.output
      · apply Or.inl
        have terminalBinding :
            messageTerminal context certificate =
              InitialSum.sumcheckPolynomial context.covers data
                context.ncCoins (ncPoint context certificate).coordinates := by
          simp only [messageTerminal, pendingEq]
          calc
            Terminal.terminalFromMessage certificate.piCcs.output
                context.ncCoins (ncPoint context certificate) =
                Mixing.qAtPoint context.covers data context.ncCoins
                  (ncPoint context certificate) :=
              Terminal.terminal_eq_qAtPoint_of_bound context.covers data
                context.ncCoins (ncPoint context certificate)
                certificate.piCcs.output bound
            _ = InitialSum.sumcheckPolynomial context.covers data
                  context.ncCoins
                  (ncPoint context certificate).coordinates :=
              (InitialSum.sumcheckPolynomial_coordinates_eq_qAtPoint
                context.covers data context.ncCoins
                (ncPoint context certificate)).symm
        unfold NcAccepted
        simp only [rawPolynomial, rawInitial, pendingEq]
        unfold NcMessageAccepted Transcript.Nc.BlockLane.Accepted at accepted
        unfold FixedPhase.Accepted
        rw [← terminalBinding]
        unfold SumCheck.Nc.Accepted at accepted
        simpa only [rawInitial, pendingEq, ncPoint_eq_transcriptPoint] using
          accepted
      · exact Or.inr bound
  | some pending =>
      have transcriptAccepted :
          MessageTerminal.TranscriptAcceptedFromMessage context.ncMachine
            (ncTranscriptState context certificate) certificate.piCcs.output
            context.ncCoins (ProductionProjection.productionWeights context)
            context.producerBeta context.batchWeight (rawInitial context)
            pending.oldBlock certificate.piCcs.nc := by
        simpa [NcMessageAccepted, messageTerminal, pendingEq,
          MessageTerminal.TranscriptAcceptedFromMessage,
          ncPoint_eq_transcriptPoint] using accepted
      rcases
          MessageTerminal.transcriptAcceptedFromMessage_implies_rawAccepted_or_outputBindingFailure
            context.covers data context.ncMachine
            (ncTranscriptState context certificate) certificate.piCcs.output
            context.ncCoins (ProductionProjection.productionWeights context)
            context.producerBeta context.batchWeight (rawInitial context)
            pending.oldBlock certificate.piCcs.nc transcriptAccepted with
        raw | failure
      · apply Or.inl
        simpa [NcAccepted, rawPolynomial, pendingEq,
          ncPoint_eq_transcriptPoint] using raw
      · exact Or.inr (by
          simpa [OutputBindingFailure, ncPoint_eq_transcriptPoint] using
            failure)

/-- Claims-only production `Pi_CCS` prefix. This is the executable protocol
surface: FE and NC both consume public messages and transcript-derived
challenges, and no private source table appears in the predicate. -/
structure MessageAccepted
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context) : Prop where
  fe : Fe.Accepted context.feMachine context.initialState context.profile
    context.piCcsInput context.feCoins certificate.piCcs.output
    certificate.piCcs.fe
  nc : NcMessageAccepted context certificate

/-- Post-extraction production `Pi_CCS` prefix. FE retains the canonical
message terminal for `yRing`; NC has been refined from the public terminal to
the independent raw-assignment polynomial. -/
structure Accepted
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context) : Prop where
  fe : Fe.Accepted context.feMachine context.initialState context.profile
    context.piCcsInput context.feCoins certificate.piCcs.output
    certificate.piCcs.fe
  nc : NcAccepted context data certificate

/-- Exact handoff from the executable claims verifier to the post-extraction
semantic predicate. The only new branch is the specifically named packed
output-opening failure. -/
theorem messageAccepted_implies_accepted_or_outputBindingFailure
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context)
    (accepted : MessageAccepted context certificate) :
    Accepted context data certificate ∨
      OutputBindingFailure context data certificate := by
  rcases ncMessageAccepted_implies_ncAccepted_or_outputBindingFailure
      context data certificate accepted.nc with raw | failure
  · exact Or.inl ⟨accepted.fe, raw⟩
  · exact Or.inr failure

/-- A separately derived packed opening discharges the exact weak-output
failure and converts public-message acceptance to the raw-source predicate.
The opening must be at this certificate's verifier-replayed NC block point. -/
theorem accepted_of_messageAccepted_and_packed
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context)
    (accepted : MessageAccepted context certificate)
    (packed : Terminal.PackedYZcolBoundAtBlock context.covers data
      (ncPoint context certificate).block certificate.piCcs.output) :
    Accepted context data certificate := by
  rcases messageAccepted_implies_accepted_or_outputBindingFailure context data
      certificate accepted with raw | failure
  · exact raw
  · exact (failure packed).elim

/-- The only unresolved output authority in the current step. Packed
`yZcol` is intentionally absent because its pending value is checked by the
successor (or terminal) raw combined-NC certificate. -/
def YRingUnbound
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context) : Prop :=
  Polynomial.Fe.OutputMismatch data (fePoint context certificate)
    certificate.piCcs.output

/-- Positive form of the sole same-step output binding retained by this
production checker. -/
def YRingBound
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context) : Prop :=
  certificate.piCcs.output.yRing =
    Polynomial.Fe.sourceYRingAt data (fePoint context certificate).row

/-- Algebraic events for current-step production `Pi_CCS` soundness. The
residual-root constructor is available only when the exact pending value in
the transcript-bound context is present. -/
inductive BadEvent
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context) : Prop where
  | fe
      (bound : context.piCcsInput = PublicInput.ofSources data)
      (bad : SumCheck.Fe.BadEvent context.profile data context.feCoins
        (fePoint context certificate)
        (Protocol.BlockLane.certificateAtSources data certificate.piCcs
          bound).fe
        context.challengeSetSize) :
      BadEvent context data certificate
  | laneSelectorRoot
      (root : LaneSelectorRoot context.covers data context.ncCoins) :
      BadEvent context data certificate
  | blockSelectorRoot
      (root : BlockSelectorRoot context.covers data context.ncCoins) :
      BadEvent context data certificate
  | gammaPolynomialRoot
      (root : GammaPolynomialRoot context.covers data context.ncCoins) :
      BadEvent context data certificate
  | residualWeightRoot
      (pending : ProductionDelayedBlockLane)
      (pendingEq : context.pending = some pending)
      (root : Acceptance.ResidualWeightRoot context.covers data
        context.ncCoins (ProductionProjection.productionWeights context)
        context.producerBeta context.batchWeight
        (DelayedPackedProjection.projectedValue pending.parentYZcol
          context.producerBeta)
        pending.oldBlock) :
      BadEvent context data certificate
  | roundCollision
      (round : Nightstream.SuperNeo.SumCheck.Round K K)
      (collision : FixedPhase.BadChallenge ops.toOps
        (rawPolynomial context data) ncSumcheckDegreeBound
        context.challengeSetSize (rawInitial context)
        (ncPoint context certificate).coordinates
        certificate.piCcs.nc.toSumCheck round) :
      BadEvent context data certificate

/-- Reindexing only the dependent FE certificate index preserves the physical
FE acceptance relation. No source equation is inferred by the cast. -/
private theorem feAccepted_atSources
    {input : PublicInput shape}
    (data : Data shape)
    (certificate : Protocol.BlockLane.Certificate input
      PiCcsDomains.production)
    (bound : input = PublicInput.ofSources data)
    (machine : Transcript.Fe.Machine State)
    (initialState : State)
    (profile : Polynomial.Fe.SupportedProfile shape
      PiCcsDomains.production.fe)
    (coins : Polynomial.Fe.Coins shape PiCcsDomains.production.fe)
    (message : OutputMessage shape)
    (accepted : Fe.Accepted machine initialState profile input coins message
      certificate.fe) :
    Fe.Accepted machine initialState profile (PublicInput.ofSources data) coins
      message (Protocol.BlockLane.certificateAtSources data certificate
        bound).fe := by
  subst input
  exact accepted

/-- The FE point is unchanged by the dependent source-index transport. -/
private theorem fePoint_atSources
    {input : PublicInput shape}
    (data : Data shape)
    (certificate : Protocol.BlockLane.Certificate input
      PiCcsDomains.production)
    (bound : input = PublicInput.ofSources data)
    (machine : Transcript.Fe.Machine State)
    (initialState : State) :
    Transcript.Fe.derive machine initialState
        (Protocol.BlockLane.certificateAtSources data certificate bound).fe =
      Transcript.Fe.derive machine initialState certificate.fe := by
  subst input
  rfl

/-- Raw NC acceptance implies the current authoritative norm relation or one
named event. In the delayed branch the parent scalar is also recovered by
`Acceptance`; it is discarded here and consumed by the adjacent-step theorem. -/
theorem ncAccepted_implies_truth_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context)
    (accepted : NcAccepted context data certificate) :
    Semantics.Nc.Truth data ∨ BadEvent context data certificate := by
  cases pendingEq : context.pending with
  | none =>
      have baseAccepted :
          SumCheck.Nc.Accepted InitialSum.claimedInitial
            (ncPoint context certificate).coordinates
            (InitialSum.sumcheckPolynomial context.covers data context.ncCoins
              (ncPoint context certificate).coordinates)
            certificate.piCcs.nc.toSumCheck := by
        simpa [NcAccepted, rawPolynomial, rawInitial, pendingEq,
          FixedPhase.Accepted, SumCheck.Nc.Accepted] using accepted
      rcases SumCheck.Nc.BlockLane.accepted_implies_truth_or_badEvent
          noZeroDivisors context.covers data context.ncCoins
          (ncPoint context certificate) certificate.piCcs.nc.toSumCheck
          context.challengeSetSize baseAccepted with truth | bad
      · exact Or.inl truth
      · apply Or.inr
        cases bad with
        | laneSelectorRoot root => exact .laneSelectorRoot root
        | blockSelectorRoot root => exact .blockSelectorRoot root
        | gammaPolynomialRoot root => exact .gammaPolynomialRoot root
        | roundCollision round collision =>
            exact .roundCollision round (by
              simpa [rawPolynomial, rawInitial, pendingEq] using collision)
  | some pending =>
      have delayedAccepted : FixedPhase.Accepted ops.toOps
          (sumcheckPolynomial context.covers data context.ncCoins
            (ProductionProjection.productionWeights context)
            context.producerBeta context.batchWeight pending.oldBlock)
          (K.mul context.batchWeight
            (DelayedPackedProjection.projectedValue pending.parentYZcol
              context.producerBeta))
          (ncPoint context certificate).coordinates
          certificate.piCcs.nc.toSumCheck := by
        simpa [NcAccepted, rawPolynomial, rawInitial, pendingEq] using accepted
      rcases Acceptance.accepted_implies_truth_and_parentProjection_or_badEvent
          noZeroDivisors context.covers data context.ncCoins
          (ProductionProjection.productionWeights context)
          context.producerBeta context.batchWeight
          (DelayedPackedProjection.projectedValue pending.parentYZcol
            context.producerBeta)
          pending.oldBlock (ncPoint context certificate)
          certificate.piCcs.nc.toSumCheck context.challengeSetSize
          delayedAccepted with
        semantic | laneRoot | blockRoot | gammaRoot | residualRoot |
          roundCollision
      · exact Or.inl semantic.1
      · exact Or.inr (.laneSelectorRoot laneRoot)
      · exact Or.inr (.blockSelectorRoot blockRoot)
      · exact Or.inr (.gammaPolynomialRoot gammaRoot)
      · exact Or.inr (.residualWeightRoot pending pendingEq residualRoot)
      · rcases roundCollision with ⟨round, collision⟩
        exact Or.inr (.roundCollision round (by
          simpa [rawPolynomial, rawInitial, pendingEq] using collision))

/-- Current-step production soundness. Unlike the generic message-terminal
theorem, this partition has no packed-`yZcol` unbound outcome: NC reads the
raw assignments, while delayed output authority is owned by the adjacent-step
composition. -/
theorem accepted_implies_paper_or_yRingUnbound_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context)
    (publicInputBound : context.piCcsInput = PublicInput.ofSources data)
    (accepted : Accepted context data certificate) :
    Semantics.Paper.Holds data ∨
      YRingUnbound context data certificate ∨
      BadEvent context data certificate := by
  let sourceCertificate := Protocol.BlockLane.certificateAtSources data
    certificate.piCcs publicInputBound
  have sourceFeAccepted : Fe.Accepted context.feMachine context.initialState
      context.profile (PublicInput.ofSources data) context.feCoins
      certificate.piCcs.output sourceCertificate.fe := by
    exact feAccepted_atSources data certificate.piCcs publicInputBound
      context.feMachine context.initialState context.profile context.feCoins
      certificate.piCcs.output accepted.fe
  have sourcePointEq :
      Transcript.Fe.derive context.feMachine context.initialState
          sourceCertificate.fe =
        Transcript.Fe.derive context.feMachine context.initialState
          certificate.piCcs.fe := by
    exact fePoint_atSources data certificate.piCcs publicInputBound
      context.feMachine context.initialState
  rcases Fe.accepted_implies_truth_or_mismatch_or_badEvent
      context.feMachine context.initialState context.profile data
      context.feCoins certificate.piCcs.output sourceCertificate.fe
      context.challengeSetSize sourceFeAccepted with
    feTruth | yRingMismatch | feBad
  · rcases ncAccepted_implies_truth_or_badEvent noZeroDivisors context data
        certificate accepted.nc with ncTruth | ncBad
    · exact Or.inl ((Semantics.truth_iff_paperHolds data).mp
        ⟨feTruth, ncTruth⟩)
    · exact Or.inr (Or.inr ncBad)
  · apply Or.inr
    apply Or.inl
    unfold YRingUnbound fePoint
    change Polynomial.Fe.OutputMismatch data
      (Transcript.Fe.derive context.feMachine context.initialState
        certificate.piCcs.fe).challengePoint certificate.piCcs.output
    simpa [sourcePointEq] using yRingMismatch
  · apply Or.inr
    apply Or.inr
    apply BadEvent.fe publicInputBound
    simpa [fePoint, sourcePointEq] using feBad

/-- Positive refinement form used by the delayed NIFS composition. The paper
branch carries the exact `yRing` equation; only packed `yZcol` is deferred. -/
theorem accepted_implies_paper_and_yRingBound_or_yRingUnbound_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context)
    (publicInputBound : context.piCcsInput = PublicInput.ofSources data)
    (accepted : Accepted context data certificate) :
    (Semantics.Paper.Holds data ∧ YRingBound context data certificate) ∨
      YRingUnbound context data certificate ∨
      BadEvent context data certificate := by
  by_cases yRing : YRingBound context data certificate
  · rcases accepted_implies_paper_or_yRingUnbound_or_badEvent
        noZeroDivisors context data certificate publicInputBound accepted with
      paper | unbound | bad
    · exact Or.inl ⟨paper, yRing⟩
    · exact False.elim (unbound yRing)
    · exact Or.inr (Or.inr bad)
  · apply Or.inr
    apply Or.inl
    exact yRing

end Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionPiCcs
