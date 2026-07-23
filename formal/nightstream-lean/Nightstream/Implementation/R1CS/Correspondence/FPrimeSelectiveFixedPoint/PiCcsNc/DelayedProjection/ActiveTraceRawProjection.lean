import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveTrace
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveTerminalRawProjection

/-!
Finite active-trace composition from the direct raw terminal projection.

The terminal witness obligation is the minimal row-facing
`ProjectionOpeningAccepted`: fourteen ordered raw commitment openings/norms
and one direct projection from those same assignments.  Backward induction
then closes every preceding delayed `y_zcol` output exactly one fold later.
No child sidecar, generic output-unbound proposition, or implementation
refinement failure occurs in this track.

Owns: the nonempty trace contract, explicit no-pending base boundary, final
raw-opening anchor, and backward one-fold propagation of every predecessor
packed `y_zcol` equation.

Does not own: generated terminal rows, physical assignment decoding, Rust
execution, the independent paper/`y_ring` proof, transcript primitives, or
commitment-scheme internals.

Emits constraints: no; proof-only trace composition.

| Stable stage path | Mathematical obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.trace.raw_terminal` | the final step carries one authoritative raw opening and projection anchor | checked terminal premise |
| `f_prime.pi_ccs_nc.delayed.trace.raw_edge` | a packed successor closes the preceding pending equation exactly one fold later | derived / named-event partition |
| `f_prime.pi_ccs_nc.delayed.trace.raw_base` | the first step has no incoming pending value and satisfies ordinary NC | checked and derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveTrace

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.Protocol.FPrime.ConcretePhi81
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator
open Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.DelayedPending
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc
open PackedWitness

universe uOuterKey uAppState uWitness uDigest uTranscriptState uEncoding

variable {OuterKey : Type uOuterKey}
variable {AppState : Type uAppState}
variable {Witness : Type uWitness}
variable {Digest : Type uDigest}
variable {TranscriptState : Type uTranscriptState}
variable {Encoding : Type uEncoding}
variable {shape : SemanticShape}
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

namespace Trace

/-- Minimal terminal rows decoded at the final trace step. -/
def TerminalRawProjectionChecked
    [DecidableEq Digest]
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1}
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1}
    {incoming outgoing : Digest}
    (trace : Trace scheme machine setup incoming outgoing) : Prop :=
  match trace with
  | .single step =>
      ∃ rawChildren : Fin productionGlobalParams.k ->
          Phi81Relation.Assignment
            (RelationShape shape publicRingColumns publicFits),
        ProductionTerminal.ProjectionOpeningAccepted
          (ProductionContext.full setup step.input) step.certificate rawChildren
  | .cons _ tail => tail.TerminalRawProjectionChecked

private theorem terminalRawProjection_implies_allPacked_or_failure
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    {incoming outgoing : Digest}
    (trace : Trace scheme machine setup incoming outgoing)
    (terminal : trace.TerminalRawProjectionChecked) :
    trace.AllPacked ∨ trace.ParentOpeningFailure := by
  induction trace with
  | single step =>
      rcases terminal with ⟨rawChildren, terminalRows⟩
      rcases
          ActiveBoundary.claimsAcceptedTerminalRawProjection_implies_packed_or_parentOpeningBadEvent
            scheme _ _ machine setup step.input step.template step.witnesses
            step.certificate step.accepted rawChildren terminalRows with
        packed | bad
      · exact Or.inl packed
      · exact Or.inr bad
  | cons head tail inductionHypothesis =>
      rcases inductionHypothesis terminal with tailPacked | tailFailure
      · let next := tail.headStep
        have nextPacked : next.2.Packed := by
          cases tail with
          | single step => exact tailPacked
          | cons step rest => exact tailPacked.1
        rcases
            ActiveBoundary.claimsAcceptedPair_of_nextPacked_implies_previousPacked_or_parentOpeningBadEvent
              noZeroDivisors scheme _ _ next.1 machine setup head.input
              head.template head.witnesses head.certificate head.accepted
              next.2.input next.2.template next.2.witnesses
              next.2.certificate next.2.accepted nextPacked with
          packed | bad | binding
        · exact Or.inl ⟨packed, tailPacked⟩
        · exact Or.inr (Or.inl (Or.inl bad))
        · exact Or.inr (Or.inl (Or.inr binding))
      · exact Or.inr (Or.inr tailFailure)

/-- Complete delayed-`y_zcol` trace theorem from the physical terminal
opening/projection obligation.  The no-pending base is explicit and the final
raw opening closes the last pending output before induction proceeds
backward. -/
theorem terminalRawProjection_implies_baseAndAllPacked_or_parentOpeningFailure
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    {incoming outgoing : Digest}
    (trace : Trace scheme machine setup incoming outgoing)
    (base : trace.BaseBoundary)
    (terminal : trace.TerminalRawProjectionChecked) :
    (trace.BaseNc ∧ trace.AllPacked) ∨ trace.ParentOpeningFailure := by
  rcases terminalRawProjection_implies_allPacked_or_failure
      noZeroDivisors scheme machine setup trace terminal with
    allPacked | failure
  · let first := trace.headStep
    have headPacked : first.2.Packed := by
      cases trace with
      | single step => exact allPacked
      | cons step tail => exact allPacked.1
    have baseNc : first.2.BaseNc :=
      ActiveBoundary.claimsAcceptedBase_of_packed_implies_ordinaryNc
        scheme _ _ machine setup first.2.input first.2.template
        first.2.witnesses first.2.certificate first.2.accepted headPacked base
    exact Or.inl ⟨baseNc, allPacked⟩
  · exact Or.inr failure

end Trace

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveTrace
