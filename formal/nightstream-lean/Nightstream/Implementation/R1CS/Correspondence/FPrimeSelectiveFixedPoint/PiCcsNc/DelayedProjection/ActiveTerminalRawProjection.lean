import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveBoundary

/-!
Terminal row boundary for active delayed-`y_zcol` authority.

This leaf consumes the minimal raw-opening/projection obligations that the
terminal R1CS must derive.  It excludes child `y_zcol` sidecars, public-input
checks, ordinary CCS evaluations, and `y_ring`; those belong to separate
protocol tracks.  The only negative outcomes are the existing canonical
parent, mixing, and parent-opening binding events.

Owns: the terminal semantic reduction from claims-level acceptance plus the
fourteen authoritative raw child openings and delayed projection equality to
the packed predecessor equation or an explicitly typed opening event.

Does not own: generated rows, physical assignment decoding, terminal-CE row
refinement, `y_ring`, transcript generation, or commitment primitives.

Emits constraints: no; the Rust/generated-row bridge instantiates
`ProjectionOpeningAccepted`.

| Stable stage path | Mathematical obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.terminal.raw_openings` | all fourteen raw assignments open the ordered output commitments with the required norms | checked terminal-CE premise |
| `f_prime.pi_ccs_nc.delayed.terminal.projection` | the pending parent vector equals the old-point projection of the radix-recomposed raw children | checked projection premise |
| `f_prime.pi_ccs_nc.delayed.terminal.packed` | accepted claims yield the predecessor packed equation or a named parent-opening event | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveBoundary

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
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

/-- Row-facing terminal anchor for the direct raw-witness projection path.
The premise contains only the fourteen ordered raw commitment openings/norms
and the projection equation derived from their physical terminal rows. -/
theorem claimsAcceptedTerminalRawProjection_implies_packed_or_parentOpeningBadEvent
    [DecidableEq Digest]
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (incomingStateDigest outgoingStateDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (certificate : FixedActive.Certificate
      (ProductionContext.full setup input))
    (accepted : ClaimsAccepted scheme incomingStateDigest outgoingStateDigest
      machine setup input template witnesses certificate)
    (rawChildren : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits))
    (terminal : ProductionTerminal.ProjectionOpeningAccepted
      (ProductionContext.full setup input) certificate rawChildren) :
    Terminal.PackedYZcolBoundAtBlock
        (ProductionContext.full setup input).covers
        (decodedData template witnesses)
        (ProductionPiCcs.ncPoint (ProductionContext.full setup input)
          certificate).block certificate.piCcs.output ∨
      ProductionBoundary.ParentOpeningTerminalBadEvent
        (ProductionContext.full setup input)
        (decodedData template witnesses) certificate := by
  classical
  have claims : ProductionNifs.MessageAccepted
      (ProductionContext.full setup input) certificate :=
    (PackedWitnessProduction.messageCheck_eq_true_iff_accepted
      (ProductionContext.canonical setup input) certificate).mp accepted.nifs
  by_cases parentBound : DelayedRawChildren.CanonicalParentBinding
      (ProductionContext.full setup input) (decodedData template witnesses)
      certificate
  · rcases
        ProductionTerminal.projectionOpeningAccepted_of_parentOpening_implies_packedYZcolBound_or_badEvent
          (ProductionContext.full setup input) (decodedData template witnesses)
          certificate parentBound claims.tail.piDec rawChildren terminal with
      packed | mixing | binding
    · exact Or.inl (by simpa [ProductionPiCcs.ncPoint] using packed)
    · exact Or.inr (.mixing mixing)
    · exact Or.inr (.parentBinding binding)
  · exact Or.inr (.canonicalParentBinding parentBound)

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveBoundary
