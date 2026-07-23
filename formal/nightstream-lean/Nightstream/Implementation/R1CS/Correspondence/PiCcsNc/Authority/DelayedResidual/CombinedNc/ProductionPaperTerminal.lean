import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPaperChecker
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionTerminal

/-!
Terminal closure of one opening-derived paper step.

The exact terminal checker reads the ordered raw child assignments, checks
their openings, and evaluates all 54 delayed lanes.  This file composes that
result with the claims-level paper checker and retains only genuine algebraic
or commitment-binding events.
-/

namespace Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPaperTerminal

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.DelayedPending

universe uState

variable {shape : SemanticShape}
variable {State : Type uState}
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

/-- The terminal execution closes the final delayed packed output and yields
the independent paper transition. No running authority, child `y_zcol`,
output-unbound case, or implementation-refinement branch occurs. -/
theorem checkedTerminal_implies_packedAndPaper_or_namedFailure
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (carrier : FixedActive.CanonicalOpening.SourceInput.Carrier shape
      publicRingColumns publicFits)
    (context : FixedActive.CanonicalOpening.Context shape State
      publicRingColumns publicFits verifierRows)
    (certificate : FixedActive.Certificate (carrier.install context).full)
    (accepted : ProductionPaperNifs.PaperStepAccepted carrier context
      certificate)
    (rawChildren : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits))
    (terminalAccepted : ProductionTerminal.check
      (carrier.install context).full certificate rawChildren = true) :
    (Terminal.PackedYZcolBoundAtBlock
        (carrier.install context).full.covers carrier.data
        (ProductionPiCcs.ncPoint (carrier.install context).full
          certificate).block
        certificate.piCcs.output ∧
      FixedActive.PaperProfile.Transition
        (FixedActive.paperProfileOf (carrier.install context).full)
        (carrier.install context).full.input
        (outputChildren (carrier.install context).full certificate)) ∨
      ProductionPiCcs.YRingUnbound (carrier.install context).full carrier.data
        certificate ∨
      ProductionPiCcs.BadEvent (carrier.install context).full carrier.data
        certificate ∨
      PiRlcSidecar.MixingCollision (carrier.install context).full.covers
        certificate.piRlcChallenges
        (InputAuthority.productAssignments carrier.data
          (carrier.install context).full.alignment)
        (DelayedProduction.outgoingPending (carrier.install context).full
          certificate).oldBlock
        (PackedYZcol.sourceClaims (carrier.install context).full certificate) ∨
      Nonempty (PiDEC.ParentOpeningBindingCollision
        (semantics (carrier.install context).full.key) productionGlobalParams
        (derive (carrier.install context).full
          certificate).piRlcOutput.commitment) := by
  have terminal := (ProductionTerminal.check_eq_true_iff
    (carrier.install context).full certificate rawChildren).1 terminalAccepted
  rcases
      ProductionTerminal.accepted_of_parentOpening_implies_packedYZcolBound_or_badEvent
        (carrier.install context).full carrier.data certificate
        accepted.canonicalParent accepted.piDecAccepted rawChildren terminal with
    packed | mixing | binding
  · rcases
        ProductionPaperNifs.paperStepAccepted_and_packed_implies_refinement_or_yRingUnbound_or_badEvent
          noZeroDivisors carrier context certificate accepted packed with
      refinement | yRingUnbound | bad
    · exact Or.inl ⟨packed, refinement.toPaperTransition packed⟩
    · exact Or.inr (Or.inl yRingUnbound)
    · exact Or.inr (Or.inr (Or.inl bad))
  · exact Or.inr (Or.inr (Or.inr (Or.inl mixing)))
  · exact Or.inr (Or.inr (Or.inr (Or.inr binding)))

end Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPaperTerminal
