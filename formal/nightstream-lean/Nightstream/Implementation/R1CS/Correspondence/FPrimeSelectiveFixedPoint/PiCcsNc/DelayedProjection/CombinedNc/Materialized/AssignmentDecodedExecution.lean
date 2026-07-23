import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ProductionMessageAcceptance

/-!
Assignment-first decoding of the materialized production combined-NC claimed
chain.

Owns: one canonical claims-level execution record decoded only from the
independently reconstructed source assignment, and the theorem that literal
selected-row satisfaction implies its complete 25-round claimed chain.

Does not own: identification of the decoded initial value with the
verifier-owned pending state, identification of decoded messages with the
certificate bytes, transcript replay for the decoded challenges, or
identification of the decoded terminal with the verifier-computed public
terminal.  Those are the exact remaining encoder/transcript refinement seam;
they are not premises of the theorem in this leaf.  This module also does not
own raw-child authority, commitment binding, security reductions, costs, or
row removal.

The execution is decoded from
`PhysicalAgreement.reconstructedAssignment`; no external certificate,
`ColumnBindings`, semantic acceptance predicate, digest, or projection claim
is an input.

Emits constraints: none.

Assurance tier: model-level composition over the artifact-checked selected
combined-NC rows for the fixed production profile.
-/

/-!
| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.assignment_decoded_execution` | Transfer decoded source-column values into the materialized execution relation. | derived |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.AssignmentDecodedExecution

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.SumCheck.Finite
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized

/-- The complete verifier-shaped NC claimed chain read from one reconstructed
assignment.  Every field is computed from the exact generated round maps; no
field is copied from an external protocol certificate. -/
structure Execution where
  claimedInitial : K
  roundMessages : List (FixedPolynomial K ProductionRound.degree)
  challenges : List K
  finalClaim : K

/-- Decode the exact fixed-profile NC execution from source-column values. -/
def decode (assignment : Nat → Nat) : Execution where
  claimedInitial := ProductionMessageAcceptance.toConcreteK
    (ClaimedChain.initial RoundMaps.values assignment)
  roundMessages :=
    (ClaimedChain.certificate RoundMaps.values assignment).rounds.map
      (ProductionMessageAcceptance.mapFixedPolynomial
        ProductionMessageAcceptance.toConcreteK)
  challenges :=
    (ClaimedChain.challenges RoundMaps.values assignment).map
      ProductionMessageAcceptance.toConcreteK
  finalClaim := ProductionMessageAcceptance.toConcreteK
    (ClaimedChain.terminal RoundMaps.values assignment)

/-- Internal message acceptance for the assignment-decoded execution.  This
checks only the complete claimed chain, ending at the decoded final-claim
column pair; it deliberately does not equate that pair with a caller-provided
or verifier-computed terminal. -/
def ClaimedChainAccepted (execution : Execution) : Prop :=
  FixedPhase.Chain ConcreteCarrier.extensionOps.toOps
    execution.claimedInitial execution.roundMessages execution.challenges
    execution.finalClaim

/-- The typed source-row consequences establish the complete assignment-first
claimed chain after the explicit carrier transport. -/
theorem consequences_imply_decodedClaimedChainAccepted
    (assignment : Nat → Nat)
    (consequences : SourceRowsSoundness.Consequences assignment) :
    ClaimedChainAccepted (decode assignment) := by
  have transported := ProductionMessageAcceptance.chain_transport
    ClaimedChain.ops ConcreteCarrier.extensionOps.toOps
    ProductionMessageAcceptance.toConcreteK
    ProductionMessageAcceptance.projectionCarrierHomomorphism
    consequences.roundChain
  simpa only [ClaimedChainAccepted, decode] using transported

/-- Literal satisfaction of the exact selected production rows derives the
complete internal NC message execution from the reconstructed assignment.
No external certificate equality or semantic authority premise occurs. -/
theorem generatedEmittedRowsSatisfy_implies_decodedClaimedChainAccepted
    {assignment : Nat → Nat}
    (selectedRows :
      SelectiveArtifactPairs.Artifact.GeneratedEmittedRowsSatisfy assignment)
    (selectorOne : assignment Metadata.steadySelectorColumn = 1)
    (constantOne : assignment 0 = 1) :
    ClaimedChainAccepted
      (decode (PhysicalAgreement.reconstructedAssignment assignment)) := by
  exact consequences_imply_decodedClaimedChainAccepted
    (PhysicalAgreement.reconstructedAssignment assignment)
    (SelectedRowsSoundness.generatedEmittedRowsSatisfy_implies_consequences
      selectedRows selectorOne constantOne)

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.AssignmentDecodedExecution
