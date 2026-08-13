import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentitySecurity
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.HonestCompleteness

/-!
Contract: honest completeness for the fixed-reference `PaddedRowIdentity`
interactive `Pi_CCS` profile.

Owns: specialization of the paper-joint uniform honest prover to the exact
24-round, degree-nine selected context; conversion from the direct logical
selective-CCS source relation to the connected paper source relation; and
perfect acceptance with a corrected-ambient output witness for every public
coin vector.

Does not own: Fiat--Shamir, Poseidon2, commitment binding, production matrix
artifacts, Rust, R1CS, or circuit correspondence.

Emits constraints: no.

Assurance tier: model-level reference snapshot. The result is honest
completeness for every typed matrix family of these fixed dimensions. It does
not select a verifier-key relation artifact.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityCompleteness

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.HonestCompleteness
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentity
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentitySoundness
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentitySecurity

universe uCommitment uPublicInput

/-- A direct logical source witness constructs one strategy before the
verifier chooses any coins. Every public-coin execution accepts, and the same
authoritative witness opens the verifier-computed corrected-ambient output. -/
theorem exists_selected_uniform_honestStrategy
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    (openingMaps : OpeningMaps Commitment PublicInput assignmentColumns)
    (matrices : ApplicationMatrices)
    (commitments : Fin shape.sourceCount -> Commitment)
    (publicInputs : Fin shape.sourceCount -> PublicInput)
    (priorPoint : CubePoint K rowVariables)
    (claimedCoefficient : CarriedCoordinate shape -> K)
    (witness : OutputWitness shape assignmentColumns)
    (source : LogicalSourceHolds openingMaps productionGlobalParams matrices
      commitments publicInputs priorPoint claimedCoefficient witness) :
    exists strategy : Strategy K shape PUnit,
      forall coins : PublicCoins K shape,
        AmbientSuccess
          (selectedContext openingMaps matrices commitments publicInputs
            priorPoint claimedCoefficient fullChallengeSupport)
          (attachWitness (execute strategy PUnit.unit coins)
            (some witness)) := by
  let context :=
    selectedContext openingMaps matrices commitments publicInputs priorPoint
      claimedCoefficient fullChallengeSupport
  have connectedSource :
      SourceHolds extensionOps K.embed openingMaps productionGlobalParams
        (statement matrices commitments publicInputs priorPoint
          claimedCoefficient) witness :=
    (sourceHolds_iff_logicalSourceHolds openingMaps productionGlobalParams
      matrices commitments publicInputs priorPoint claimedCoefficient
      witness).2 source
  have contextSource :
      SourceHolds context.extensionOps context.lift context.openingMaps
        context.params context.statement witness := by
    simpa [context, selectedContext] using connectedSource
  have ambientAdmissible :
      context.params.b <=
        PiRLC.PaperCorrections.correctedAmbientBoundFor context.params := by
    change
      productionGlobalParams.b <=
        PiRLC.PaperCorrections.correctedAmbientBoundFor
          productionGlobalParams
    decide
  exact exists_uniform_honestStrategy context ambientAdmissible witness
    contextSource

/-- Executable Boolean-check form of selected perfect completeness. -/
theorem exists_selected_uniform_honestStrategy_check
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    (openingMaps : OpeningMaps Commitment PublicInput assignmentColumns)
    (matrices : ApplicationMatrices)
    (commitments : Fin shape.sourceCount -> Commitment)
    (publicInputs : Fin shape.sourceCount -> PublicInput)
    (priorPoint : CubePoint K rowVariables)
    (claimedCoefficient : CarriedCoordinate shape -> K)
    (witness : OutputWitness shape assignmentColumns)
    (source : LogicalSourceHolds openingMaps productionGlobalParams matrices
      commitments publicInputs priorPoint claimedCoefficient witness) :
    exists strategy : Strategy K shape PUnit,
      forall coins : PublicCoins K shape,
        ambientCheck
          (selectedContext openingMaps matrices commitments publicInputs
            priorPoint claimedCoefficient fullChallengeSupport)
          (attachWitness (execute strategy PUnit.unit coins)
            (some witness)) = true := by
  rcases exists_selected_uniform_honestStrategy openingMaps matrices
      commitments publicInputs priorPoint claimedCoefficient witness source
      with ⟨strategy, complete⟩
  exact ⟨strategy, fun coins =>
    (ambientCheck_eq_true_iff _ _).2 (complete coins)⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityCompleteness
