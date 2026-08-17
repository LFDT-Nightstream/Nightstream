import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentitySecurity
import Nightstream.SuperNeo.Folding.Nifs.PaperProfile
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcComposition.PiDec

/-!
Contract: finite interactive composition for the selected one-joint profile.

Owns: the exact `1 + 14` source adapter from selected `Pi_CCS` into paper
`Pi_RLC`; construction of both selected `Pi_CCS` probability contracts from
root counting; the exact 16-coordinate-fork numerator; and the complete
`Pi_CCS -> Pi_RLC -> Pi_DEC` reduction with zero added `Pi_DEC` loss.

Does not own: low-norm invertibility in the Phi81 quotient ring, Ajtai/MSIS
binding, Fiat--Shamir or Poseidon2 security, Rust, R1CS, artifacts, or outer
proof security. These boundaries remain explicit inputs to the final theorem.

Emits constraints: no.

Assurance tier: security reduction. No selected `Pi_CCS` algebraic security
contract is an assumption in this file.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityComposition

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcBatch
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcComposition
open Nightstream.SuperNeo.Folding.PiRLC.PaperForkExtraction
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteUniform
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.SuperNeo.InteractiveReduction.Paper
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentity
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentitySecurity
open MatrixCoefficientSource
open PaperLinearAlgebra

universe uCommitment uPublicInput

/-- Exact selected `Pi_CCS` context connected to one caller-supplied
`Pi_RLC` algebra over the production Phi81 challenge ring. The algebra is an
explicit boundary because its concrete extraction laws and commitment
security are separate obligations. -/
noncomputable def selectedCompatibleContext
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    (openingMaps : OpeningMaps Commitment PublicInput assignmentColumns)
    (matrices : ApplicationMatrices)
    (commitments : Fin shape.sourceCount -> Commitment)
    (publicInputs : Fin shape.sourceCount -> PublicInput)
    (priorPoint : CubePoint K rowVariables)
    (claimedCoefficient : CarriedCoordinate shape -> K)
    (piRlcAlgebra : PiRLC.Algebra
      (MatrixSource F shape assignmentColumns
        (Phi81ColumnLayout.blockCount assignmentColumns))
      (Assignment F assignmentColumns)
      PublicInput
      (CubePoint K shape.cubeVariables)
      (EvaluationFamily K shape)
      Commitment RingF
      (paperRelationSemantics baseOps extensionOps K.embed openingMaps)
      productionGlobalParams) :
    CompatibleContext K Commitment PublicInput RingF shape assignmentColumns
      (Phi81ColumnLayout.blockCount assignmentColumns) where
  piCcs := selectedContext openingMaps matrices commitments publicInputs
    priorPoint claimedCoefficient fullChallengeSupport
  arity := PaperProfile.arity
  freshCount_eq := rfl
  runningCount_eq := rfl
  piRlcSemantics :=
    paperRelationSemantics baseOps extensionOps K.embed openingMaps
  ambientAgreement := by
    intro statement assignment sourceEq
    exact Iff.rfl
  piRlcEvaluationsSize := fun _ _ _ => rfl
  piRlcAlgebra := piRlcAlgebra

@[simp] theorem selectedCompatibleContext_total
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    (openingMaps : OpeningMaps Commitment PublicInput assignmentColumns)
    (matrices : ApplicationMatrices)
    (commitments : Fin shape.sourceCount -> Commitment)
    (publicInputs : Fin shape.sourceCount -> PublicInput)
    (priorPoint : CubePoint K rowVariables)
    (claimedCoefficient : CarriedCoordinate shape -> K)
    (piRlcAlgebra : PiRLC.Algebra
      (MatrixSource F shape assignmentColumns
        (Phi81ColumnLayout.blockCount assignmentColumns))
      (Assignment F assignmentColumns)
      PublicInput
      (CubePoint K shape.cubeVariables)
      (EvaluationFamily K shape)
      Commitment RingF
      (paperRelationSemantics baseOps extensionOps K.embed openingMaps)
      productionGlobalParams) :
    (selectedCompatibleContext openingMaps matrices commitments publicInputs
      priorPoint claimedCoefficient piRlcAlgebra).arity.total = 17 := by
  rfl

/-- The production fresh norm bound is inside the corrected ambient relation.
This numeric side condition is proved, not accepted from the caller. -/
theorem selectedAmbientAdmissible :
    productionGlobalParams.b <=
      PiRLC.PaperCorrections.correctedAmbientBoundFor
        productionGlobalParams := by
  decide

/-- Exact ideal-interactive loss after `Pi_CCS`, the 15-coordinate `Pi_RLC`
fork, and exact `Pi_DEC`. The only symbolic term is the square-root envelope
for relaxed Ajtai binding. -/
def selectedInteractiveLoss
    (piRlcAlphabetCardinality : Nat)
    (relaxedBindingRoot : Rat) : Rat :=
  ratio 16 piRlcAlphabetCardinality +
    ((ratio 10599 (goldilocksP * goldilocksP) +
      ratio 216 (goldilocksP * goldilocksP)) + relaxedBindingRoot)

theorem selectedInteractiveLoss_explicit
    (piRlcAlphabetCardinality : Nat)
    (relaxedBindingRoot : Rat) :
    selectedInteractiveLoss piRlcAlphabetCardinality relaxedBindingRoot =
      ratio 16 piRlcAlphabetCardinality +
        ((ratio 10599 (goldilocksP * goldilocksP) +
          ratio 216 (goldilocksP * goldilocksP)) +
            relaxedBindingRoot) := by
  rfl

section SelectedComposition

universe uProverSeed uProverTape

variable
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    (openingMaps : OpeningMaps Commitment PublicInput assignmentColumns)
    (matrices : ApplicationMatrices)
    (commitments : Fin shape.sourceCount -> Commitment)
    (publicInputs : Fin shape.sourceCount -> PublicInput)
    (priorPoint : CubePoint K rowVariables)
    (claimedCoefficient : CarriedCoordinate shape -> K)
    (piRlcAlgebra : PiRLC.Algebra
      (MatrixSource F shape assignmentColumns
        (Phi81ColumnLayout.blockCount assignmentColumns))
      (Assignment F assignmentColumns)
      PublicInput
      (CubePoint K shape.cubeVariables)
      (EvaluationFamily K shape)
      Commitment RingF
      (paperRelationSemantics baseOps extensionOps K.embed openingMaps)
      productionGlobalParams)

local notation "selected" =>
  selectedCompatibleContext openingMaps matrices commitments publicInputs
    priorPoint claimedCoefficient piRlcAlgebra

/-- Complete selected-profile finite interactive reduction through `Pi_DEC`.

The theorem constructs the two `Pi_CCS` soundness contracts by root counting.
Its remaining premises are the exact boundaries that cannot be obtained from
the joint polynomial alone: Phi81 extraction algebra and strong-set units,
the executable `Pi_RLC` verifier, relaxed Ajtai binding, and the independently
typed `Pi_DEC` algebra. -/
theorem selectedFiniteReductionThroughPiDec
    {ProverSeed : Type uProverSeed}
    {ProverTape : Type uProverTape}
    (piDec : PiRlcComposition.PiDec.CompatiblePiDecContext selected)
    (laws : ExtractionAlgebra (selected).piRlc.semantics
      (selected).piRlc.params (selected).piRlc.algebra)
    (strongSet : StrongSetUnits laws.ring
      (selected).piRlc.algebra.challengeValid)
    (verifier :
      PiRLC.PaperWeakFiniteUniform.VerifierData (selected).piRlc)
    (strongAdversaryExpectedPolynomialTime :
      OperationalExperiment.Adversary (selected).piCcs ProverSeed
        (ForkSeed verifier.alphabet (selected).arity.total) ProverTape -> Prop)
    (relaxedBindingRaw relaxedBindingRoot : Rat)
    (rootNonnegative : 0 <= relaxedBindingRoot)
    (rawBinding_le_rootSquare :
      relaxedBindingRaw <= relaxedBindingRoot * relaxedBindingRoot)
    (ops : PiRLC.RelaxedBindingOps
      (Assignment F assignmentColumns) Commitment RingF)
    (bindingLaws :
      PiRLC.PaperForkCollision.RelaxedBindingLaws
        (selected).piRlc.semantics (selected).piRlc.params
        (selected).piRlc.algebra laws ops)
    (binding : PiRLC.PaperWeakFiniteUniform.RelaxedBindingSecurity
      laws strongSet ops verifier relaxedBindingRaw) :
    ReductionOfKnowledge
      Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.scale
      (Nightstream.SuperNeo.InteractiveReduction.KnowledgeComposition.knowledgeGame
        Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.scale
        (Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition.knowledgeGame
          Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.scale
          (FinitePaperStrong.successGatedFiniteStrongGame (selected).piCcs
            fullChallengeSupport strongAdversaryExpectedPolynomialTime)
          (PiRLC.PaperWeakFiniteMixture.weakGame
            (Prefix := PrefixSeed K shape ProverSeed)
            laws strongSet verifier)
          (PiRlcComposition.Coupling.operationalCoupling selected laws strongSet
            fullChallengeSupport verifier
            strongAdversaryExpectedPolynomialTime))
        (PiRlcComposition.PiDec.abortingKnowledgeGame piDec.paper
          Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.scale)
        (PiRlcComposition.PiDec.operationalCoupling selected piDec laws
          strongSet fullChallengeSupport verifier
          strongAdversaryExpectedPolynomialTime))
      (selectedInteractiveLoss verifier.alphabet.cardinality
        relaxedBindingRoot) := by
  have contracts :=
    selectedNamedSecurityContracts openingMaps matrices commitments publicInputs
      priorPoint claimedCoefficient fullChallengeSupport
      strongAdversaryExpectedPolynomialTime
  have reduction :=
    PiRlcComposition.PiDec.finiteReductionOfKnowledge selected piDec laws
      strongSet fullChallengeSupport verifier
      strongAdversaryExpectedPolynomialTime relaxedBindingRaw
      relaxedBindingRoot
      (ratio mixingNumerator fullChallengeSupport.cardinality)
      (ratio sumCheckNumerator fullChallengeSupport.cardinality)
      rootNonnegative rawBinding_le_rootSquare
      (by
        unfold ratio
        positivity)
      (by
        unfold ratio
        positivity)
      ops bindingLaws binding selectedAmbientAdmissible contracts
  simpa [selectedInteractiveLoss, mixingNumerator, sumCheckNumerator,
    fullChallengeSupport_cardinality] using reduction

end SelectedComposition

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityComposition
