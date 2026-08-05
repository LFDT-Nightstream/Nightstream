import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentityComposition
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentityConcreteExtraction

/-!
Concrete interactive composition for `PaddedRowIdentity`.

Owns: the selected `Pi_CCS` context connected to the canonical Phi81
`Pi_RLC` semantics; the complete concrete `Pi_DEC` context; and the finite
`Pi_CCS -> Pi_RLC -> Pi_DEC` reduction with all semantic algebra inputs
discharged.

Does not own: the analytic `LowNormInvertibility` proof, Ajtai/MSIS binding,
Fiat--Shamir/Poseidon2 security, Rust, R1CS, or matrix artifact bytes.

The final reduction keeps only true security boundaries as premises. The
verifier program, relaxed-binding reduction, and cryptographic security claim
remain explicit because they are not algebraic facts.
-/

set_option autoImplicit false
set_option maxHeartbeats 800000
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteComposition

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
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityComposition
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentitySecurity
open MatrixCoefficientSource
open PaperLinearAlgebra

namespace Algebra
export PaddedRowIdentityConcreteAlgebra
  (AjtaiKey Commitment PublicInput Assignment openingMaps semantics
    semantics_evaluations_size piRlcAlgebra piDecAlgebra publicInputSplit
    evaluationArity ambientAgreement)
end Algebra

namespace Extraction
export PaddedRowIdentityConcreteExtraction
  (extractionAlgebra extractionStrongSetUnits)
end Extraction

/-- The exact selected context with canonical Phi81 semantics. The
`ambientAgreement` theorem connects it to the paper relation at the sole
verifier-owned matrix source. -/
noncomputable def compatibleContext
    (key : Algebra.AjtaiKey)
    (matrices : ApplicationMatrices)
    (commitments : Fin shape.sourceCount -> Algebra.Commitment)
    (publicInputs : Fin shape.sourceCount -> Algebra.PublicInput)
    (priorPoint : CubePoint K rowVariables)
    (claimedCoefficient : CarriedCoordinate shape -> K) :
    CompatibleContext K Algebra.Commitment Algebra.PublicInput RingF shape
      assignmentColumns (Phi81ColumnLayout.blockCount assignmentColumns) where
  piCcs := selectedContext (Algebra.openingMaps key) matrices commitments
    publicInputs priorPoint claimedCoefficient fullChallengeSupport
  arity := PaperProfile.arity
  freshCount_eq := rfl
  runningCount_eq := rfl
  piRlcSemantics := Algebra.semantics key
  ambientAgreement := Algebra.ambientAgreement key matrices
  piRlcEvaluationsSize := Algebra.semantics_evaluations_size key
  piRlcAlgebra := Algebra.piRlcAlgebra key

/-- The concrete selected PiDEC verifier shares exactly the semantics and
parameters of `compatibleContext`. -/
def piDecContext
    (key : Algebra.AjtaiKey)
    (matrices : ApplicationMatrices)
    (commitments : Fin shape.sourceCount -> Algebra.Commitment)
    (publicInputs : Fin shape.sourceCount -> Algebra.PublicInput)
    (priorPoint : CubePoint K rowVariables)
    (claimedCoefficient : CarriedCoordinate shape -> K) :
    PiRlcComposition.PiDec.CompatiblePiDecContext
      (compatibleContext key matrices commitments publicInputs priorPoint
        claimedCoefficient) where
  algebra := Algebra.piDecAlgebra key
  publicSplit := Algebra.publicInputSplit key
  evaluationArity := Algebra.evaluationArity key
  kPositive := by
    change 0 < productionGlobalParams.k
    decide

@[simp] theorem compatibleContext_total
    (key : Algebra.AjtaiKey)
    (matrices : ApplicationMatrices)
    (commitments : Fin shape.sourceCount -> Algebra.Commitment)
    (publicInputs : Fin shape.sourceCount -> Algebra.PublicInput)
    (priorPoint : CubePoint K rowVariables)
    (claimedCoefficient : CarriedCoordinate shape -> K) :
    (compatibleContext key matrices commitments publicInputs priorPoint
      claimedCoefficient).arity.total = 15 := by
  rfl

section Reduction

universe uProverSeed uProverTape

variable
    (key : Algebra.AjtaiKey)
    (matrices : ApplicationMatrices)
    (commitments : Fin shape.sourceCount -> Algebra.Commitment)
    (publicInputs : Fin shape.sourceCount -> Algebra.PublicInput)
    (priorPoint : CubePoint K rowVariables)
    (claimedCoefficient : CarriedCoordinate shape -> K)

local notation "ctx" =>
  compatibleContext key matrices commitments publicInputs priorPoint
    claimedCoefficient

/-- The exact selected reduction statement. The extraction laws and unit
witness remain parameters here so Lean does not expand the full concrete
records inside the dependent game type. -/
def reductionStatement
    {ProverSeed : Type uProverSeed}
    {ProverTape : Type uProverTape}
    (laws : ExtractionAlgebra (ctx).piRlc.semantics
      (ctx).piRlc.params (ctx).piRlc.algebra)
    (strongSet : StrongSetUnits laws.ring
      (ctx).piRlc.algebra.challengeValid)
    (verifier : PiRLC.PaperWeakFiniteUniform.VerifierData (ctx).piRlc)
    (strongAdversaryExpectedPolynomialTime :
      OperationalExperiment.Adversary (ctx).piCcs ProverSeed
        (ForkSeed verifier.alphabet (ctx).arity.total) ProverTape -> Prop)
    (relaxedBindingRoot : Rat) : Prop :=
    ReductionOfKnowledge
      Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.scale
      (Nightstream.SuperNeo.InteractiveReduction.KnowledgeComposition.knowledgeGame
        Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.scale
        (Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition.knowledgeGame
          Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.scale
          (FinitePaperStrong.successGatedFiniteStrongGame (ctx).piCcs
            fullChallengeSupport strongAdversaryExpectedPolynomialTime)
          (PiRLC.PaperWeakFiniteMixture.weakGame
            (Prefix := PrefixSeed K shape ProverSeed)
            laws strongSet verifier)
          (PiRlcComposition.Coupling.operationalCoupling ctx
            laws strongSet
            fullChallengeSupport verifier
            strongAdversaryExpectedPolynomialTime))
        (PiRlcComposition.PiDec.abortingKnowledgeGame
          (piDecContext key matrices commitments publicInputs priorPoint
            claimedCoefficient).paper
          Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.scale)
        (PiRlcComposition.PiDec.operationalCoupling ctx
          (piDecContext key matrices commitments publicInputs priorPoint
            claimedCoefficient)
          laws strongSet
          fullChallengeSupport verifier
          strongAdversaryExpectedPolynomialTime))
      (selectedInteractiveLoss verifier.alphabet.cardinality
        relaxedBindingRoot)

/-- Complete selected finite reduction through PiDEC. Lean constructs the
exact extraction algebra, the strong-set unit witness, and the PiDEC algebra.
The only remaining input is the named Ajtai/MSIS binding boundary for those
constructed values. -/
theorem existsFiniteReductionThroughPiDec
    {ProverSeed : Type uProverSeed}
    {ProverTape : Type uProverTape}
    (theorem8 : Phi81StrongSet.LowNormInvertibility)
    (verifier : PiRLC.PaperWeakFiniteUniform.VerifierData (ctx).piRlc)
    (strongAdversaryExpectedPolynomialTime :
      OperationalExperiment.Adversary (ctx).piCcs ProverSeed
        (ForkSeed verifier.alphabet (ctx).arity.total) ProverTape -> Prop)
    (relaxedBindingRaw relaxedBindingRoot : Rat)
    (rootNonnegative : 0 <= relaxedBindingRoot)
    (rawBinding_le_rootSquare :
      relaxedBindingRaw <= relaxedBindingRoot * relaxedBindingRoot)
    (ops : PiRLC.RelaxedBindingOps Algebra.Assignment Algebra.Commitment RingF) :
    ∃ laws : ExtractionAlgebra (ctx).piRlc.semantics
        (ctx).piRlc.params (ctx).piRlc.algebra,
      ∃ strongSet : StrongSetUnits laws.ring
          (ctx).piRlc.algebra.challengeValid,
        laws = Extraction.extractionAlgebra key ∧
        (PiRLC.PaperForkCollision.RelaxedBindingLaws
            (ctx).piRlc.semantics (ctx).piRlc.params
            (ctx).piRlc.algebra laws ops ->
          PiRLC.PaperWeakFiniteUniform.RelaxedBindingSecurity
              (context := (ctx).piRlc) laws strongSet ops verifier
              relaxedBindingRaw ->
            reductionStatement key matrices commitments publicInputs
              priorPoint claimedCoefficient laws strongSet verifier
              strongAdversaryExpectedPolynomialTime relaxedBindingRoot) := by
  let laws := Extraction.extractionAlgebra key
  let strongSet := Extraction.extractionStrongSetUnits key theorem8
  refine ⟨laws, strongSet, rfl, ?_⟩
  intro bindingLaws binding
  have contracts :=
    selectedNamedSecurityContracts (Algebra.openingMaps key) matrices
      commitments publicInputs priorPoint claimedCoefficient
      fullChallengeSupport strongAdversaryExpectedPolynomialTime
  have reduction :=
    PiRlcComposition.PiDec.finiteReductionOfKnowledge ctx
      (piDecContext key matrices commitments publicInputs priorPoint
        claimedCoefficient)
      laws strongSet
      fullChallengeSupport verifier
      strongAdversaryExpectedPolynomialTime relaxedBindingRaw
      relaxedBindingRoot
      (ratio mixingNumerator fullChallengeSupport.cardinality)
      (ratio sumCheckNumerator fullChallengeSupport.cardinality)
      rootNonnegative rawBinding_le_rootSquare
      (by unfold ratio; positivity)
      (by unfold ratio; positivity)
      ops bindingLaws binding selectedAmbientAdmissible contracts
  simpa [reductionStatement, selectedInteractiveLoss, mixingNumerator,
    sumCheckNumerator, fullChallengeSupport_cardinality] using reduction

end Reduction

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteComposition
