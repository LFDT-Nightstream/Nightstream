import NightstreamFPrime.Spec.Folding.Refresh.Children
import NightstreamFPrime.Spec.Folding.Nifs.PaperSecurityComposition

/-! Knowledge for the refresh fold: SuperNeo v1.2 Lemma 1 (Π_RLC after Π_CCS) composed with
Theorem 13 (Π_DEC) by Lemma 4. When `verify` accepts and the Π_RLC challenges fork, the
extracted fresh witnesses satisfy the CCS and the recomposed children open the input
accumulator at norm below `B`, or a Π_CCS bad event occurred. The children's norm terms are
the norm check that light folds skip. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Refresh

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources
open NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction
  (ExtractionAlgebra StrongSetUnits CompleteFork)
open NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive
  (Key Fresh Running piCcsCheck_eq_true_iff_fixedWidthAccepted)
open NightstreamFPrime.Spec.Folding.Nifs.PaperSecurityComposition
  (piRlcBatch extractedOutputWitness extractedOutputWitness_ambient)

universe uCommitment uPublicInput uScalar uState

variable {Commitment : Type uCommitment} {PublicInput : Type uPublicInput}
  {Scalar : Type uScalar} {State : Type uState} {shape : Shape}
  {columns blockCount degreeBound : Nat}

/-- SuperNeo v1.2 Lemma 1, Theorem 13 and Lemma 4 for the refresh fold. -/
theorem accepted_implies_securityOutcome
    (key : Key K Commitment PublicInput Scalar State shape columns blockCount degreeBound)
    (accumulator : Accumulator Commitment PublicInput shape columns blockCount)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Commitment shape degreeBound)
    (parent : Accumulator Commitment PublicInput shape columns blockCount)
    (accepted : verify key accumulator fresh proof = some parent)
    (system : accumulator.constraintSystem = key.relationSource)
    (laws : ExtractionAlgebra key.piRlcSemantics key.params key.piRlcAlgebra)
    (strongSet : StrongSetUnits laws.ring key.piRlcAlgebra.challengeValid)
    (fork : CompleteFork key.piRlcSemantics key.params key.piRlcAlgebra
      (piRlcBatch key (children key accumulator proof.split) fresh proof.nifs)) :
    let running := children key accumulator proof.split
    let witness := extractedOutputWitness key running fresh proof.nifs laws strongSet fork
    ((∀ source, CCS.Holds
          (paperRelationSemantics key.baseOps key.extensionOps key.lift key.openingMaps)
          key.params (SourceMembership.freshInstance (key.statement running fresh) source)
          (witness.assignments (freshSourceIndex source))) ∧
        CE.Holds key.piRlcSemantics key.params accumulator
          (key.piDecAlgebra.recomposeAssignment fun child =>
            witness.assignments (runningSourceIndex (Fin.cast key.outputCount_eq child)))) ∨
      SignedCoefficientObject.MixingRoot key.extensionOps
        (((key.statement running fresh).sourceProtocolData key.lift witness).toJointData
          key.extensionOps)
        (key.piCcsProbe running fresh proof.nifs).coins.alpha
        (key.piCcsProbe running fresh proof.nifs).coins.gamma ∨
      FixedWidthSumCheckFailure key.extensionOps key.lift (key.statement running fresh)
        degreeBound key.challengeSetSize (key.piCcsProbe running fresh proof.nifs) witness := by
  intro running witness
  obtain ⟨inputAccepted, piCcsChecked, _⟩ :=
    (verify_eq_some_iff key accumulator fresh proof parent).mp accepted
  have piCcsAccepted :=
    (piCcsCheck_eq_true_iff_fixedWidthAccepted key running fresh proof.nifs).mp piCcsChecked
  rcases fixedWidthAcceptedProbe_extracts_source_or_badEvent key.baseLaws key.baseZero
      key.noZeroDivisors key.extensionOps key.extensionLaws key.extensionZeroLaws key.lift
      key.liftLaws key.openingMaps key.params key.freshBound (key.statement running fresh)
      key.constantLaw degreeBound (key.statement_sumcheckDegreeBound_le running fresh)
      key.challengeSetSize (key.piCcsProbe running fresh proof.nifs) witness
      (extractedOutputWitness_ambient key running fresh proof.nifs laws strongSet fork)
      piCcsAccepted with
    sourceHolds | mixingRoot | sumCheck
  · left
    obtain ⟨freshHolds, runningHolds⟩ :=
      (SourceMembership.sourceHolds_iff_memberships key.extensionOps key.extensionLaws
        key.lift key.openingMaps key.params key.freshBound (key.statement running fresh)
        witness).mp sourceHolds
    refine ⟨freshHolds, ?_⟩
    apply PiDEC.PaperVerifier.reduce_knowledge key.piRlcSemantics key.params key.piDecAlgebra
      key.piDecPublicInputSplit key.piDecEvaluationArity (inputAttempt key accumulator proof.split)
      _ key.kPositive inputAccepted
    intro child
    have castBack : Fin.cast key.runningCount_eq_outputCount (Fin.cast key.outputCount_eq child) =
        child := Fin.ext rfl
    have holds := runningHolds (Fin.cast key.outputCount_eq child)
    rw [runningInstance_eq_child key accumulator fresh proof.split system, castBack] at holds
    exact (ceHolds_iff key
      (PiDEC.PaperVerifier.children key.piDecPublicInputSplit
        (inputAttempt key accumulator proof.split) child) system _).mp holds
  · exact Or.inr (Or.inl mixingRoot)
  · exact Or.inr (Or.inr sumCheck)

end NightstreamFPrime.Spec.Folding.Refresh
