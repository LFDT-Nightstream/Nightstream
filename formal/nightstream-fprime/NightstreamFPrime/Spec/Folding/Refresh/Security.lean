import NightstreamFPrime.Spec.Folding.Refresh.Children
import NightstreamFPrime.Spec.Folding.Nifs.PaperSecurityComposition
import NightstreamFPrime.Spec.Folding.PiRLC.NormFreeFork

/-! Knowledge for the refresh fold: SuperNeo v1.2 Lemma 1 (Π_RLC after Π_CCS) composed with
Theorem 13 (Π_DEC) by Lemma 4. When `verify` accepts and the Π_RLC challenges fork, the
extracted fresh witnesses satisfy the CCS and the recomposed children open the input
accumulator at norm below `B`, or a Π_CCS bad event occurred. The children's norm terms are
the norm check that light folds skip.

The fork outputs need no norm bound: Lemma 8 extraction reads only their commitments, public
inputs and evaluations. A chain extracts a refresh output at up to `B + cap · T · (b − 1)`, and
the input still comes back below `B`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Refresh

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources
open NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction
  (ExtractionAlgebra StrongSetUnits CompleteFork extractedAssignment)
open NightstreamFPrime.Spec.Folding.PiRLC.NormFreeFork (normFree normFreeAlgebra normFreeLaws)
open NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive
  (Key Fresh Running piCcsCheck_eq_true_iff_fixedWidthAccepted)
open NightstreamFPrime.Spec.Folding.Nifs.PaperSecurityComposition (piRlcBatch)

universe uCommitment uPublicInput uScalar uState

variable {Commitment : Type uCommitment} {PublicInput : Type uPublicInput}
  {Scalar : Type uScalar} {State : Type uState} {shape : Shape}
  {columns blockCount degreeBound : Nat}

/-- The Π_CCS output witness that a fork of the refresh Π_RLC extracts. -/
def forkWitness
    (key : Key K Commitment PublicInput Scalar State shape columns blockCount degreeBound)
    (running : Running K Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (nifs : Nifs.PaperNonInteractive.Proof K Commitment shape degreeBound)
    (laws : ExtractionAlgebra key.piRlcSemantics key.params key.piRlcAlgebra)
    (strongSet : StrongSetUnits laws.ring key.piRlcAlgebra.challengeValid)
    (fork : CompleteFork (normFree key.piRlcSemantics) key.params
      (normFreeAlgebra key.piRlcAlgebra) (piRlcBatch key running fresh nifs)) :
    OutputWitness shape columns where
  assignments := fun source =>
    extractedAssignment (normFreeLaws laws) strongSet fork
      (Fin.cast key.total_eq_sourceCount.symm source)

theorem forkWitness_ambient
    (key : Key K Commitment PublicInput Scalar State shape columns blockCount degreeBound)
    (running : Running K Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (nifs : Nifs.PaperNonInteractive.Proof K Commitment shape degreeBound)
    (laws : ExtractionAlgebra key.piRlcSemantics key.params key.piRlcAlgebra)
    (strongSet : StrongSetUnits laws.ring key.piRlcAlgebra.challengeValid)
    (fork : CompleteFork (normFree key.piRlcSemantics) key.params
      (normFreeAlgebra key.piRlcAlgebra) (piRlcBatch key running fresh nifs)) :
    AmbientOutputHolds key.extensionOps key.lift key.openingMaps key.params
      (key.statement running fresh) (key.piCcsProbe running fresh nifs)
      (forkWitness key running fresh nifs laws strongSet fork) := by
  intro source
  have extracted := PiRLC.NormFreeFork.correctedAmbientHolds laws strongSet fork
    (Fin.cast key.total_eq_sourceCount.symm source)
  rw [key.ambientAgreement
    ((key.statement running fresh).publicOutput (key.piCcsProbe running fresh nifs) source)
    ((forkWitness key running fresh nifs laws strongSet fork).assignments source) rfl]
  simpa [piRlcBatch, Nifs.PaperNonInteractive.Key.piCcsOutputs, forkWitness] using extracted

/-- SuperNeo v1.2 Lemma 1, Theorem 13 and Lemma 4 for the refresh fold. A fork at the output
stage bound gives the fork here through `NormFreeFork.forgetNorms`. -/
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
    (fork : CompleteFork (normFree key.piRlcSemantics) key.params
      (normFreeAlgebra key.piRlcAlgebra)
      (piRlcBatch key (children key accumulator proof.split) fresh proof.nifs)) :
    let running := children key accumulator proof.split
    let witness := forkWitness key running fresh proof.nifs laws strongSet fork
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
      (forkWitness_ambient key running fresh proof.nifs laws strongSet fork)
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
