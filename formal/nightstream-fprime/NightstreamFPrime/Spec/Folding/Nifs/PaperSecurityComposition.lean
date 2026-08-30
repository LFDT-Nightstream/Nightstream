import NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive.Verifier
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction

/-!
Owns the deterministic PiCCS--PiRLC security composition for the paper NIFS.

The verifier supplies the exact PiCCS probe and the exact PiRLC public input
batch. A complete coordinate fork supplies only successful combined openings.
The theorem constructs the PiCCS output witness from those extracted openings
and concludes source validity or one named PiCCS algebraic failure.

This module does not own a probabilistic forking lemma, Fiat--Shamir security,
commitment binding, PiDEC child extraction, a concrete field, or a backend.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Nifs.PaperSecurityComposition

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction

universe uExtension uCommitment uPublicInput uScalar uState

/-- Exact PiRLC input batch produced by one PiCCS execution. -/
def piRlcBatch
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) :
    InputBatch (RelationSource shape columns blockCount) PublicInput
      (CubePoint Extension shape.cubeVariables)
      (EvaluationFamily Extension shape) Commitment key.params key.arity where
  system := key.relationSource
  point := (key.piCcsExecution running fresh proof).coins.roundPoint
  inputs := key.piCcsOutputs running fresh proof
  sameSystem := by
    intro index
    rfl
  samePoint := by
    intro index
    rfl
  evaluationCount := 1
  evaluationsSize := by
    intro index
    rfl

/-- Assignments extracted by the PiRLC coordinate fork, reindexed into the
exact PiCCS `K+k` source order. -/
def extractedOutputWitness
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound)
    (laws : ExtractionAlgebra key.piRlcSemantics key.params key.piRlcAlgebra)
    (strongSet : StrongSetUnits laws.ring key.piRlcAlgebra.challengeValid)
    (fork : CompleteFork key.piRlcSemantics key.params key.piRlcAlgebra
      (piRlcBatch key running fresh proof)) :
    OutputWitness shape columns where
  assignments := fun source =>
    extractedAssignment laws strongSet fork
      (Fin.cast key.total_eq_sourceCount.symm source)

/-- The PiRLC fork opens the literal PiCCS public outputs in the corrected
ambient relation. No compatibility equality is supplied by the caller. -/
theorem extractedOutputWitness_ambient
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound)
    (laws : ExtractionAlgebra key.piRlcSemantics key.params key.piRlcAlgebra)
    (strongSet : StrongSetUnits laws.ring key.piRlcAlgebra.challengeValid)
    (fork : CompleteFork key.piRlcSemantics key.params key.piRlcAlgebra
      (piRlcBatch key running fresh proof)) :
    AmbientOutputHolds key.extensionOps key.lift key.openingMaps key.params
      (key.statement running fresh) (key.piCcsProbe running fresh proof)
      (extractedOutputWitness key running fresh proof laws strongSet fork) := by
  intro source
  have extracted := completeFork_implies_correctedAmbientHolds
    key.piRlcSemantics key.params key.arity key.piRlcAlgebra laws strongSet
    (piRlcBatch key running fresh proof) fork
      (Fin.cast key.total_eq_sourceCount.symm source)
  rw [key.ambientAgreement
    ((key.statement running fresh).publicOutput
      (key.piCcsProbe running fresh proof) source)
    ((extractedOutputWitness key running fresh proof laws strongSet fork
      ).assignments source) rfl]
  simpa [piRlcBatch, Key.piCcsOutputs, extractedOutputWitness] using extracted

/-- One fork's extracted assignment vector in the exact PiRLC input order. -/
def extractedAssignments
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound)
    (laws : ExtractionAlgebra key.piRlcSemantics key.params key.piRlcAlgebra)
    (strongSet : StrongSetUnits laws.ring key.piRlcAlgebra.challengeValid)
    (fork : CompleteFork key.piRlcSemantics key.params key.piRlcAlgebra
      (piRlcBatch key running fresh proof)) :
    Fin key.arity.total -> PaperLinearAlgebra.Assignment F columns :=
  fun coordinate => extractedAssignment laws strongSet fork coordinate

theorem extractedAssignments_ambient
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound)
    (laws : ExtractionAlgebra key.piRlcSemantics key.params key.piRlcAlgebra)
    (strongSet : StrongSetUnits laws.ring key.piRlcAlgebra.challengeValid)
    (fork : CompleteFork key.piRlcSemantics key.params key.piRlcAlgebra
      (piRlcBatch key running fresh proof)) :
    PiRLC.AmbientOpenings key.piRlcSemantics key.params
      (piRlcBatch key running fresh proof).inputs
      (extractedAssignments key running fresh proof laws strongSet fork) := by
  intro coordinate
  apply (PiRLC.PaperCorrections.correctedAmbientHolds_iff_ceHolds_of_ambient
    key.piRlcSemantics key.params
    (PiRLC.ambientInput ((piRlcBatch key running fresh proof).inputs coordinate))
    (extractedAssignments key running fresh proof laws strongSet fork coordinate)
    rfl).mp
  simpa [PiRLC.PaperCorrections.CorrectedAmbientHolds, PiRLC.ambientInput,
    extractedAssignments] using
      (completeFork_implies_correctedAmbientHolds key.piRlcSemantics key.params
        key.arity key.piRlcAlgebra laws strongSet
        (piRlcBatch key running fresh proof) fork coordinate)

/-- Two weak extractions over the literal same PiCCS output commitment vector
are unique, or they expose the paper's relaxed-binding collision. This is the
same-`phi` joint required by SuperNeo's strong--weak composition theorem. -/
theorem twoForkExtractions_unique_or_relaxedBindingCollision
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound)
    (laws : ExtractionAlgebra key.piRlcSemantics key.params key.piRlcAlgebra)
    (strongSet : StrongSetUnits laws.ring key.piRlcAlgebra.challengeValid)
    (ops : PiRLC.RelaxedBindingOps
      (PaperLinearAlgebra.Assignment F columns) Commitment Scalar)
    (bridge : PiRLC.UniquenessBridge key.piRlcSemantics key.params ops
      (n := key.arity.total))
    (leftFork rightFork : CompleteFork key.piRlcSemantics key.params
      key.piRlcAlgebra (piRlcBatch key running fresh proof)) :
    extractedAssignments key running fresh proof laws strongSet leftFork =
        extractedAssignments key running fresh proof laws strongSet rightFork \/
      exists coordinate, Nonempty
        (PiRLC.RelaxedBindingCollision key.piRlcSemantics key.params ops
          ((piRlcBatch key running fresh proof).inputs coordinate).commitment) := by
  exact PiRLC.same_phi_extractions_unique_or_collision key.piRlcSemantics
    key.params ops bridge (piRlcBatch key running fresh proof).inputs
    (piRlcBatch key running fresh proof).inputs
    (extractedAssignments key running fresh proof laws strongSet leftFork)
    (extractedAssignments key running fresh proof laws strongSet rightFork) rfl
    (extractedAssignments_ambient key running fresh proof laws strongSet leftFork)
    (extractedAssignments_ambient key running fresh proof laws strongSet rightFork)

/-- Accepted NIFS plus a complete PiRLC coordinate fork extracts the exact
PiCCS source relation, or exposes the precise mixing-root or fixed-width
SumCheck failure. -/
theorem accepted_with_completeFork_extracts_source_or_badEvent
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound)
    (result : Running Extension Commitment PublicInput shape)
    (laws : ExtractionAlgebra key.piRlcSemantics key.params key.piRlcAlgebra)
    (strongSet : StrongSetUnits laws.ring key.piRlcAlgebra.challengeValid)
    (fork : CompleteFork key.piRlcSemantics key.params key.piRlcAlgebra
      (piRlcBatch key running fresh proof))
    (accepted : verify key running fresh proof = some result) :
    SourceHolds key.extensionOps key.lift key.openingMaps key.params
        (key.statement running fresh)
        (extractedOutputWitness key running fresh proof laws strongSet fork) \/
      SignedCoefficientObject.MixingRoot key.extensionOps
        (((key.statement running fresh).sourceProtocolData key.lift
          (extractedOutputWitness key running fresh proof laws strongSet fork)
        ).toJointData key.extensionOps)
        (key.piCcsProbe running fresh proof).coins.alpha
        (key.piCcsProbe running fresh proof).coins.gamma \/
      FixedWidthSumCheckFailure key.extensionOps key.lift
        (key.statement running fresh) degreeBound key.challengeSetSize
        (key.piCcsProbe running fresh proof)
        (extractedOutputWitness key running fresh proof laws strongSet fork) := by
  have piCcsChecked : piCcsCheck key running fresh proof = true :=
    (verify_eq_some_iff key running fresh proof result).mp accepted |>.1
  have piCcsAccepted :
      (key.piCcsProbe running fresh proof).FixedWidthAccepted
        key.extensionOps key.lift (key.statement running fresh) degreeBound :=
    (piCcsCheck_eq_true_iff_fixedWidthAccepted key running fresh proof).mp
      piCcsChecked
  exact fixedWidthAcceptedProbe_extracts_source_or_badEvent
    key.baseLaws key.baseZero key.noZeroDivisors key.extensionOps
    key.extensionLaws key.extensionZeroLaws key.lift key.liftLaws
    key.openingMaps key.params key.freshBound (key.statement running fresh)
    key.constantLaw degreeBound
    (key.statement_sumcheckDegreeBound_le running fresh) key.challengeSetSize
    (key.piCcsProbe running fresh proof)
    (extractedOutputWitness key running fresh proof laws strongSet fork)
    (extractedOutputWitness_ambient key running fresh proof laws strongSet fork)
    piCcsAccepted

/-- The base response of an aligned PiRLC fork opens the exact parent used by
the NIFS verifier. -/
theorem baseOutput_eq_parentForChallenges
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound)
    (fork : CompleteFork key.piRlcSemantics key.params key.piRlcAlgebra
      (piRlcBatch key running fresh proof))
    (challenges : Fin key.arity.total -> Scalar)
    (baseChallenges : fork.base.challenges = challenges) :
    fork.base.output key.piRlcAlgebra (piRlcBatch key running fresh proof) =
      key.parentForChallenges running fresh proof challenges := by
  unfold Response.output
  rw [baseChallenges]
  rfl

/-- Complete deterministic SuperNeo composition for one accepted NIFS
message, one aligned PiRLC coordinate fork, and valid openings of every exact
PiDEC child. The success branch contains the source witness, the final child
openings, and exact parent recomposition. -/
theorem accepted_with_fork_and_children_extracts_or_failure
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound)
    (result : Running Extension Commitment PublicInput shape)
    (laws : ExtractionAlgebra key.piRlcSemantics key.params key.piRlcAlgebra)
    (strongSet : StrongSetUnits laws.ring key.piRlcAlgebra.challengeValid)
    (fork : CompleteFork key.piRlcSemantics key.params key.piRlcAlgebra
      (piRlcBatch key running fresh proof))
    (challenges : Fin key.arity.total -> Scalar)
    (sampled : key.piRlcChallenges running fresh proof = some challenges)
    (baseChallenges : fork.base.challenges = challenges)
    (attempt : PiDEC.PaperVerifier.Attempt
      (RelationSource shape columns blockCount) PublicInput
      (CubePoint Extension shape.cubeVariables)
      (EvaluationFamily Extension shape) Commitment key.params)
    (attemptEq : key.piDecAttempt running fresh proof = some attempt)
    (childAssignments : Fin key.params.k ->
      PaperLinearAlgebra.Assignment F columns)
    (childrenValid : forall child,
      CE.Holds key.piRlcSemantics key.params
        (PiDEC.PaperVerifier.children key.piDecPublicInputSplit attempt child)
        (childAssignments child))
    (accepted : verify key running fresh proof = some result) :
    (SourceHolds key.extensionOps key.lift key.openingMaps key.params
        (key.statement running fresh)
        (extractedOutputWitness key running fresh proof laws strongSet fork) /\
      fork.base.assignment =
        key.piDecAlgebra.recomposeAssignment childAssignments /\
      forall child,
        CE.Holds key.piRlcSemantics key.params
          (PiDEC.PaperVerifier.children key.piDecPublicInputSplit attempt child)
          (childAssignments child)) \/
      SignedCoefficientObject.MixingRoot key.extensionOps
        (((key.statement running fresh).sourceProtocolData key.lift
          (extractedOutputWitness key running fresh proof laws strongSet fork)
        ).toJointData key.extensionOps)
        (key.piCcsProbe running fresh proof).coins.alpha
        (key.piCcsProbe running fresh proof).coins.gamma \/
      FixedWidthSumCheckFailure key.extensionOps key.lift
        (key.statement running fresh) degreeBound key.challengeSetSize
        (key.piCcsProbe running fresh proof)
        (extractedOutputWitness key running fresh proof laws strongSet fork) \/
      Nonempty (PiDEC.ParentOpeningBindingCollision key.piRlcSemantics
        key.params attempt.parent.commitment) := by
  rcases accepted_with_completeFork_extracts_source_or_badEvent key running fresh
      proof result laws strongSet fork accepted with
    sourceHolds | mixingFailure | sumcheckFailure
  · have checks := (verify_eq_some_iff key running fresh proof result).mp accepted
    rcases (piDecCheck_eq_true_iff key running fresh proof).mp checks.2.1 with
      ⟨checkedAttempt, checkedAttemptEq, checkedAccepted⟩
    have checkedAttemptSame : checkedAttempt = attempt := by
      exact Option.some.inj (checkedAttemptEq.symm.trans attemptEq)
    subst checkedAttempt
    have expectedAttempt :
        key.piDecAttempt running fresh proof = some
          (key.piDecAttemptForParent proof
            (key.parentForChallenges running fresh proof challenges)) := by
      simp [Key.piDecAttempt, Key.parent, sampled]
    have attemptSame : attempt = key.piDecAttemptForParent proof
        (key.parentForChallenges running fresh proof challenges) := by
      exact Option.some.inj (attemptEq.symm.trans expectedAttempt)
    have parentEq : attempt.parent =
        key.parentForChallenges running fresh proof challenges := by
      rw [attemptSame]
      rfl
    have parentValid : CE.Holds key.piRlcSemantics key.params attempt.parent
        fork.base.assignment := by
      have baseValid := fork.baseSuccess
      unfold Response.Success at baseValid
      rw [baseOutput_eq_parentForChallenges key running fresh proof fork
        challenges baseChallenges] at baseValid
      rw [parentEq]
      exact baseValid
    rcases PiDEC.PaperVerifier.parent_eq_recompose_or_bindingCollision
        key.piRlcSemantics key.params key.piDecAlgebra
        key.piDecPublicInputSplit key.piDecEvaluationArity attempt
        fork.base.assignment childAssignments checkedAccepted parentValid
        childrenValid with parentSame | bindingFailure
    · exact Or.inl ⟨sourceHolds, parentSame, childrenValid⟩
    · exact Or.inr (Or.inr (Or.inr bindingFailure))
  · exact Or.inr (Or.inl mixingFailure)
  · exact Or.inr (Or.inr (Or.inl sumcheckFailure))

/-- A complete PiRLC fork whose base response uses the exact verifier-sampled
challenge vector. -/
structure AlignedFork
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) where
  fork : CompleteFork key.piRlcSemantics key.params key.piRlcAlgebra
    (piRlcBatch key running fresh proof)
  challenges : Fin key.arity.total -> Scalar
  sampled : key.piRlcChallenges running fresh proof = some challenges
  baseChallenges : fork.base.challenges = challenges

/-- Valid openings for every exact child produced by one accepted PiDEC
attempt. -/
structure ChildOpenings
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (attempt : PiDEC.PaperVerifier.Attempt
      (RelationSource shape columns blockCount) PublicInput
      (CubePoint Extension shape.cubeVariables)
      (EvaluationFamily Extension shape) Commitment key.params) where
  assignments : Fin key.params.k -> PaperLinearAlgebra.Assignment F columns
  valid : forall child,
    CE.Holds key.piRlcSemantics key.params
      (PiDEC.PaperVerifier.children key.piDecPublicInputSplit attempt child)
      (assignments child)

/-- The probabilistic PiRLC forking step did not supply a complete aligned
coordinate fork. Probability bounds belong outside this deterministic event. -/
def PiRLCForkingFailure
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) : Prop :=
  Not (Nonempty (AlignedFork key running fresh proof))

/-- The commitment-knowledge boundary did not supply valid openings for all
exact PiDEC children. -/
def PiDECChildOpeningFailure
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (attempt : PiDEC.PaperVerifier.Attempt
      (RelationSource shape columns blockCount) PublicInput
      (CubePoint Extension shape.cubeVariables)
      (EvaluationFamily Extension shape) Commitment key.params) : Prop :=
  Not (Nonempty (ChildOpenings key attempt))

/-- Exhaustive deterministic security outcome of one accepted paper NIFS
message. Cryptographic analyses assign probability only to the failure
constructors. -/
inductive SecurityOutcome
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound)
    (laws : ExtractionAlgebra key.piRlcSemantics key.params key.piRlcAlgebra)
    (strongSet : StrongSetUnits laws.ring key.piRlcAlgebra.challengeValid) : Prop
  | knowledge
      (aligned : AlignedFork key running fresh proof)
      (attempt : PiDEC.PaperVerifier.Attempt
        (RelationSource shape columns blockCount) PublicInput
        (CubePoint Extension shape.cubeVariables)
        (EvaluationFamily Extension shape) Commitment key.params)
      (attemptEq : key.piDecAttempt running fresh proof = some attempt)
      (children : ChildOpenings key attempt)
      (source : SourceHolds key.extensionOps key.lift key.openingMaps key.params
        (key.statement running fresh)
        (extractedOutputWitness key running fresh proof laws strongSet
          aligned.fork))
      (parent : aligned.fork.base.assignment =
        key.piDecAlgebra.recomposeAssignment children.assignments)
  | mixingFailure
      (aligned : AlignedFork key running fresh proof)
      (failure : SignedCoefficientObject.MixingRoot key.extensionOps
        (((key.statement running fresh).sourceProtocolData key.lift
          (extractedOutputWitness key running fresh proof laws strongSet
            aligned.fork)).toJointData key.extensionOps)
        (key.piCcsProbe running fresh proof).coins.alpha
        (key.piCcsProbe running fresh proof).coins.gamma)
  | sumcheckFailure
      (aligned : AlignedFork key running fresh proof)
      (failure : FixedWidthSumCheckFailure key.extensionOps key.lift
        (key.statement running fresh) degreeBound key.challengeSetSize
        (key.piCcsProbe running fresh proof)
        (extractedOutputWitness key running fresh proof laws strongSet
          aligned.fork))
  | parentBindingFailure
      (aligned : AlignedFork key running fresh proof)
      (attempt : PiDEC.PaperVerifier.Attempt
        (RelationSource shape columns blockCount) PublicInput
        (CubePoint Extension shape.cubeVariables)
        (EvaluationFamily Extension shape) Commitment key.params)
      (attemptEq : key.piDecAttempt running fresh proof = some attempt)
      (children : ChildOpenings key attempt)
      (failure : Nonempty (PiDEC.ParentOpeningBindingCollision
        key.piRlcSemantics key.params attempt.parent.commitment))
  | piRlcForkingFailure
      (failure : PiRLCForkingFailure key running fresh proof)
  | piDecChildOpeningFailure
      (attempt : PiDEC.PaperVerifier.Attempt
        (RelationSource shape columns blockCount) PublicInput
        (CubePoint Extension shape.cubeVariables)
        (EvaluationFamily Extension shape) Commitment key.params)
      (attemptEq : key.piDecAttempt running fresh proof = some attempt)
      (failure : PiDECChildOpeningFailure key attempt)

/-- One accepted paper NIFS message has complete extracted source/parent/child
knowledge or one explicit security failure. -/
theorem accepted_implies_securityOutcome
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound)
    (result : Running Extension Commitment PublicInput shape)
    (laws : ExtractionAlgebra key.piRlcSemantics key.params key.piRlcAlgebra)
    (strongSet : StrongSetUnits laws.ring key.piRlcAlgebra.challengeValid)
    (accepted : verify key running fresh proof = some result) :
    SecurityOutcome key running fresh proof laws strongSet := by
  classical
  by_cases forkExists : Nonempty (AlignedFork key running fresh proof)
  · rcases forkExists with ⟨aligned⟩
    have checks := (verify_eq_some_iff key running fresh proof result).mp accepted
    rcases (piDecCheck_eq_true_iff key running fresh proof).mp checks.2.1 with
      ⟨attempt, attemptEq, _attemptAccepted⟩
    by_cases childrenExist : Nonempty (ChildOpenings key attempt)
    · rcases childrenExist with ⟨children⟩
      rcases accepted_with_fork_and_children_extracts_or_failure
          key running fresh proof result laws strongSet aligned.fork
          aligned.challenges aligned.sampled aligned.baseChallenges attempt
          attemptEq children.assignments children.valid accepted with
        knowledge | mixing | sumcheck | binding
      · exact .knowledge aligned attempt attemptEq children knowledge.1
          knowledge.2.1
      · exact .mixingFailure aligned mixing
      · exact .sumcheckFailure aligned sumcheck
      · exact .parentBindingFailure aligned attempt attemptEq children binding
    · exact .piDecChildOpeningFailure attempt attemptEq childrenExist
  · exact .piRlcForkingFailure forkExists

end NightstreamFPrime.Spec.Folding.Nifs.PaperSecurityComposition
