import NightstreamFPrime.Spec.Folding.Nifs.PaperSecurityComposition

/-!
SuperNeo v1.1, Theorem 13 and Appendix B.4: consume the valid output witness
of the reduction experiment. The exact returned running product supplies all
child openings, and their radix recomposition opens the accepted parent.

The output witness is supplied by a successful output-relation experiment,
or by a later relation/terminal extractor. Public verifier acceptance alone
does not supply it. No commitment inversion or cryptographic assumption is
used here.
-/

namespace NightstreamFPrime.Spec.Folding.PiDEC.OutputWitnessConsumer

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive
open NightstreamFPrime.Spec.Folding.Nifs.PaperSecurityComposition

universe uExtension uCommitment uPublicInput uScalar uState

variable {Extension : Type uExtension} {Commitment : Type uCommitment}
  {PublicInput : Type uPublicInput} {Scalar : Type uScalar} {State : Type uState}
  {shape : Shape} {columns blockCount degreeBound : Nat}

variable (key : Key Extension Commitment PublicInput Scalar State shape
  columns blockCount degreeBound)

/-- One literal CE(b) coordinate of the returned running product. Structure
and the fresh norm stage come from the verifier key. -/
def runningStatement
    (result : Running Extension Commitment PublicInput shape)
    (index : Fin shape.runningCount) :
    CE.Instance (RelationSource shape columns blockCount) PublicInput
      (CubePoint Extension shape.cubeVariables)
      (EvaluationFamily Extension shape) Commitment where
  constraintSystem := key.relationSource
  commitment := result.commitments index
  publicInput := result.publicInputs index
  point := result.point
  evaluations := #[result.evaluations index]
  stage := .fresh

variable [DecidableEq Extension]
variable (running : Running Extension Commitment PublicInput shape)
  (fresh : Fresh Commitment PublicInput shape)
  (proof : Proof Extension Commitment shape degreeBound)
  (result : Running Extension Commitment PublicInput shape)
  (attempt : PaperVerifier.Attempt (RelationSource shape columns blockCount)
    PublicInput (CubePoint Extension shape.cubeVariables)
    (EvaluationFamily Extension shape) Commitment key.params)

private theorem attemptAccepted
    (attemptEq : key.piDecAttempt running fresh proof = some attempt)
    (accepted : verify key running fresh proof = some result) :
    PaperVerifier.Accepted key.piDecAlgebra key.piDecPublicInputSplit
      key.piDecEvaluationArity attempt := by
  have checked := (verify_eq_some_iff key running fresh proof result).mp accepted
  rcases (piDecCheck_eq_true_iff key running fresh proof).mp checked.2.1 with
    ⟨checkedAttempt, checkedEq, checkedAccepted⟩
  have same : checkedAttempt = attempt :=
    Option.some.inj (checkedEq.symm.trans attemptEq)
  subst checkedAttempt
  exact checkedAccepted

/-- The actual NIFS result and the verifier-produced child have identical
structure, point, public input, commitment, evaluation family, and stage. -/
theorem runningStatement_eq_child
    (attemptEq : key.piDecAttempt running fresh proof = some attempt)
    (accepted : verify key running fresh proof = some result)
    (child : Fin key.params.k) :
    runningStatement key result (Fin.cast key.outputCount_eq child) =
      PaperVerifier.children key.piDecPublicInputSplit attempt child := by
  have checked := attemptAccepted key running fresh proof result attempt
    attemptEq accepted
  have outputEq :=
    (verify_eq_some_iff key running fresh proof result).mp accepted |>.2.2
  have resultEq : result = key.outputForAttempt proof attempt
      (key.piDecPublicInputSplit.split attempt.parent.publicInput) :=
    Option.some.inj (outputEq.symm.trans
      (key.output_eq_some_of_parentBounded running fresh proof attempt
        attemptEq checked.parentBounded))
  rw [resultEq]
  cases sampled : key.piRlcChallenges running fresh proof with
  | none =>
      simp [Key.piDecAttempt, Key.parent, sampled] at attemptEq
  | some challenges =>
      have computed : key.piDecAttempt running fresh proof = some
          (key.piDecAttemptForParent proof
            (key.parentForChallenges running fresh proof challenges)) := by
        simp [Key.piDecAttempt, Key.parent, sampled]
      have same : attempt = key.piDecAttemptForParent proof
          (key.parentForChallenges running fresh proof challenges) :=
        Option.some.inj (attemptEq.symm.trans computed)
      rw [same]
      rfl

/-- Consume the output relation's witness in its actual running-index order.
The fixed production instantiation supplies all 16 children through this
total reindexing; it supplies no prover-selected omission or permutation. -/
def childOpeningsOfOutput
    (attemptEq : key.piDecAttempt running fresh proof = some attempt)
    (accepted : verify key running fresh proof = some result)
    (outputWitness : Fin shape.runningCount →
      PaperLinearAlgebra.Assignment F columns)
    (outputValid : ∀ index,
      CE.Holds key.piRlcSemantics key.params (runningStatement key result index)
        (outputWitness index)) : ChildOpenings key attempt where
  assignments := fun child => outputWitness (Fin.cast key.outputCount_eq child)
  valid := by
    intro child
    rw [← runningStatement_eq_child key running fresh proof result attempt
      attemptEq accepted child]
    exact outputValid (Fin.cast key.outputCount_eq child)

/-- Valid witnesses for the exact returned output rule out the unexplained
child-opening-failure branch of the bare-acceptance security statement. -/
theorem outputWitness_excludes_childOpeningFailure
    (attemptEq : key.piDecAttempt running fresh proof = some attempt)
    (accepted : verify key running fresh proof = some result)
    (outputWitness : Fin shape.runningCount →
      PaperLinearAlgebra.Assignment F columns)
    (outputValid : ∀ index,
      CE.Holds key.piRlcSemantics key.params (runningStatement key result index)
        (outputWitness index)) : ¬ PiDECChildOpeningFailure key attempt := by
  intro failure
  exact failure ⟨childOpeningsOfOutput key running fresh proof result attempt
    attemptEq accepted outputWitness outputValid⟩

/-- Theorem 13's deterministic extraction step: one successful output-relation
execution yields a valid parent opening by radix recomposition. This consumes
the actual NIFS output witness and has no child-opening failure alternative. -/
theorem accepted_outputWitness_extracts_parent
    (attemptEq : key.piDecAttempt running fresh proof = some attempt)
    (accepted : verify key running fresh proof = some result)
    (outputWitness : Fin shape.runningCount →
      PaperLinearAlgebra.Assignment F columns)
    (outputValid : ∀ index,
      CE.Holds key.piRlcSemantics key.params (runningStatement key result index)
        (outputWitness index)) :
    CE.Holds key.piRlcSemantics key.params attempt.parent
      (key.piDecAlgebra.recomposeAssignment fun child =>
        outputWitness (Fin.cast key.outputCount_eq child)) := by
  let openings := childOpeningsOfOutput key running fresh proof result attempt
    attemptEq accepted outputWitness outputValid
  exact PaperVerifier.reduce_knowledge key.piRlcSemantics key.params
    key.piDecAlgebra key.piDecPublicInputSplit key.piDecEvaluationArity attempt
    openings.assignments key.kPositive
    (attemptAccepted key running fresh proof result attempt attemptEq accepted)
    openings.valid

end NightstreamFPrime.Spec.Folding.PiDEC.OutputWitnessConsumer
