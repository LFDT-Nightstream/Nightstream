import NightstreamFPrime.Spec.Folding.Refresh.Children
import NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive.Completeness
import NightstreamFPrime.Spec.Folding.PiDEC.OutputWitnessConsumer

/-! Completeness of the refresh fold: SuperNeo v1.2 Theorem 13 completeness for the input split,
then Lemmas 10 and 11 for Π_CCS and Π_RLC. An honest prover with an accumulator witness below
`B` and fresh witnesses that satisfy the CCS gets a proof that `verify` accepts, and the output
parent holds below `B`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Refresh

open NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources
open NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive (Key Fresh Running)

universe uCommitment uPublicInput uScalar uState

variable {Commitment : Type uCommitment} {PublicInput : Type uPublicInput}
  {Scalar : Type uScalar} {State : Type uState} {shape : Shape}
  {columns blockCount degreeBound : Nat}

/-- The honest input split: SuperNeo v1.2 Section 7.5 on the accumulator's witness. -/
def honestSplit
    (key : Key K Commitment PublicInput Scalar State shape columns blockCount degreeBound)
    (accumulator : Accumulator Commitment PublicInput shape columns blockCount)
    (assignment : PaperLinearAlgebra.Assignment F columns) : Split Commitment shape where
  commitments := fun running =>
    (PiDEC.PaperVerifier.honestMessages key.piDecAlgebra accumulator assignment
      (Fin.cast key.runningCount_eq_outputCount running)).commitment
  evaluations := fun running =>
    (PiDEC.PaperVerifier.honestMessages key.piDecAlgebra accumulator assignment
      (Fin.cast key.runningCount_eq_outputCount running)).evaluations[0]'(by
        simp [PiDEC.PaperVerifier.honestMessages, key.piRlcEvaluationsSize])

theorem inputAttempt_honestSplit
    (key : Key K Commitment PublicInput Scalar State shape columns blockCount degreeBound)
    (accumulator : Accumulator Commitment PublicInput shape columns blockCount)
    (assignment : PaperLinearAlgebra.Assignment F columns) :
    inputAttempt key accumulator (honestSplit key accumulator assignment) =
      PiDEC.PaperVerifier.honestAttempt key.piDecAlgebra accumulator assignment := by
  unfold inputAttempt PiDEC.PaperVerifier.honestAttempt
  congr 1
  funext child
  have castBack : Fin.cast key.runningCount_eq_outputCount (Fin.cast key.outputCount_eq child) =
      child := Fin.ext rfl
  simp only [honestSplit, castBack]
  congr 1
  obtain ⟨value, equal⟩ := Array.size_eq_one_iff.mp
    (key.piRlcEvaluationsSize accumulator.constraintSystem
      (key.piDecAlgebra.splitAssignment assignment child) accumulator.point)
  simp [PiDEC.PaperVerifier.honestMessages, equal]

def honestWitness
    (key : Key K Commitment PublicInput Scalar State shape columns blockCount degreeBound)
    (assignment : PaperLinearAlgebra.Assignment F columns)
    (freshWitness : Fin shape.freshCount -> PaperLinearAlgebra.Assignment F columns) :
    OutputWitness shape columns where
  assignments := Fin.addCases freshWitness fun running =>
    key.piDecAlgebra.splitAssignment assignment (Fin.cast key.runningCount_eq_outputCount running)

/-- SuperNeo v1.2 Theorem 13 and Lemmas 10 and 11 for the refresh fold. -/
theorem complete
    (key : Key K Commitment PublicInput Scalar State shape columns blockCount degreeBound)
    (accumulator : Accumulator Commitment PublicInput shape columns blockCount)
    (fresh : Fresh Commitment PublicInput shape)
    (assignment : PaperLinearAlgebra.Assignment F columns)
    (freshWitness : Fin shape.freshCount -> PaperLinearAlgebra.Assignment F columns)
    (system : accumulator.constraintSystem = key.relationSource)
    (combined : accumulator.stage = .combined)
    (accumulatorHolds : CE.Holds key.piRlcSemantics key.params accumulator assignment)
    (freshHolds : ∀ source, CCS.Holds
      (paperRelationSemantics key.baseOps key.extensionOps key.lift key.openingMaps) key.params
      (SourceMembership.freshInstance
        (key.statement (children key accumulator (honestSplit key accumulator assignment)) fresh)
        source)
      (freshWitness source)) :
    let running := children key accumulator (honestSplit key accumulator assignment)
    ∃ (messages : Fin shape.cubeVariables → FixedPolynomial K degreeBound)
      (fullOutput : FullOutputCoordinates.FullOutput K shape),
      let coins := FiatShamir.derive key.oracle.transcript
        ({ priorState := key.publicInputState running fresh
           input := (key.statement running fresh).verifierInput key.lift } :
          PiCCS.TranscriptReplay.Statement K State shape)
        { rounds := fun round => (messages round).toMessage }
      ∀ rho : Fin key.arity.total → Scalar,
        key.piRlcResponse (key.absorbPiCcsOutput coins.finalState fullOutput) = some rho →
        ∃ (proof : Proof Commitment shape degreeBound)
          (parent : Accumulator Commitment PublicInput shape columns blockCount)
          (parentWitness : PaperLinearAlgebra.Assignment F columns),
          proof.split = honestSplit key accumulator assignment ∧
          proof.nifs.piCcsRounds = messages ∧
          verify key accumulator fresh proof = some parent ∧
          CE.Holds key.piRlcSemantics key.params parent parentWitness := by
  intro running
  let split := honestSplit key accumulator assignment
  obtain ⟨inputAccepted, childrenHold⟩ := PiDEC.PaperVerifier.complete key.piRlcSemantics
    key.params key.piDecAlgebra key.piDecPublicInputSplit key.piDecEvaluationArity
    accumulator assignment combined accumulatorHolds
  rw [← inputAttempt_honestSplit key accumulator assignment] at inputAccepted childrenHold
  let witness := honestWitness key assignment freshWitness
  have valid : SourceHolds key.extensionOps key.lift key.openingMaps key.params
      (key.statement running fresh) witness := by
    apply (SourceMembership.sourceHolds_iff_memberships key.extensionOps key.extensionLaws
      key.lift key.openingMaps key.params key.freshBound (key.statement running fresh)
      witness).mpr
    constructor
    · intro source
      have value : witness.assignments (freshSourceIndex source) = freshWitness source :=
        addCases_freshSource freshWitness
          (fun running => key.piDecAlgebra.splitAssignment assignment
            (Fin.cast key.runningCount_eq_outputCount running)) source
      rw [value]
      exact freshHolds source
    · intro index
      have value : witness.assignments (runningSourceIndex index) =
          key.piDecAlgebra.splitAssignment assignment
            (Fin.cast key.runningCount_eq_outputCount index) :=
        addCases_runningSource freshWitness
          (fun running => key.piDecAlgebra.splitAssignment assignment
            (Fin.cast key.runningCount_eq_outputCount running)) index
      rw [value, runningInstance_eq_child key accumulator fresh split system index]
      exact (ceHolds_iff key
        (PiDEC.PaperVerifier.children key.piDecPublicInputSplit
          (inputAttempt key accumulator split) (Fin.cast key.runningCount_eq_outputCount index))
        system _).mpr (childrenHold (Fin.cast key.runningCount_eq_outputCount index))
  obtain ⟨messages, fullOutput, _, _, suffix⟩ :=
    Nifs.PaperNonInteractive.Completeness.exists_honest_proof_of_sampler_success key running fresh
      witness valid
  refine ⟨messages, fullOutput, ?_⟩
  intro _ rho sampled
  obtain ⟨nifs, result, childWitnesses, roundsEq, _, challengesEq, verified, outputHolds⟩ :=
    suffix rho sampled
  have checks := (Nifs.PaperNonInteractive.verify_eq_some_iff key running fresh nifs result).mp
    verified
  have parentEq : key.parent running fresh nifs =
      some (key.parentForChallenges running fresh nifs rho) := by
    simp [Nifs.PaperNonInteractive.Key.parent, challengesEq]
  have attemptEq : key.piDecAttempt running fresh nifs =
      some (key.piDecAttemptForParent nifs (key.parentForChallenges running fresh nifs rho)) := by
    simp [Nifs.PaperNonInteractive.Key.piDecAttempt, parentEq]
  have parentHolds := PiDEC.OutputWitnessConsumer.accepted_outputWitness_extracts_parent key
    running fresh nifs result _ attemptEq verified childWitnesses outputHolds
  refine ⟨{ split := split, nifs := nifs }, key.parentForChallenges running fresh nifs rho, _,
    rfl, roundsEq, ?_, parentHolds⟩
  exact (verify_eq_some_iff key accumulator fresh _ _).mpr ⟨inputAccepted, checks.1, parentEq⟩

end NightstreamFPrime.Spec.Folding.Refresh
