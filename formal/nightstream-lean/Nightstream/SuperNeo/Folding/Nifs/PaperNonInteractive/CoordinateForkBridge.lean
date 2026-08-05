import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.RandomOracleBoundary
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Soundness
import Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction

/-!
Coordinate-fork bridge for the paper non-interactive NIFS.

Source: SuperNeo Section 7.4, Appendix C Theorem 10, Appendix D.5, and
HyperNova Section 3.

Owns: the exact `Pi_RLC` context and input batch reached after the typed
`Pi_CCS` replay; an outcome type whose adversary batch is definitionally
aligned with that NIFS batch; the exact coordinate-programming receipt; six
concrete Fiat--Shamir event predicates; conversion of an accepted coordinate
fork into the corrected ambient opening required by NIFS soundness; and the
structural refinement of deterministic NIFS soundness into a transition, one
of five residual interactive paper events, or the concrete oracle-programming
failure.

Does not own: an oracle-programming probability theorem, a distribution on
fork outcomes, any of the six event bounds, a Poseidon2 encoding, commitment
security, Rust, R1CS, artifacts, minimality, or costs.

Emits constraints: no.

The multi-fork predicate is not an alias for failure of the desired NIFS
transition or failure of an accepted coordinate fork. It says only that the
concrete outcome lacks a complete coordinate-programming receipt. Once a
receipt exists, rejection of the complete fork is separately owned by the
interactive `Pi_RLC` fork-sampling event. This prevents the same fork loss from
being charged once to the interactive budget and again to Fiat--Shamir.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open Nightstream.SuperNeo.Folding.PiRLC.PaperForkExtraction
open Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking
open Nightstream.SuperNeo.InteractiveReduction.FiatShamirContract
open MatrixCoefficientSource
open PaperLinearAlgebra

universe uExtension uCommitment uPublicInput uScalar uState

namespace Key

/-- The literal paper `Pi_RLC` context selected by the NIFS key. -/
def nifsPiRlcContext
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound) :
    PiRLC.PaperCompleteness.Context
      (MatrixSource F shape columns blockCount)
      (Assignment F columns)
      PublicInput
      (CubePoint Extension shape.cubeVariables)
      (EvaluationFamily Extension shape)
      Commitment
      Scalar where
  semantics := key.semantics
  params := key.params
  arity := key.arity
  algebra := key.piRlcAlgebra
  evaluationCount := fun _ => 1
  evaluationsSize := key.piRlcEvaluationsSize

/-- The authoritative `Pi_RLC` batch is exactly the coefficient-complete
public output of the preceding `Pi_CCS` replay. -/
def nifsPiRlcBatch
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
    InputBatch
      (MatrixSource F shape columns blockCount)
      PublicInput
      (CubePoint Extension shape.cubeVariables)
      (EvaluationFamily Extension shape)
      Commitment
      key.params
      key.arity where
  system := key.matrixSource
  point := (key.piCcsExecution running fresh proof).coins.roundPoint
  inputs := key.piCcsOutputs running fresh proof
  sameSystem := fun _ => rfl
  samePoint := fun _ => rfl
  evaluationCount := 1
  evaluationsSize := fun _ => rfl

@[simp] theorem nifsPiRlcBatch_input
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
    (source : Fin key.arity.total) :
    (key.nifsPiRlcBatch running fresh proof).inputs source =
      key.piCcsOutputs running fresh proof source := by
  rfl

end Key

/-- One candidate output of the Fiat--Shamir coordinate-fork experiment.
`batchAligned` prevents the fork oracle from changing the `Pi_CCS` prefix or
its coefficient-complete output batch. -/
structure AlignedForkOutcome
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound) where
  running : Running Extension Commitment PublicInput shape
  fresh : Fresh Commitment PublicInput shape
  proof : Proof Extension Commitment shape degreeBound
  result : Running Extension Commitment PublicInput shape
  adversary : PiRLC.PaperWeakReduction.Adversary key.nifsPiRlcContext
  sample : ForkSample Scalar key.arity.total
  batchAligned :
    adversary.batch = key.nifsPiRlcBatch running fresh proof

/-- Exact receipt for programming one base challenge vector and one valid,
coordinate-aligned alternative at every `Pi_RLC` coordinate. -/
structure CoordinateProgrammingReceipt
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (outcome : AlignedForkOutcome key) : Prop where
  baseAligned :
    outcome.sample.base =
      key.piRlcChallenges outcome.running outcome.fresh outcome.proof
  baseValid : forall index,
    key.piRlcAlgebra.challengeValid (outcome.sample.base index)
  forkValid : forall coordinate index,
    key.piRlcAlgebra.challengeValid
      (outcome.sample.forks coordinate index)
  agreeExcept : forall coordinate index, index ≠ coordinate ->
    outcome.sample.base index = outcome.sample.forks coordinate index
  changed : forall coordinate,
    outcome.sample.base coordinate ≠
      outcome.sample.forks coordinate coordinate

/-- The exact Fiat--Shamir programming failure: the outcome does not carry the
base-aligned, coordinate-wise reprogramming receipt. It says nothing about
whether a correctly programmed fork is accepted. -/
def MultiForkProgrammingFailure
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (outcome : AlignedForkOutcome key) : Prop :=
  ¬ CoordinateProgrammingReceipt outcome

/-- The intrinsic interactive `Pi_RLC` fork-sampling failure. A complete
programming receipt exists, but the literal paper verifier does not accept the
base plus all coordinate forks. -/
def CoordinateForkSamplingFailure
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (outcome : AlignedForkOutcome key) : Prop :=
  CoordinateProgrammingReceipt outcome ∧
    ¬ PiRLC.PaperWeakReduction.AcceptedFork key.nifsPiRlcContext
      outcome.adversary outcome.sample

/-- The six concrete Fiat--Shamir predicates for one aligned NIFS fork
outcome. The first five are the exact typed transcript predicates; the sixth
is `MultiForkProgrammingFailure`, not an arbitrary caller predicate. -/
def nifsEventPredicates
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound) :
    EventPredicates (AlignedForkOutcome key) where
  publicInputBindingCollision := fun outcome =>
    exists otherRunning otherFresh,
      PublicInputBindingCollision key outcome.running outcome.fresh
        otherRunning otherFresh
  transcriptReplayCollision := fun outcome =>
    exists other,
      ProtocolVerifier.TranscriptReplayCollision key.oracle
        (piCcsReplayInput key outcome.running outcome.fresh outcome.proof)
        other
  transcriptStateCollision := fun outcome =>
    exists other,
      ProtocolVerifier.TranscriptStateCollision key.oracle
        (piCcsReplayInput key outcome.running outcome.fresh outcome.proof)
        other
  outputAbsorptionCollision := fun outcome =>
    exists otherState otherOutput,
      FullOutputAbsorptionCollision key
        ((piCcsReplayInput key outcome.running outcome.fresh outcome.proof).derive
          key.oracle).finalState
        otherState
        outcome.proof.piCcsOutput
        otherOutput
  challengeSamplingFailure := fun outcome =>
    PiRlcSamplingSetFailure key outcome.running outcome.fresh outcome.proof
  multiForkProgrammingFailure := MultiForkProgrammingFailure

/-- A typed transcript event inhabits exactly its corresponding concrete
predicate in the six-event experiment. -/
theorem transcriptSecurityEvent_implies_eventPredicate
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (outcome : AlignedForkOutcome key)
    (event : TranscriptSecurityEvent key outcome.running outcome.fresh
      outcome.proof) :
    (nifsEventPredicates key).at event.securityClass outcome := by
  cases event with
  | publicInputBinding otherRunning otherFresh collision =>
      exact ⟨otherRunning, otherFresh, collision⟩
  | replayChallenge other collision =>
      exact ⟨other, collision⟩
  | replayState other collision =>
      exact ⟨other, collision⟩
  | outputAbsorption otherState otherMessage collision =>
      exact ⟨otherState, otherMessage, collision⟩
  | piRlcSamplingSet failure =>
      exact failure

/-- An accepted coordinate fork for the NIFS-derived batch extracts the
complete corrected-ambient witness required by deterministic NIFS soundness. -/
theorem acceptedFork_implies_ambientTargetOpenings
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra
      key.nifsPiRlcContext.semantics key.nifsPiRlcContext.params
      key.nifsPiRlcContext.algebra)
    (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring
      key.nifsPiRlcContext.algebra.challengeValid)
    (outcome : AlignedForkOutcome key)
    (accepted : PiRLC.PaperWeakReduction.AcceptedFork
      key.nifsPiRlcContext outcome.adversary outcome.sample) :
    exists witness : OutputWitness shape columns,
      AmbientTargetOpenings key outcome.running outcome.fresh outcome.proof
        witness := by
  have extracted :=
    PiRLC.PaperWeakReduction.acceptedFork_extracts_correctedAmbient
      key.nifsPiRlcContext laws strongSet outcome.adversary outcome.sample
      accepted
  rcases extracted with ⟨_accepted, corrected⟩
  let witness : OutputWitness shape columns := {
    assignments := fun source =>
      PiRLC.PaperWeakReduction.extractedFamily key.nifsPiRlcContext laws
        strongSet outcome.adversary outcome.sample accepted
        (Fin.cast key.total_eq_sourceCount.symm source)
  }
  refine ⟨witness, ?_⟩
  intro source
  let index : Fin key.arity.total :=
    Fin.cast key.total_eq_sourceCount.symm source
  have atIndex := corrected index
  rw [outcome.batchAligned] at atIndex
  have atIndexPaper :=
    (key.ambientAgreement
      (key.piCcsOutputs outcome.running outcome.fresh outcome.proof index)
      (PiRLC.PaperWeakReduction.extractedFamily key.nifsPiRlcContext laws
        strongSet outcome.adversary outcome.sample accepted index)
      (by rfl)).mpr atIndex
  simpa [AmbientTargetOpenings, StrongReduction.AmbientOutputHolds,
    Key.nifsPiRlcBatch, Key.piCcsOutputs, witness, index] using atIndexPaper

/-- Absence of the NIFS ambient witness forces either failure of the oracle
programming receipt or the separately owned interactive fork-sampling event.
No loss is assigned to both classes. -/
theorem piRlcExtractionFailure_implies_forkSampling_or_programmingFailure
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra
      key.nifsPiRlcContext.semantics key.nifsPiRlcContext.params
      key.nifsPiRlcContext.algebra)
    (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring
      key.nifsPiRlcContext.algebra.challengeValid)
    (outcome : AlignedForkOutcome key)
    (failure : PiRlcCoordinateForkExtractionFailure key outcome.running
      outcome.fresh outcome.proof) :
    CoordinateForkSamplingFailure outcome ∨
      MultiForkProgrammingFailure outcome := by
  by_cases programmed : CoordinateProgrammingReceipt outcome
  · left
    refine ⟨programmed, ?_⟩
    intro accepted
    exact failure
      (acceptedFork_implies_ambientTargetOpenings laws strongSet outcome
        accepted)
  · exact Or.inr programmed

/-- The five deterministic interactive paper failures not owned by
Fiat--Shamir programming. This is closed and contains no transition-negation
constructor. -/
inductive ResidualBadEvent
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (outcome : AlignedForkOutcome key) : Prop where
  | piRlcForkSampling
      (failure : CoordinateForkSamplingFailure outcome)
  | piDecChildExtraction
      (failure : PiDecChildExtractionFailure key outcome.running outcome.fresh
        outcome.proof)
  | piCcsMixingRoot
      (sourceWitness : OutputWitness shape columns)
      (root : PiCcsMixingRoot key outcome.running outcome.fresh outcome.proof
        sourceWitness)
  | piCcsSumCheckCollision
      (sourceWitness : OutputWitness shape columns)
      (collision : PiCcsSumCheckCollision key outcome.running outcome.fresh
        outcome.proof sourceWitness)
  | parentBindingCollision
      (collision : Nonempty (PiDEC.ParentOpeningBindingCollision
        key.semantics key.params
        (key.parent outcome.running outcome.fresh outcome.proof).commitment))

/-- Structural non-interactive NIFS soundness after the exact coordinate-fork
bridge. No event probability is assumed: accepted execution yields the paper
transition, one of the five residual paper events, or the concrete sixth
Fiat--Shamir predicate for this aligned outcome. -/
theorem verify_sound_or_residual_or_multiFork
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra
      key.nifsPiRlcContext.semantics key.nifsPiRlcContext.params
      key.nifsPiRlcContext.algebra)
    (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring
      key.nifsPiRlcContext.algebra.challengeValid)
    (outcome : AlignedForkOutcome key)
    (accepted : verify key outcome.running outcome.fresh outcome.proof =
      some outcome.result) :
    Transition key outcome.running outcome.fresh outcome.result ∨
      ResidualBadEvent key outcome ∨
      (nifsEventPredicates key).at
        .multiForkProgrammingFailure outcome := by
  rcases verify_sound key outcome.running outcome.fresh outcome.proof
      outcome.result accepted with transition | badEvent
  · exact Or.inl transition
  · right
    cases badEvent with
    | piRlcCoordinateForkExtraction failure =>
        rcases
            piRlcExtractionFailure_implies_forkSampling_or_programmingFailure
              laws strongSet outcome failure with
          forkSampling | programming
        · exact Or.inl (.piRlcForkSampling forkSampling)
        · exact Or.inr programming
    | piDecChildExtraction failure =>
        exact Or.inl (.piDecChildExtraction failure)
    | piCcsMixingRoot sourceWitness root =>
        exact Or.inl (.piCcsMixingRoot sourceWitness root)
    | piCcsSumCheckCollision sourceWitness collision =>
        exact Or.inl (.piCcsSumCheckCollision sourceWitness collision)
    | parentBindingCollision collision =>
        exact Or.inl (.parentBindingCollision collision)

end Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
