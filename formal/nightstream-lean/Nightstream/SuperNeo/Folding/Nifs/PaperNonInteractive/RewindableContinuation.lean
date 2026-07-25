import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.CoordinateForkBridge
import Nightstream.SuperNeo.Folding.PiDEC.PaperReduction

/-!
Rewindable continuation boundary for the paper non-interactive NIFS.

Source: SuperNeo Sections 7.4--7.5, Appendices D.5--D.6, and the sequential
`Pi_DEC ∘ Pi_RLC` extraction order from Theorem 7.

Owns: the split between the challenge-independent `Pi_CCS` prefix and the
challenge-dependent `Pi_DEC` reply; one continuation function that returns
the public child messages and extracted child assignments for each complete
`Pi_RLC` challenge vector; the exact base proof; a projection to
`AlignedForkOutcome` whose PiRLC assignment oracle is definitionally the
PiDEC recomposition of that same continuation; and conversion of successful
PiDEC continuations into an accepted PiRLC coordinate fork.

Does not own: a distribution on continuations or forks, random-oracle
programming, a forking probability theorem, success of any continuation,
event bounds, Poseidon2, Ajtai, Rust, R1CS, artifacts, minimality, or costs.

Emits constraints: no.

No proposition field asserts continuation alignment.  The alignment is the
definition of `toAlignedForkOutcome.adversary.oracle`.  Likewise, PiRLC
response success is not assumed: it follows pointwise from the exact PiDEC
reduction-of-knowledge theorem when `ContinuationSuccessAt` holds.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking

universe uExtension uCommitment uPublicInput uScalar uState

/-- The challenge-independent part of the sole NIFS prover message. -/
structure PrefixMessage
    (Extension : Type uExtension)
    (shape : Shape)
    (degreeBound : Nat) where
  piCcsRounds : Fin shape.cubeVariables ->
    Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial Extension degreeBound
  piCcsOutput : FullOutputCoordinates.FullOutput Extension shape

/-- One challenge-dependent continuation reply.  Public PiDEC child messages
and the straight-line extractor's child assignments have one owner. -/
structure ContinuationReply
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound) where
  piDecCommitments : Fin shape.runningCount -> Commitment
  piDecEvaluations :
    Fin shape.runningCount -> EvaluationFamily Extension shape
  childAssignments : Fin key.params.k -> Assignment F columns

/-- One malicious prover prefix and its single vector-at-once continuation.
The continuation receives the complete PiRLC challenge vector, as in
Appendix D.5. -/
structure RewindableProver
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound) where
  piCcsPrefix : PrefixMessage Extension shape degreeBound
  reply :
    PiRLC.PaperWeakReduction.Challenge key.nifsPiRlcContext ->
      ContinuationReply key

namespace RewindableProver

/-- Assemble the actual one-message proof returned under one complete PiRLC
challenge vector. -/
def proofAt
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (prover : RewindableProver key)
    (challenges :
      PiRLC.PaperWeakReduction.Challenge key.nifsPiRlcContext) :
    Proof Extension Commitment shape degreeBound where
  piCcsRounds := prover.piCcsPrefix.piCcsRounds
  piCcsOutput := prover.piCcsPrefix.piCcsOutput
  piDecCommitments := (prover.reply challenges).piDecCommitments
  piDecEvaluations := (prover.reply challenges).piDecEvaluations

/-- A typing-only challenge vector used to materialize the prefix replay.
The continuation reply selected here cannot affect any PiCCS coin or output. -/
def probeChallenges
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound) :
    PiRLC.PaperWeakReduction.Challenge key.nifsPiRlcContext :=
  fun index => key.piRlcResponse key.initialTranscriptState index

/-- The base PiRLC challenge vector derived from the fixed prefix. -/
def baseChallenges
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (prover : RewindableProver key)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape) :
    PiRLC.PaperWeakReduction.Challenge key.nifsPiRlcContext :=
  key.piRlcChallenges running fresh
    (prover.proofAt (probeChallenges key))

/-- The actual base proof uses the continuation reply to the challenge vector
derived from its own fixed prefix. -/
def baseProof
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (prover : RewindableProver key)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape) :
    Proof Extension Commitment shape degreeBound :=
  prover.proofAt (prover.baseChallenges running fresh)

/-- Replacing the typing-only reply by the actual base reply leaves the
prefix-derived challenge vector unchanged. -/
@[simp] theorem piRlcChallenges_baseProof
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (prover : RewindableProver key)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape) :
    key.piRlcChallenges running fresh (prover.baseProof running fresh) =
      prover.baseChallenges running fresh := by
  rfl

/-- The PiRLC response witness is exactly the PiDEC straight-line
recomposition of this continuation's extracted children. -/
def assignmentAt
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (prover : RewindableProver key)
    (challenges :
      PiRLC.PaperWeakReduction.Challenge key.nifsPiRlcContext) :
    Assignment F columns :=
  key.piDecAlgebra.recomposeAssignment
    (prover.reply challenges).childAssignments

end RewindableProver

namespace Key

/-- Exact PiDEC reduction context selected by the NIFS key. -/
def nifsPiDecContext
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound) :
    PiDEC.PaperReduction.Context
      (MatrixCoefficientSource.MatrixSource F shape columns blockCount)
      (Assignment F columns)
      PublicInput
      (CubePoint Extension shape.cubeVariables)
      (EvaluationFamily Extension shape)
      Commitment where
  semantics := key.semantics
  params := key.params
  algebra := key.piDecAlgebra
  publicSplit := key.piDecPublicInputSplit
  evaluationArity := key.piDecEvaluationArity
  kPositive := key.kPositive

end Key

/-- One fork outcome generated from a single owned continuation. -/
structure RewindableForkOutcome
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
  prover : RewindableProver key
  sample : ForkSample Scalar key.arity.total

namespace RewindableForkOutcome

/-- Forget continuation ownership while retaining an `AlignedForkOutcome`.
The result and PiRLC batch are verifier-computed, and the assignment oracle is
definitionally the continuation's PiDEC recomposition. -/
def toAlignedForkOutcome
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (outcome : RewindableForkOutcome key) :
    AlignedForkOutcome key where
  running := outcome.running
  fresh := outcome.fresh
  proof := outcome.prover.baseProof outcome.running outcome.fresh
  result := key.output outcome.running outcome.fresh
    (outcome.prover.baseProof outcome.running outcome.fresh)
  adversary := {
    batch := key.nifsPiRlcBatch outcome.running outcome.fresh
      (outcome.prover.baseProof outcome.running outcome.fresh)
    oracle := outcome.prover.assignmentAt
  }
  sample := outcome.sample
  batchAligned := rfl

/-- The forgotten PiRLC oracle is literally the owned continuation's PiDEC
recomposition, not a separately aligned function. -/
@[simp] theorem toAlignedForkOutcome_oracle
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (outcome : RewindableForkOutcome key)
    (challenges :
      PiRLC.PaperWeakReduction.Challenge key.nifsPiRlcContext) :
    outcome.toAlignedForkOutcome.adversary.oracle challenges =
      outcome.prover.assignmentAt challenges := by
  rfl

/-- The forgotten batch is literally the PiCCS output batch from the same
base proof. -/
@[simp] theorem toAlignedForkOutcome_batch
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (outcome : RewindableForkOutcome key) :
    outcome.toAlignedForkOutcome.adversary.batch =
      key.nifsPiRlcBatch outcome.running outcome.fresh
        (outcome.prover.baseProof outcome.running outcome.fresh) := by
  rfl

/-- Verifier-computed PiRLC parent for one continuation challenge. -/
def parentAt
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (outcome : RewindableForkOutcome key)
    (challenges :
      PiRLC.PaperWeakReduction.Challenge key.nifsPiRlcContext) :=
  PiRLC.combinedOutput key.piRlcAlgebra
    outcome.toAlignedForkOutcome.adversary.batch.system
    outcome.toAlignedForkOutcome.adversary.batch.point
    outcome.toAlignedForkOutcome.adversary.batch.inputs
    challenges

/-- Exact PiDEC execution induced by one continuation reply. -/
def piDecExecutionAt
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (outcome : RewindableForkOutcome key)
    (challenges :
      PiRLC.PaperWeakReduction.Challenge key.nifsPiRlcContext) :
    PiDEC.PaperReduction.Execution key.nifsPiDecContext where
  attempt := {
    parent := outcome.parentAt challenges
    messages := fun child => {
      commitment :=
        (outcome.prover.reply challenges).piDecCommitments
          (Fin.cast key.outputCount_eq child)
      evaluations := #[
        (outcome.prover.reply challenges).piDecEvaluations
          (Fin.cast key.outputCount_eq child)]
    }
  }
  childAssignments :=
    (outcome.prover.reply challenges).childAssignments

/-- Exact success event for the challenge-dependent PiDEC continuation. -/
def ContinuationSuccessAt
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (outcome : RewindableForkOutcome key)
    (challenges :
      PiRLC.PaperWeakReduction.Challenge key.nifsPiRlcContext) : Prop :=
  PiDEC.PaperReduction.Success key.nifsPiDecContext
    (outcome.piDecExecutionAt challenges)

/-- PiDEC's straight-line theorem opens the exact verifier-computed PiRLC
parent with the continuation-owned recomposition. -/
theorem continuationSuccessAt_implies_parentOpening
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (outcome : RewindableForkOutcome key)
    (challenges :
      PiRLC.PaperWeakReduction.Challenge key.nifsPiRlcContext)
    (success : outcome.ContinuationSuccessAt challenges) :
    CE.Holds key.semantics key.params
      (outcome.parentAt challenges)
      (outcome.prover.assignmentAt challenges) := by
  exact PiDEC.PaperReduction.success_implies_extractedSource
    key.nifsPiDecContext (outcome.piDecExecutionAt challenges) success

/-- The same PiDEC opening is exactly operational PiRLC response success over
the aligned NIFS batch. -/
theorem continuationSuccessAt_implies_piRlcVerifies
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (outcome : RewindableForkOutcome key)
    (challenges :
      PiRLC.PaperWeakReduction.Challenge key.nifsPiRlcContext)
    (success : outcome.ContinuationSuccessAt challenges) :
    PiRLC.PaperWeakReduction.verifies key.nifsPiRlcContext
      outcome.toAlignedForkOutcome.adversary challenges
      (outcome.toAlignedForkOutcome.adversary.oracle challenges) := by
  simpa [PiRLC.PaperWeakReduction.verifies,
    PiRLC.PaperWeakReduction.response,
    PiRLC.PaperForkExtraction.Response.Success,
    PiRLC.PaperForkExtraction.Response.output,
    parentAt] using
    outcome.continuationSuccessAt_implies_parentOpening challenges success

/-- Successful PiDEC continuations at the base and every programmed fork
produce the exact accepted coordinate fork consumed by Appendix D.5. -/
theorem continuationSuccesses_imply_acceptedFork
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (outcome : RewindableForkOutcome key)
    (receipt : CoordinateProgrammingReceipt outcome.toAlignedForkOutcome)
    (baseSuccess :
      outcome.ContinuationSuccessAt outcome.sample.base)
    (forkSuccess : forall coordinate,
      outcome.ContinuationSuccessAt (outcome.sample.forks coordinate)) :
    PiRLC.PaperWeakReduction.AcceptedFork key.nifsPiRlcContext
      outcome.toAlignedForkOutcome.adversary outcome.sample where
  baseAccepted :=
    outcome.continuationSuccessAt_implies_piRlcVerifies
      outcome.sample.base baseSuccess
  forkAccepted := fun coordinate =>
    outcome.continuationSuccessAt_implies_piRlcVerifies
      (outcome.sample.forks coordinate) (forkSuccess coordinate)
  baseValid := receipt.baseValid
  forkValid := receipt.forkValid
  agreeExcept := receipt.agreeExcept
  changed := receipt.changed

end RewindableForkOutcome

end Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
