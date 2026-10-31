import NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive.Verifier

/-! The refresh fold: SuperNeo v1.2's fold with Π_DEC moved to the front. The verifier checks
the 16 children that the proof gives for the input accumulator (Section 7.5), runs Π_CCS on
the fresh claim and those children (Section 7.3), and returns the Π_RLC parent (Section 7.4)
as the next accumulator. It reuses `Nifs.PaperNonInteractive`; the output split of that
module is not used. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Refresh

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive (Key Fresh Running piCcsCheck)

universe uCommitment uPublicInput uScalar uState

/-- The 16 children the prover gives for the input accumulator (SuperNeo v1.2 Section 7.5). -/
structure Split (Commitment : Type uCommitment) (shape : Shape) where
  commitments : Fin shape.runningCount -> Commitment
  evaluations : Fin shape.runningCount -> EvaluationFamily K shape

structure Proof (Commitment : Type uCommitment) (shape : Shape) (degreeBound : Nat) where
  split : Split Commitment shape
  nifs : Nifs.PaperNonInteractive.Proof K Commitment shape degreeBound

variable {Commitment : Type uCommitment} {PublicInput : Type uPublicInput}
  {Scalar : Type uScalar} {State : Type uState} {shape : Shape}
  {columns blockCount degreeBound : Nat}

abbrev Accumulator (Commitment : Type uCommitment) (PublicInput : Type uPublicInput)
    (shape : Shape) (columns blockCount : Nat) :=
  CE.Instance (RelationSource shape columns blockCount) PublicInput
    (CubePoint K shape.cubeVariables) (EvaluationFamily K shape) Commitment

def inputAttempt
    (key : Key K Commitment PublicInput Scalar State shape columns blockCount degreeBound)
    (accumulator : Accumulator Commitment PublicInput shape columns blockCount)
    (split : Split Commitment shape) :
    PiDEC.PaperVerifier.Attempt (RelationSource shape columns blockCount) PublicInput
      (CubePoint K shape.cubeVariables) (EvaluationFamily K shape) Commitment key.params where
  parent := accumulator
  messages := fun child => {
    commitment := split.commitments (Fin.cast key.outputCount_eq child)
    evaluations := #[split.evaluations (Fin.cast key.outputCount_eq child)] }

def children
    (key : Key K Commitment PublicInput Scalar State shape columns blockCount degreeBound)
    (accumulator : Accumulator Commitment PublicInput shape columns blockCount)
    (split : Split Commitment shape) : Running K Commitment PublicInput shape where
  point := accumulator.point
  commitments := split.commitments
  publicInputs := fun index =>
    key.piDecPublicInputSplit.split accumulator.publicInput
      (Fin.cast key.runningCount_eq_outputCount index)
  evaluations := split.evaluations

def inputCheck
    (key : Key K Commitment PublicInput Scalar State shape columns blockCount degreeBound)
    (accumulator : Accumulator Commitment PublicInput shape columns blockCount)
    (split : Split Commitment shape) : Bool :=
  letI := key.piDecDecision (inputAttempt key accumulator split)
  decide (PiDEC.PaperVerifier.Accepted key.piDecAlgebra key.piDecPublicInputSplit
    key.piDecEvaluationArity (inputAttempt key accumulator split))

/-- The refresh fold's verifier: Π_DEC on the input, then Π_CCS and Π_RLC. -/
def verify
    (key : Key K Commitment PublicInput Scalar State shape columns blockCount degreeBound)
    (accumulator : Accumulator Commitment PublicInput shape columns blockCount)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Commitment shape degreeBound) :
    Option (Accumulator Commitment PublicInput shape columns blockCount) :=
  if inputCheck key accumulator proof.split &&
      piCcsCheck key (children key accumulator proof.split) fresh proof.nifs then
    key.parent (children key accumulator proof.split) fresh proof.nifs
  else
    none

theorem inputCheck_eq_true_iff
    (key : Key K Commitment PublicInput Scalar State shape columns blockCount degreeBound)
    (accumulator : Accumulator Commitment PublicInput shape columns blockCount)
    (split : Split Commitment shape) :
    inputCheck key accumulator split = true ↔
      PiDEC.PaperVerifier.Accepted key.piDecAlgebra key.piDecPublicInputSplit
        key.piDecEvaluationArity (inputAttempt key accumulator split) := by
  letI := key.piDecDecision (inputAttempt key accumulator split)
  unfold inputCheck
  exact decide_eq_true_iff

theorem verify_eq_some_iff
    (key : Key K Commitment PublicInput Scalar State shape columns blockCount degreeBound)
    (accumulator : Accumulator Commitment PublicInput shape columns blockCount)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Commitment shape degreeBound)
    (parent : Accumulator Commitment PublicInput shape columns blockCount) :
    verify key accumulator fresh proof = some parent ↔
      PiDEC.PaperVerifier.Accepted key.piDecAlgebra key.piDecPublicInputSplit
          key.piDecEvaluationArity (inputAttempt key accumulator proof.split) ∧
        piCcsCheck key (children key accumulator proof.split) fresh proof.nifs = true ∧
        key.parent (children key accumulator proof.split) fresh proof.nifs = some parent := by
  rw [← inputCheck_eq_true_iff]
  unfold verify
  by_cases input : inputCheck key accumulator proof.split = true <;>
    by_cases piCcs :
      piCcsCheck key (children key accumulator proof.split) fresh proof.nifs = true <;>
    simp [input, piCcs]

end NightstreamFPrime.Spec.Folding.Refresh
