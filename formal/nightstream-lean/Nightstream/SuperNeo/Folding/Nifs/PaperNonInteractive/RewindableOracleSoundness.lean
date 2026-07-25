import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.OracleSoundness
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.RewindableContinuation

/-!
Conditional non-interactive soundness over owned rewindable continuations.

Source: SuperNeo Sections 7.3--7.5 and Appendices D.3--D.6, with
Fiat--Shamir under the explicit random-oracle contract.

Owns: deterministic pushforward of a `RewindableForkOutcome` experiment to
the exact aligned outcome consumed by the eleven-event theorem; preservation
of the union bound; rewindable acceptance, transition, and failure predicates;
the resulting subtractive soundness inequality; and refinement of the
PiRLC fork-sampling event to failure of one exact PiDEC continuation.

Does not own: an outcome distribution, a malicious-prover machine, oracle
programming, any event bound, challenge uniformity, Poseidon2, Ajtai, Rust,
R1CS, artifacts, minimality, or costs.

Emits constraints: no.

The experiment remains an explicit input because this module proves no
random-oracle theorem.  Unlike the generic aligned theorem, however, every
outcome in this statement owns one continuation and its PiRLC oracle is
definitionally that continuation's PiDEC recomposition.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.InteractiveReduction.Paper
open Nightstream.SuperNeo.InteractiveReduction.ProbabilityCalculus

universe uExtension uCommitment uPublicInput uScalar uState uWeight

/-- Push one experiment through the definitional owned-continuation
projection. -/
def rewindableAlignedExperiment
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {Weight : Type uWeight}
    {shape : PiCCS.PaperJoint.Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {scale : ProbabilityScale Weight}
    (experiment :
      ProbabilityExperiment scale (RewindableForkOutcome key)) :
    ProbabilityExperiment scale (AlignedForkOutcome key) where
  probability := fun event =>
    experiment.probability fun outcome =>
      event outcome.toAlignedForkOutcome
  monotone := by
    intro left right implication
    exact experiment.monotone fun outcome accepted =>
      implication outcome.toAlignedForkOutcome accepted

/-- Pushforward probabilities are definitionally preimage probabilities. -/
@[simp] theorem rewindableAlignedExperiment_probability
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {Weight : Type uWeight}
    {shape : PiCCS.PaperJoint.Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {scale : ProbabilityScale Weight}
    (experiment :
      ProbabilityExperiment scale (RewindableForkOutcome key))
    (event : AlignedForkOutcome key -> Prop) :
    (rewindableAlignedExperiment experiment).probability event =
      experiment.probability fun outcome =>
        event outcome.toAlignedForkOutcome := by
  rfl

/-- A union bound is preserved by the deterministic continuation-forgetting
map. -/
def rewindableAlignedUnionBound
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {Weight : Type uWeight}
    {shape : PiCCS.PaperJoint.Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {scale : ProbabilityScale Weight}
    (experiment :
      ProbabilityExperiment scale (RewindableForkOutcome key))
    (unionLaw : UnionBound experiment) :
    UnionBound (rewindableAlignedExperiment experiment) where
  unionBound := by
    intro left right
    exact unionLaw.unionBound
      (fun outcome => left outcome.toAlignedForkOutcome)
      (fun outcome => right outcome.toAlignedForkOutcome)

/-- Executable acceptance of the base proof from one owned continuation. -/
def RewindableAcceptedOutcome
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : PiCCS.PaperJoint.Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (outcome : RewindableForkOutcome key) : Prop :=
  AcceptedOutcome outcome.toAlignedForkOutcome

/-- Independent paper transition of the base proof from one owned
continuation. -/
def RewindableTransitionOutcome
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : PiCCS.PaperJoint.Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (outcome : RewindableForkOutcome key) : Prop :=
  TransitionOutcome outcome.toAlignedForkOutcome

/-- Exact eleven-event union pulled back to one owned continuation. -/
def RewindableNonInteractiveFailure
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : PiCCS.PaperJoint.Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (outcome : RewindableForkOutcome key) : Prop :=
  NonInteractiveFailure outcome.toAlignedForkOutcome

/-- Exact success of the PiDEC continuation at the base and every coordinate
fork. -/
def AllContinuationsSuccessful
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : PiCCS.PaperJoint.Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (outcome : RewindableForkOutcome key) : Prop :=
  outcome.ContinuationSuccessAt outcome.sample.base /\
    forall coordinate,
      outcome.ContinuationSuccessAt (outcome.sample.forks coordinate)

/-- A complete programming receipt exists, but at least one exact PiDEC
continuation does not satisfy the operational success relation.  This refines
the existing PiRLC fork-sampling event; it is not a twelfth budget item. -/
def PiDecContinuationFailure
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : PiCCS.PaperJoint.Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (outcome : RewindableForkOutcome key) : Prop :=
  CoordinateProgrammingReceipt outcome.toAlignedForkOutcome /\
    ¬ AllContinuationsSuccessful outcome

/-- On an owned continuation, rejection of the PiRLC coordinate fork implies
failure of at least one exact PiDEC continuation. -/
theorem piRlcForkSamplingFailure_implies_piDecContinuationFailure
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : PiCCS.PaperJoint.Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (outcome : RewindableForkOutcome key)
    (failure :
      CoordinateForkSamplingFailure outcome.toAlignedForkOutcome) :
    PiDecContinuationFailure outcome := by
  refine ⟨failure.1, ?_⟩
  intro allSuccessful
  exact failure.2
    (outcome.continuationSuccesses_imply_acceptedFork failure.1
      allSuccessful.1 allSuccessful.2)

/-- Headline conditional non-interactive NIFS soundness over outcomes whose
PiRLC oracle is definitionally owned by one rewindable PiDEC continuation. -/
theorem rewindable_accepted_probability_sub_total_le_transition
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {Weight : Type uWeight}
    [DecidableEq Extension]
    {shape : PiCCS.PaperJoint.Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {scale : ProbabilityScale Weight}
    (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra
      key.nifsPiRlcContext.semantics key.nifsPiRlcContext.params
      key.nifsPiRlcContext.algebra)
    (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring
      key.nifsPiRlcContext.algebra.challengeValid)
    (scaleLaws : ScaleLaws scale)
    (experiment :
      ProbabilityExperiment scale (RewindableForkOutcome key))
    (unionLaw : UnionBound experiment)
    (extractionBudget : NifsExtractionErrorBudget Weight)
    (interactiveBudget : InteractiveErrorBudget Weight)
    (fiatShamirBudget : FiatShamirErrorBudget Weight)
    (interactiveContract :
      InteractiveResidualContract
        (rewindableAlignedExperiment experiment) extractionBudget
        interactiveBudget)
    (randomOracleContract :
      NifsExplicitRandomOracleContract
        (rewindableAlignedExperiment experiment) fiatShamirBudget) :
    scale.le
      (scale.subtract
        (experiment.probability RewindableAcceptedOutcome)
        (nonInteractiveTotal scale extractionBudget interactiveBudget
          fiatShamirBudget))
      (experiment.probability RewindableTransitionOutcome) := by
  exact accepted_probability_sub_total_le_transition laws strongSet scaleLaws
    (rewindableAlignedExperiment experiment)
    (rewindableAlignedUnionBound experiment unionLaw)
    extractionBudget interactiveBudget fiatShamirBudget
    interactiveContract randomOracleContract

end Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
