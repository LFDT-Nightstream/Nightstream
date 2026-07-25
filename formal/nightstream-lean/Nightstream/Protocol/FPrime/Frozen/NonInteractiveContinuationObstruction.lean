import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.OracleSoundness

/-!
Kernel obstruction to treating `AlignedForkOutcome` as an actual rewindable
NIFS continuation.

Owns: an oracle-replacement operation that preserves the exact NIFS
execution, the `Pi_CCS` output batch, and both frozen acceptance predicates;
and a theorem exhibiting two distinct replacement oracles with those same
observations.

Does not own: a rewindable random-oracle experiment, a continuation
semantics, a multi-forking theorem, any event bound, Poseidon2, Ajtai, Rust,
R1CS, artifacts, minimality, or costs.

Emits constraints: no.

`AlignedForkOutcome.batchAligned` binds the public `Pi_RLC` batch to the
preceding `Pi_CCS` output, but it does not bind `adversary.oracle` to the
malicious prover continuation that produced the NIFS proof.  The theorem
below makes that missing ownership visible without introducing an escape
event or assuming the desired linkage.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.FPrime.Frozen.NonInteractiveContinuationObstruction

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive

universe uExtension uCommitment uPublicInput uScalar uState

/-- Replace only the `Pi_RLC` response oracle.  The NIFS execution fields and
the batch fixed by the preceding `Pi_CCS` replay are retained literally. -/
def replaceForkOracle
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
    (oracle :
      PiRLC.PaperWeakReduction.Challenge key.nifsPiRlcContext ->
        Assignment F columns) :
    AlignedForkOutcome key where
  running := outcome.running
  fresh := outcome.fresh
  proof := outcome.proof
  result := outcome.result
  adversary := {
    batch := outcome.adversary.batch
    oracle := oracle
  }
  sample := outcome.sample
  batchAligned := outcome.batchAligned

/-- Replacing the fork oracle cannot change executable NIFS acceptance. -/
theorem replaceForkOracle_acceptedOutcome_iff
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
    (outcome : AlignedForkOutcome key)
    (oracle :
      PiRLC.PaperWeakReduction.Challenge key.nifsPiRlcContext ->
        Assignment F columns) :
    AcceptedOutcome (replaceForkOracle outcome oracle) ↔
      AcceptedOutcome outcome := by
  rfl

/-- Replacing the fork oracle cannot change the independently stated paper
transition for the retained NIFS execution. -/
theorem replaceForkOracle_transitionOutcome_iff
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
    (outcome : AlignedForkOutcome key)
    (oracle :
      PiRLC.PaperWeakReduction.Challenge key.nifsPiRlcContext ->
        Assignment F columns) :
    TransitionOutcome (replaceForkOracle outcome oracle) ↔
      TransitionOutcome outcome := by
  rfl

/-- If two assignments differ, constant response oracles returning them are
distinct even though replacing one by the other preserves NIFS acceptance
and the paper transition.  Thus the current outcome fields cannot prove that
the fork oracle is the continuation of the prover that produced the proof. -/
theorem distinct_replacement_oracles_same_nifs_execution
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
    (outcome : AlignedForkOutcome key)
    (left right : Assignment F columns)
    (different : left ≠ right) :
    let leftOutcome := replaceForkOracle outcome (fun _ => left)
    let rightOutcome := replaceForkOracle outcome (fun _ => right)
    leftOutcome.adversary.oracle ≠ rightOutcome.adversary.oracle ∧
      (AcceptedOutcome leftOutcome ↔ AcceptedOutcome rightOutcome) ∧
      (TransitionOutcome leftOutcome ↔ TransitionOutcome rightOutcome) := by
  dsimp only
  constructor
  · intro sameOracle
    apply different
    exact congrFun sameOracle
      (key.piRlcChallenges outcome.running outcome.fresh outcome.proof)
  · constructor <;> rfl

end Nightstream.Protocol.FPrime.Frozen.NonInteractiveContinuationObstruction
