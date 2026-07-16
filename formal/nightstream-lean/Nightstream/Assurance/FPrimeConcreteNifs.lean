import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.ConcreteNifs
import Nightstream.Protocol.FPrime.Step

/-!
High-level F-prime `Step.Semantics` bridge for the exact local row-decoded NIFS
checker. This module proves which callback result follows from accepted rows;
it does not by itself identify that result with the independent paper-level
SuperNeo fold relation.
-/
namespace Nightstream.Assurance.FPrimeConcreteNifs

open Nightstream.SuperNeo.ProjectionCheck
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection

/-! ## F' specialization

Only non-NIFS services remain parameters below. In particular, callers cannot
replace the row-decoded NIFS checker: the callback is definitionally
`recursiveNativeVerify`, which adds coefficient exactness to the sampled
semantic checks. Generated rows reach that callback or the explicit
projection `BadRoot` branch.
-/

/-- Plain/stateless F' semantics with the production row-decoded NIFS checker. -/
def stepSemantics
    (chunkDigest : Nat → List Fresh → Digest)
    (freshLink : Digest → Fresh → Bool)
    (applicationStep : Digest → List Fresh → Digest → Bool) :
    Nightstream.Protocol.FPrime.Step.Semantics
      Digest Accumulator Fresh Proof Unit Unit where
  emptyRunning := emptyAccumulator
  initialNebula := none
  runningDigest := Accumulator.handle
  chunkDigest := chunkDigest
  freshLink := freshLink
  nifsVerify := fun context running latest proof =>
    if decide (running = emptyAccumulator) then
      if recursiveContextCheck context proof &&
          recursiveLatestCheck latest proof then
        recursiveNativeVerify proof
      else none
    else none
  applicationStep := applicationStep
  nebulaVerify := fun prior opening next =>
    decide (prior = none ∧ opening = none ∧ next = none)

/-- Exact recursive row families force the concrete F' callback to return the
decoded next accumulator.  No refinement premise is accepted. -/
theorem stepSemantics_nifsVerify
    (chunkDigest : Nat → List Fresh → Digest)
    (freshLink : Digest → Fresh → Bool)
    (applicationStep : Digest → List Fresh → Digest → Bool)
    (context : Nightstream.Protocol.FPrime.Step.NifsContext Digest Unit)
    (latest : List Fresh)
    (proof : Proof)
    (binding : FPrimeFullHistoryTranscriptSound.ContextBinding
      proof.assignment context latest)
    (accepted : RecursiveSemanticAccepted proof.assignment)
    (exact : BatchExact
      (ProjectionProgram.BatchIdentity recursiveTraces proof.assignment)) :
    (stepSemantics chunkDigest freshLink applicationStep).nifsVerify
        context emptyAccumulator latest proof =
      some (recursiveAccumulator proof) := by
  have native := recursiveNativeVerify_of_exact accepted exact
  simp [stepSemantics, recursiveContextCheck, recursiveLatestCheck,
    binding.contextEq, binding.latestEq, native]

/-- Exact recursive row families drive the fixed production callback to the
decoded accumulator, or expose the sole deterministic projection-root event.
No caller-supplied verifier-result proposition appears in the interface. -/
theorem recursive_rows_nifsVerify_or_badRoot
    (prime : EuclidPrime goldilocksP)
    (chunkDigest : Nat → List Fresh → Digest)
    (freshLink : Digest → Fresh → Bool)
    (applicationStep : Digest → List Fresh → Digest → Bool)
    (context : Nightstream.Protocol.FPrime.Step.NifsContext Digest Unit)
    (latest : List Fresh)
    (proof : Proof)
    (one : proof.assignment 0 = 1)
    (rows : RecursiveRows proof.assignment)
    (binding : FPrimeFullHistoryTranscriptSound.ContextBinding
      proof.assignment context latest) :
    (stepSemantics chunkDigest freshLink applicationStep).nifsVerify
        context emptyAccumulator latest proof =
        some (recursiveAccumulator proof) ∨
      BatchBadRoot ProjectionProgram.K.ops
        (ProjectionProgram.BatchIdentity recursiveTraces proof.assignment) := by
  have semantic := recursive_rows_sound prime proof.canonical one rows
  rcases recursive_semantic_sound_or_badRoot semantic with artifact | bad
  · left
    exact stepSemantics_nifsVerify chunkDigest freshLink applicationStep
      context latest proof binding semantic
      artifact.projection.exact
  · exact Or.inr bad

/-- Successful executable recursive verification has the complete exact
artifact certificate, or names the sole deterministic projection failure. -/
theorem recursive_verify_sound_or_badRoot
    {proof : Proof}
    (accepted : recursiveCheck proof = true) :
    RecursiveArtifactAccepted proof.assignment ∨
      BatchBadRoot ProjectionProgram.K.ops
        (ProjectionProgram.BatchIdentity recursiveTraces proof.assignment) := by
  exact recursive_semantic_sound_or_badRoot
    ((recursiveCheck_eq_true_iff proof).1 accepted)

/-- Successful executable terminal verification has the exact arity-15
certificate, or the named projection-root event. -/
theorem terminal_verify_sound_or_badRoot
    {proof : Proof}
    (accepted : terminalCheck proof = true) :
    TerminalArtifactAccepted proof.assignment ∨
      BatchBadRoot ProjectionProgram.K.ops
        (ProjectionProgram.BatchIdentity terminalTraces proof.assignment) := by
  exact terminal_semantic_sound_or_badRoot
    ((terminalCheck_eq_true_iff proof).1 accepted)


end Nightstream.Assurance.FPrimeConcreteNifs
