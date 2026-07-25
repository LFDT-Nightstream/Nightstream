import Nightstream.Assurance.FPrimeFullHistoryCircuit
import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter

/-!
Contract: frozen fixed-one F' acceptance extracted from the exact supported
full-history R1CS artifact.

Assurance tier: artifact-checked.

Owns:
- the exact adapter parameterization of the plain/stateless `[1, 1]`
  full-history semantics;
- the two concrete frozen fixed-one inputs and outputs;
- R1CS-satisfaction-to-frozen-checker soundness for both generated steps,
  modulo the existing named recursive projection-root event.

Does not own:
- equality of the generated full-history rows with the selected canonical
  typed-lowering program;
- Rust-source or compiled-Rust semantics;
- honest assignment construction;
- terminal-verifier refinement;
- probability bounds for the named root event;
- Poseidon2, commitment, or extraction security.

The theorem consumes the exact current artifact rows.  It does not reclassify
that artifact as the canonical obligation-10 encoding.

Emits constraints: no.
-/

namespace Nightstream.Assurance.FPrimeFullHistoryCanonicalSteps

open Nightstream.Implementation.R1CS
open Nightstream.Protocol.FPrime

namespace Circuit

abbrev Digest :=
  Nightstream.Assurance.FPrimeFullHistoryCircuit.Digest

abbrev Fresh :=
  Nightstream.Assurance.FPrimeFullHistoryCircuit.Fresh

abbrev Accumulator :=
  Nightstream.Assurance.FPrimeFullHistoryCircuit.Accumulator

abbrev Proof :=
  Nightstream.Assurance.FPrimeFullHistoryCircuit.Proof

abbrev DirectState :=
  Nightstream.HyperNova.Construction2.State
    Digest Accumulator Fresh Unit

end Circuit

namespace Adapter

open
  Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter

/-- The exact fixed-one paper machine induced by the row-decoded full-history
services. -/
def parameters :
    Parameters Unit Unit Circuit.Digest Circuit.Digest Circuit.Accumulator
      Circuit.Fresh Circuit.Proof Unit Circuit.Digest Unit where
  hash :=
    Nightstream.Assurance.FPrimeFullHistoryCircuit.environment.hashSemantics
  step :=
    Nightstream.Assurance.FPrimeFullHistoryCircuit.environment.stepSemantics
  mode :=
    Nightstream.Assurance.FPrimeFullHistoryCircuit.environment.mode
  context :=
    Nightstream.Assurance.FPrimeFullHistoryCircuit.environment.context

def setup :=
  Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter.setup
    parameters

def machine :=
  Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter.machine
    parameters

def baseInput (assignment : Nat → Nat) :=
  input parameters
    Nightstream.Assurance.FPrimeFullHistoryCircuit.initialState
    (Nightstream.Assurance.FPrimeFullHistoryCircuit.baseInput assignment)
    Nightstream.Assurance.FPrimeFullHistoryCircuit.baseProof

def baseOutput (assignment : Nat → Nat) :=
  output parameters
    (Nightstream.Assurance.FPrimeFullHistoryCircuit.middleState assignment)
    Nightstream.Assurance.FPrimeFullHistoryCircuit.baseProof

def recursiveInput
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) :=
  input parameters
    (Nightstream.Assurance.FPrimeFullHistoryCircuit.middleState assignment)
    (Nightstream.Assurance.FPrimeFullHistoryCircuit.recursiveInput assignment)
    (Nightstream.Assurance.FPrimeFullHistoryCircuit.recursiveProof
      assignment canonical)

def recursiveOutput
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) :=
  output parameters
    (Nightstream.Assurance.FPrimeFullHistoryCircuit.finalState
      assignment canonical)
    (Nightstream.Assurance.FPrimeFullHistoryCircuit.recursiveProof
      assignment canonical)

end Adapter

/-- Both concrete fixed-one invocations accepted by the frozen executable
checker. -/
structure AcceptedSteps
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) : Prop where
  base :
    Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Accepts
      Adapter.setup
      Adapter.machine
      (Adapter.baseInput assignment)
      (Adapter.baseOutput assignment)
  recursive :
    Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Accepts
      Adapter.setup
      Adapter.machine
      (Adapter.recursiveInput assignment canonical)
      (Adapter.recursiveOutput assignment canonical)

/-- Exact generated full-history R1CS satisfaction implies acceptance of both
frozen fixed-one F' steps, or exposes the already named recursive
projection-root event. -/
theorem fullRows_imply_frozenSteps_or_bad
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (rows :
      Satisfies
        Nightstream.Implementation.R1CS.FPrimeFullHistoryRows.fullRows
        assignment) :
    AcceptedSteps assignment canonical ∨
      Nightstream.Assurance.FPrimeFullHistoryCircuit.BadEvent assignment := by
  letI : DecidableEq Circuit.Proof :=
    fun left right => Classical.propDecidable (left = right)
  rcases
      Nightstream.Assurance.FPrimeFullHistoryCircuit.exactSteps_of_fullRows_or_bad
        prime canonical one rows with steps | bad
  · left
    constructor
    · exact
        (Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter.canonicalAccepts_iff_holds
          Adapter.parameters
          Nightstream.Assurance.FPrimeFullHistoryCircuit.initialState
          (Nightstream.Assurance.FPrimeFullHistoryCircuit.middleState
            assignment)
          (Nightstream.Assurance.FPrimeFullHistoryCircuit.baseInput assignment)
          Nightstream.Assurance.FPrimeFullHistoryCircuit.baseProof).2 steps.1
    · exact
        (Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter.canonicalAccepts_iff_holds
          Adapter.parameters
          (Nightstream.Assurance.FPrimeFullHistoryCircuit.middleState
            assignment)
          (Nightstream.Assurance.FPrimeFullHistoryCircuit.finalState
            assignment canonical)
          (Nightstream.Assurance.FPrimeFullHistoryCircuit.recursiveInput
            assignment)
          (Nightstream.Assurance.FPrimeFullHistoryCircuit.recursiveProof
            assignment canonical)).2 steps.2
  · exact Or.inr bad

end Nightstream.Assurance.FPrimeFullHistoryCanonicalSteps
