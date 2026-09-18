import Nightstream.Protocol.FPrime.Step

/-!
Contract: the circuit-facing split acceptance boundary for F'.

A standalone producer circuit owns `Step.checkLocal`. The next recursive
consumer, or the terminal-link circuit for the trailing batch, owns
`checkOutgoing`. This module proves that those two executable acceptances are
exactly the full semantic step checker consumed by TRACE-VALID.

Artifact theorems should target these Boolean checks. They must not assume a
`LocalHolds` or `Holds` conclusion directly.
-/

namespace Nightstream.Assurance.FPrimeCircuit

open Nightstream.HyperNova.Construction2
open Nightstream.Protocol.FPrime

universe uDigest uParams uStructure uHeader uRunning uFresh uNifsProof
  uNebulaDigest uNebulaOpen

def checkOutgoing
    {Digest : Type uDigest}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    (stepSemantics : Step.Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (input : Step.Input Fresh Nebula NebulaOpen)
    (proof : Step.Proof Digest NifsProof NebulaOpen) : Bool :=
  input.nextLatest.all (stepSemantics.freshLink proof.xOut)

theorem checkOutgoing_eq_true_iff
    {Digest : Type uDigest}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    (stepSemantics : Step.Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (input : Step.Input Fresh Nebula NebulaOpen)
    (proof : Step.Proof Digest NifsProof NebulaOpen) :
    checkOutgoing stepSemantics input proof = true ↔
      Step.OutgoingLinked stepSemantics input proof := by
  rfl

/-- Executable circuit composition contract. A producer-local acceptance and
its unique consumer/terminal acceptance are neither weaker nor stronger than
the closed M3 step checker. -/
theorem split_check_eq_true_iff
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Digest : Type uDigest}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {NebulaOpen : Type uNebulaOpen}
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (hashSemantics : XOut.Semantics
      Params StructureDigest Header Digest Nebula NebulaDigest)
    (stepSemantics : Step.Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : XOut.Context Params StructureDigest Header Digest)
    (prior next : State Digest Running Fresh Nebula)
    (input : Step.Input Fresh Nebula NebulaOpen)
    (proof : Step.Proof Digest NifsProof NebulaOpen) :
    Step.check hashSemantics stepSemantics mode context
        prior next input proof = true ↔
      Step.checkLocal hashSemantics stepSemantics mode context
          prior next input proof = true ∧
        checkOutgoing stepSemantics input proof = true := by
  rw [Step.check_eq_true_iff_holds,
    Step.checkLocal_eq_true_iff_localHolds,
    checkOutgoing_eq_true_iff,
    Step.holds_iff_local_and_outgoing]

/-- One-way form used after exact producer and consumer artifacts have each
established their own executable check. -/
theorem close_split_acceptance
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Digest : Type uDigest}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {NebulaOpen : Type uNebulaOpen}
    [DecidableEq Digest]
    [DecidableEq Running]
    [DecidableEq Fresh]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (hashSemantics : XOut.Semantics
      Params StructureDigest Header Digest Nebula NebulaDigest)
    (stepSemantics : Step.Semantics Digest Running Fresh NifsProof Nebula NebulaOpen)
    (mode : XOut.Mode)
    (context : XOut.Context Params StructureDigest Header Digest)
    (prior next : State Digest Running Fresh Nebula)
    (input : Step.Input Fresh Nebula NebulaOpen)
    (proof : Step.Proof Digest NifsProof NebulaOpen)
    (localAccepted : Step.checkLocal hashSemantics stepSemantics mode context
      prior next input proof = true)
    (outgoingAccepted : checkOutgoing stepSemantics input proof = true) :
    Step.check hashSemantics stepSemantics mode context
      prior next input proof = true :=
  (split_check_eq_true_iff hashSemantics stepSemantics mode context
    prior next input proof).2 ⟨localAccepted, outgoingAccepted⟩

end Nightstream.Assurance.FPrimeCircuit
