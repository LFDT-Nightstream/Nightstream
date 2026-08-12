import Nightstream.Protocol.FPrime.Step

/-!
Contract: exact one-step-delayed closure of a local F-prime trace.

Assurance tier: protocol model.

Owns an ordered trace of standalone local invocations, exact state identity
between adjacent invocations, terminal closure of the trailing fresh batch,
and an induction that derives the fresh-public link for every produced batch.

Does not own circuit-row refinement, NIFS knowledge soundness, a concrete
fresh-link function, terminal backend verification, or cryptographic bounds.

The central result does not assume any nonterminal outgoing link. A producer
installs its exact batch in the outgoing state. The next local invocation
must consume that same active batch and check its link. Only the final batch
is supplied by the separate terminal-link relation.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.FPrime.DelayedTrace

open Nightstream.HyperNova.Construction2
open Nightstream.Protocol.FPrime

universe uDigest uParams uStructure uHeader uRunning uFresh uNifsProof
  uNebulaDigest uNebulaOpen

/-- Fixed semantic configuration shared by every invocation in one trace. -/
structure Configuration
    (Params : Type uParams)
    (StructureDigest : Type uStructure)
    (Header : Type uHeader)
    (Digest : Type uDigest)
    (Running : Type uRunning)
    (Fresh : Type uFresh)
    (NifsProof : Type uNifsProof)
    (Nebula : Type)
    (NebulaDigest : Type uNebulaDigest)
    (NebulaOpen : Type uNebulaOpen) where
  hashSemantics : XOut.Semantics Params StructureDigest Header Digest Nebula
    NebulaDigest
  stepSemantics : Step.Semantics Digest Running Fresh NifsProof Nebula
    NebulaOpen
  mode : XOut.Mode
  context : XOut.Context Params StructureDigest Header Digest

/-- One standalone base or recursive invocation. `local` excludes the link
for `input.nextLatest`; that link is deliberately delayed. -/
structure Invocation
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
    (configuration : Configuration Params StructureDigest Header Digest
      Running Fresh NifsProof Nebula NebulaDigest NebulaOpen) where
  prior : State Digest Running Fresh Nebula
  next : State Digest Running Fresh Nebula
  input : Step.Input Fresh Nebula NebulaOpen
  proof : Step.Proof Digest NifsProof NebulaOpen
  localHolds : Step.LocalHolds configuration.hashSemantics
    configuration.stepSemantics configuration.mode configuration.context
    prior next input proof

namespace Invocation

/-- Exact base-branch discriminator for one local invocation. -/
def IsBase
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
    {configuration : Configuration Params StructureDigest Header Digest
      Running Fresh NifsProof Nebula NebulaDigest NebulaOpen}
    (invocation : Invocation configuration) : Prop :=
  invocation.prior.proof = .initial ∧ invocation.proof.fold = .noFold

/-- Exact recursive-branch discriminator for one local invocation. It exposes
the same running batch and proof selected by `Step.LocalHolds`. -/
def IsRecursive
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
    {configuration : Configuration Params StructureDigest Header Digest
      Running Fresh NifsProof Nebula NebulaDigest NebulaOpen}
    (invocation : Invocation configuration) : Prop :=
  ∃ running latest nifsProof,
    invocation.prior.proof = .active running latest ∧
      invocation.proof.fold = .recursive nifsProof

/-- A locally satisfying invocation has exactly one legal branch shape. -/
theorem classified
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
    {configuration : Configuration Params StructureDigest Header Digest
      Running Fresh NifsProof Nebula NebulaDigest NebulaOpen}
    (invocation : Invocation configuration) :
    invocation.IsBase ∨ invocation.IsRecursive := by
  cases priorProof : invocation.prior.proof with
  | initial =>
      cases foldProof : invocation.proof.fold with
      | noFold => exact Or.inl ⟨priorProof, foldProof⟩
      | recursive nifsProof =>
          have localProof := invocation.localHolds
          simp [Step.LocalHolds, priorProof, foldProof] at localProof
  | active running latest =>
      cases foldProof : invocation.proof.fold with
      | noFold =>
          have localProof := invocation.localHolds
          simp [Step.LocalHolds, priorProof, foldProof] at localProof
      | recursive nifsProof =>
          exact Or.inr ⟨running, latest, nifsProof, priorProof, foldProof⟩

/-- The canonical initial prior-proof tag is enough to derive the complete
base discriminator. The local relation rules out a recursive fold for that
tag, so `proof.fold = noFold` is a conclusion. -/
theorem isBase_of_prior_initial
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
    {configuration : Configuration Params StructureDigest Header Digest
      Running Fresh NifsProof Nebula NebulaDigest NebulaOpen}
    (invocation : Invocation configuration)
    (priorInitial : invocation.prior.proof = .initial) :
    invocation.IsBase := by
  cases foldProof : invocation.proof.fold with
  | noFold => exact ⟨priorInitial, foldProof⟩
  | recursive nifsProof =>
      have localProof := invocation.localHolds
      simp [Step.LocalHolds, priorInitial, foldProof] at localProof

/-- Exact adjacency forces the next invocation onto the recursive branch.
The preceding local invocation installs an active batch in its outgoing state,
and the next local relation rejects `noFold` on an active prior state. -/
theorem next_isRecursive
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
    {configuration : Configuration Params StructureDigest Header Digest
      Running Fresh NifsProof Nebula NebulaDigest NebulaOpen}
    (current next : Invocation configuration)
    (continuous : current.next = next.prior) :
    next.IsRecursive := by
  have produced := Step.localHolds_producer_facts
    configuration.hashSemantics configuration.stepSemantics
    configuration.mode configuration.context current.prior current.next
    current.input current.proof current.localHolds
  rcases produced.installedLatest with ⟨running, installed⟩
  have priorActive :
      next.prior.proof = .active running current.input.nextLatest := by
    rw [← continuous]
    exact installed
  cases foldProof : next.proof.fold with
  | noFold =>
      have localProof := next.localHolds
      simp [Step.LocalHolds, priorActive, foldProof] at localProof
  | recursive nifsProof =>
      exact ⟨running, current.input.nextLatest, nifsProof, priorActive,
        foldProof⟩

/-- The delayed obligation for the exact batch produced by this invocation. -/
def OutgoingLinked
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
    {configuration : Configuration Params StructureDigest Header Digest
      Running Fresh NifsProof Nebula NebulaDigest NebulaOpen}
    (invocation : Invocation configuration) : Prop :=
  Step.OutgoingLinked configuration.stepSemantics invocation.input
    invocation.proof

/-- Strong closed-edge relation for this invocation. -/
def Holds
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
    {configuration : Configuration Params StructureDigest Header Digest
      Running Fresh NifsProof Nebula NebulaDigest NebulaOpen}
    (invocation : Invocation configuration) : Prop :=
  Step.Holds configuration.hashSemantics configuration.stepSemantics
    configuration.mode configuration.context invocation.prior invocation.next
    invocation.input invocation.proof

/-- Every local invocation installs the exact produced batch and recomputes
its outgoing digest. -/
theorem producerFacts
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
    {configuration : Configuration Params StructureDigest Header Digest
      Running Fresh NifsProof Nebula NebulaDigest NebulaOpen}
    (invocation : Invocation configuration) :
    Step.LocalProducerFacts configuration.hashSemantics
      configuration.stepSemantics configuration.mode configuration.context
      invocation.next invocation.input invocation.proof :=
  Step.localHolds_producer_facts configuration.hashSemantics
    configuration.stepSemantics configuration.mode configuration.context
    invocation.prior invocation.next invocation.input invocation.proof
    invocation.localHolds

/-- Exact adjacency makes the next local invocation check the preceding
producer's delayed batch. No outgoing-link premise is used. -/
theorem outgoingLinked_of_next
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
    {configuration : Configuration Params StructureDigest Header Digest
      Running Fresh NifsProof Nebula NebulaDigest NebulaOpen}
    (current next : Invocation configuration)
    (continuous : current.next = next.prior) :
    current.OutgoingLinked := by
  rcases current.producerFacts.installedLatest with ⟨running, installed⟩
  have nextPriorActive :
      next.prior.proof = .active running current.input.nextLatest := by
    rw [← continuous]
    exact installed
  have consumed := Step.localHolds_consumes_active_latest
    configuration.hashSemantics configuration.stepSemantics
    configuration.mode configuration.context next.prior next.next next.input
    next.proof running current.input.nextLatest nextPriorActive next.localHolds
  unfold OutgoingLinked Step.OutgoingLinked
  rw [current.producerFacts.recomputedXOut]
  simpa only [continuous] using consumed

/-- Closing a local invocation with its delayed link gives the strong step
relation without adding another semantic premise. -/
theorem close
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
    {configuration : Configuration Params StructureDigest Header Digest
      Running Fresh NifsProof Nebula NebulaDigest NebulaOpen}
    (invocation : Invocation configuration)
    (outgoing : invocation.OutgoingLinked) : invocation.Holds :=
  Step.closeLocal configuration.hashSemantics configuration.stepSemantics
    configuration.mode configuration.context invocation.prior invocation.next
    invocation.input invocation.proof invocation.localHolds outgoing

end Invocation

/-- A complete delayed trace. Adjacent local invocations share the exact
state object. Only the last constructor accepts an outgoing link, because the
terminal relation owns the trailing check. -/
inductive Candidate
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
    (configuration : Configuration Params StructureDigest Header Digest
      Running Fresh NifsProof Nebula NebulaDigest NebulaOpen) :
    Invocation configuration → List (Invocation configuration) → Prop
  | terminal (last : Invocation configuration)
      (terminalLink : last.OutgoingLinked) : Candidate configuration last []
  | recursive {current next : Invocation configuration}
      {rest : List (Invocation configuration)}
      (continuous : current.next = next.prior)
      (tail : Candidate configuration next rest) :
      Candidate configuration current (next :: rest)

/-- The same trace after every delayed link has been recovered. -/
inductive Closed
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
    (configuration : Configuration Params StructureDigest Header Digest
      Running Fresh NifsProof Nebula NebulaDigest NebulaOpen) :
    Invocation configuration → List (Invocation configuration) → Prop
  | terminal (last : Invocation configuration)
      (outgoing : last.OutgoingLinked) (holds : last.Holds) :
      Closed configuration last []
  | recursive {current next : Invocation configuration}
      {rest : List (Invocation configuration)}
      (continuous : current.next = next.prior)
      (outgoing : current.OutgoingLinked) (holds : current.Holds)
      (tail : Closed configuration next rest) :
      Closed configuration current (next :: rest)

namespace Candidate

/-- Every invocation after the first uses the recursive branch. -/
theorem rest_isRecursive
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
    {configuration : Configuration Params StructureDigest Header Digest
      Running Fresh NifsProof Nebula NebulaDigest NebulaOpen}
    {first : Invocation configuration}
    {rest : List (Invocation configuration)}
    (candidate : Candidate configuration first rest) :
    ∀ invocation ∈ rest, invocation.IsRecursive := by
  induction candidate with
  | terminal last terminalLink =>
      intro invocation member
      simp at member
  | @recursive current next rest continuous tail inductionHypothesis =>
      intro invocation member
      have nextRecursive := current.next_isRecursive next continuous
      simp only [List.mem_cons] at member
      rcases member with rfl | member
      · exact nextRecursive
      · exact inductionHypothesis invocation member

/-- Once generated base rows establish the true base discriminator, state
continuity derives the complete base-then-recursive branch schedule. -/
theorem exactBranchSchedule
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
    {configuration : Configuration Params StructureDigest Header Digest
      Running Fresh NifsProof Nebula NebulaDigest NebulaOpen}
    {first : Invocation configuration}
    {rest : List (Invocation configuration)}
    (candidate : Candidate configuration first rest)
    (firstBase : first.IsBase) :
    first.IsBase ∧ ∀ invocation ∈ rest, invocation.IsRecursive :=
  ⟨firstBase, candidate.rest_isRecursive⟩

/-- Global delayed-consumption induction. Every nonterminal link is derived
from the next local invocation. The terminal premise closes only the trailing
batch. -/
theorem closeAll
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
    {configuration : Configuration Params StructureDigest Header Digest
      Running Fresh NifsProof Nebula NebulaDigest NebulaOpen}
    {first : Invocation configuration}
    {rest : List (Invocation configuration)}
    (candidate : Candidate configuration first rest) :
    Closed configuration first rest := by
  induction candidate with
  | terminal last terminalLink =>
      exact .terminal last terminalLink (last.close terminalLink)
  | @recursive current next rest continuous tail inductionHypothesis =>
      have outgoing := current.outgoingLinked_of_next next continuous
      exact .recursive continuous outgoing (current.close outgoing)
        inductionHypothesis

/-- The first producer's delayed link is a conclusion of complete-trace
closure, even when the trace contains recursive invocations. -/
theorem headOutgoingLinked
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
    {configuration : Configuration Params StructureDigest Header Digest
      Running Fresh NifsProof Nebula NebulaDigest NebulaOpen}
    {first : Invocation configuration}
    {rest : List (Invocation configuration)}
    (candidate : Candidate configuration first rest) :
    first.OutgoingLinked := by
  cases candidate.closeAll with
  | terminal _ outgoing _ => exact outgoing
  | recursive _ outgoing _ _ => exact outgoing

end Candidate

end Nightstream.Protocol.FPrime.DelayedTrace
