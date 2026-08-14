import Nightstream.Protocol.Nebula.SequenceBinding

/-!
Contract: staged prechallenge knowledge boundary for a V2 lane root.

Assurance tier: cryptographic-reduction boundary.

Owns a fixed extractor, its exact root-correctness condition, the named
failure when it cannot open a committed root, and the deterministic reduction
from an accepted replay to sequence equality, a root collision, or that
knowledge failure.

The extractor is an input to the prechallenge stage. Challenge derivation can
receive the extracted root and public frame, but it does not receive the later
replay witness. This type separation records the protocol order. It does not
prove a Poseidon2 knowledge, late-preimage, or random-oracle probability
bound. It does not assume equality of the prechallenge and replay sequences.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Assurance.Nebula.PrechallengeKnowledge

open Nightstream.Protocol.Nebula.SequenceBinding

abbrev Sequence (Profile Plan Commitment : Type) :=
  FramedSequence Profile Plan Commitment

/-- A reduction-owned extractor fixed before the memory challenge is
released. `correct` is only an opening equation. It contains no replay or
sequence-equality conclusion. -/
structure Extractor
    (Profile Plan Commitment Digest : Type)
    (chainRoot : Sequence Profile Plan Commitment → Digest) where
  extract : Digest → Option (Sequence Profile Plan Commitment)
  correct : ∀ {root sequence}, extract root = some sequence →
    chainRoot sequence = root

/-- Exact knowledge failure for one committed root. A security proof must
bound this event in the prechallenge/late-preimage game. -/
def KnowledgeFailure
    {Profile Plan Commitment Digest : Type}
    {chainRoot : Sequence Profile Plan Commitment → Digest}
    (extractor : Extractor Profile Plan Commitment Digest chainRoot)
    (committedRoot : Digest) : Prop :=
  extractor.extract committedRoot = none

/-- Successful extraction constructs the `KnownPrecommit` object required by
the deterministic sequence theorem. Otherwise the exact knowledge failure is
returned. -/
theorem known_precommit_or_knowledge_failure
    {Profile Plan Commitment Digest : Type}
    {chainRoot : Sequence Profile Plan Commitment → Digest}
    (extractor : Extractor Profile Plan Commitment Digest chainRoot)
    (committedRoot : Digest) :
    (∃ precommit : KnownPrecommit chainRoot,
      precommit.committedRoot = committedRoot ∧
      extractor.extract committedRoot = some precommit.sequence) ∨
      KnowledgeFailure extractor committedRoot := by
  cases extracted : extractor.extract committedRoot with
  | none => exact Or.inr extracted
  | some sequence =>
      exact Or.inl
        ⟨{ sequence := sequence
           committedRoot := committedRoot
           rootCorrect := extractor.correct extracted },
          rfl, rfl⟩

/-- A checked replay that closes against a prechallenge root is bound to one
extracted complete sequence. The only other deterministic outcomes are a root
collision or the named knowledge failure. -/
theorem close_binds_extracted_sequence_or_named_failure
    {Profile Plan Commitment Digest : Type}
    {chainRoot : Sequence Profile Plan Commitment → Digest}
    (extractor : Extractor Profile Plan Commitment Digest chainRoot)
    (committedRoot : Digest)
    (replay : CheckedReplay chainRoot)
    (closeRoot : replay.seenRoot = committedRoot) :
    (∃ precommit : KnownPrecommit chainRoot,
      precommit.committedRoot = committedRoot ∧
      extractor.extract committedRoot = some precommit.sequence ∧
      replay.sequence = precommit.sequence) ∨
      RootCollision chainRoot ∨ KnowledgeFailure extractor committedRoot := by
  rcases known_precommit_or_knowledge_failure extractor committedRoot with
    ⟨precommit, rootExact, extracted⟩ | failure
  · have closeAgainstPrecommit :
        replay.seenRoot = precommit.committedRoot := by
      exact closeRoot.trans rootExact.symm
    rcases close_binds_exact_sequence_or_collision precommit replay
        closeAgainstPrecommit with equalSequence | collision
    · exact Or.inl ⟨precommit, rootExact, extracted, equalSequence⟩
    · exact Or.inr (Or.inl collision)
  · exact Or.inr (Or.inr failure)

end Nightstream.Assurance.Nebula.PrechallengeKnowledge
