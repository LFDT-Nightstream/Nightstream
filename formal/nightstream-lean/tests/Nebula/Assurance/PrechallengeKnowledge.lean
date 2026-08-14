import Nightstream.Assurance.Nebula.PrechallengeKnowledge
import Nightstream.Protocol.Nebula

set_option autoImplicit false

namespace tests.NebulaPrechallengeKnowledge

open Nightstream.Assurance.Nebula.PrechallengeKnowledge
open Nightstream.Protocol.Nebula.Lifecycle
open Nightstream.Protocol.Nebula.SequenceBinding

abbrev TestSequence := FramedSequence Bool Unit Bool

def commitments : Fin claimsPerSegment → Bool := fun _ => false

def sequence : TestSequence where
  profile := false
  plan := ()
  domain := .operations
  commitments := commitments

def identityRoot (value : TestSequence) : TestSequence := value

def completeExtractor :
    Extractor Bool Unit Bool TestSequence identityRoot where
  extract := some
  correct := by
    intro root extracted equal
    simpa using (congrArg id equal).symm

def replay : CheckedReplay identityRoot where
  sequence := sequence
  seenRoot := sequence
  rootCorrect := rfl

theorem complete_extractor_binds_replay :
    ∃ precommit : KnownPrecommit identityRoot,
      precommit.committedRoot = sequence ∧
      completeExtractor.extract sequence = some precommit.sequence ∧
      replay.sequence = precommit.sequence := by
  rcases close_binds_extracted_sequence_or_named_failure
      completeExtractor sequence replay rfl with exact | collision | failure
  · exact exact
  · rcases collision with ⟨left, right, different, equal⟩
    exact False.elim (different equal)
  · contradiction

/- A digest-only protocol can accept a root for which the fixed extractor has
no preimage. The theorem reports this as the exact knowledge failure; it does
not manufacture a sequence after the replay is known. -/
def emptyExtractor :
    Extractor Bool Unit Bool TestSequence identityRoot where
  extract := fun _ => none
  correct := by simp

theorem digest_without_extracted_preimage_is_named_failure :
    KnowledgeFailure emptyExtractor sequence :=
  rfl

end tests.NebulaPrechallengeKnowledge
