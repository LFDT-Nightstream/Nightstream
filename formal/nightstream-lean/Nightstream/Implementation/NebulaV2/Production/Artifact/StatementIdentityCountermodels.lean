import Nightstream.Implementation.NebulaV2.Memory.Transition.OpenSegment
import Nightstream.Protocol.NebulaV2.ProductionProfileCandidates

/-!
Contract: hostile model for checking only the verifier-key portion of a V2
statement identity.

The memory-challenge transcript also binds the application relation, program,
and memory plan. Two statements can share the complete verifier-key record
while one of those three digests differs. Therefore verifier-key equality
does not imply equality of the static memory-challenge authority.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ProductionStatementIdentityCountermodels

open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.Soundness

def zeroDigest : Digest.Value :=
  { lanes := fun _ => ⟨0, by decide⟩ }

def oneDigest : Digest.Value :=
  { lanes := fun _ => ⟨1, by decide⟩ }

def verifierKey : VerifierKeyIdentity Digest.Value :=
  { digest := zeroDigest
    relationManifestDigest := zeroDigest
    laneLayoutDigest := zeroDigest
    setupManifestDigest := zeroDigest
    transcriptManifestDigest := zeroDigest
    codecManifestDigest := zeroDigest
    terminalManifestDigest := zeroDigest
    applicationStateSchemaDigest := zeroDigest }

def leftIdentity : StatementIdentity Digest.Value :=
  { profile := ProductionProfileCandidates.identity .e1
    verifierKey := verifierKey
    applicationRelationDigest := zeroDigest
    programDigest := zeroDigest
    memoryPlanDigest := zeroDigest }

def rightIdentity : StatementIdentity Digest.Value :=
  { leftIdentity with programDigest := oneDigest }

theorem zeroDigest_ne_oneDigest : zeroDigest ≠ oneDigest := by
  intro equal
  have lane := congrArg (fun digest => (digest.lanes (0 : Fin 4)).val) equal
  norm_num [zeroDigest, oneDigest] at lane

theorem same_verifier_key :
    leftIdentity.verifierKey = rightIdentity.verifierKey :=
  rfl

theorem identities_differ : leftIdentity ≠ rightIdentity := by
  intro equal
  have digestEqual := congrArg StatementIdentity.programDigest equal
  exact zeroDigest_ne_oneDigest digestEqual

theorem challenge_authorities_differ :
    MemoryOpenSegment.Authority.ofIdentityAndState leftIdentity zeroDigest
        zeroDigest ≠
      MemoryOpenSegment.Authority.ofIdentityAndState rightIdentity zeroDigest
        zeroDigest := by
  intro equal
  have digestEqual :=
    congrArg MemoryOpenSegment.Authority.programDigest equal
  exact zeroDigest_ne_oneDigest digestEqual

/-- A full verifier-key match leaves three independent statement digests
unbound. The exact statement-identity equality is necessary. -/
theorem verifier_key_equality_does_not_bind_challenge_identity :
    ∃ left right : StatementIdentity Digest.Value,
      left.verifierKey = right.verifierKey ∧
        left ≠ right ∧
        MemoryOpenSegment.Authority.ofIdentityAndState left zeroDigest
            zeroDigest ≠
          MemoryOpenSegment.Authority.ofIdentityAndState right zeroDigest
            zeroDigest := by
  exact ⟨leftIdentity, rightIdentity, same_verifier_key, identities_differ,
    challenge_authorities_differ⟩

end Nightstream.Implementation.NebulaV2.ProductionStatementIdentityCountermodels
