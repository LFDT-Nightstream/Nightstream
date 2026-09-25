import NightstreamFPrime.Export.Stage1.VerifierContextCandidate
import NightstreamFPrime.Export.Stage1.Data
import NightstreamFPrime.Lifecycle.VerifierContext

/-!
Owns the canonical verifier-context recipe for the selected Stage 1 package.

The package identity selects the exact Lean-emitted package rows. The current
package uses that identity in the relation and application component positions;
it does not yet prove that the prefix contains the final logical relation or
application transition. That connection remains a required Stage 1 fixed-point
edge. The NIFS component also binds the fixed profile, digest-only transcript
schedule, package identity, and commitment-key component digest. The final
component hashes the verifier-owned commitment setup serialization itself.

The sealed package identity and the resulting context digest remain distinct.
The package contains public context columns, not fixed context values, so this
recipe is non-self-referential.
-/

namespace NightstreamFPrime.Export.Stage1.VerifierContext

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-- Trusted implementation-link condition checked by the strict Rust loader.
It is not a semantic axiom and is not used to prove row soundness. -/
def PackageIdentityHolds : Prop :=
  packageIdentityWords = Data.relationIdentifier ()

theorem authority_relationWords_of_packageIdentity
    (commitmentKeyWords : List F) (holds : PackageIdentityHolds) :
    (authority commitmentKeyWords).relationWords = Data.relationIdentifier () := by
  exact holds

theorem authority_applicationWords_of_packageIdentity
    (commitmentKeyWords : List F) (holds : PackageIdentityHolds) :
    (authority commitmentKeyWords).applicationWords = Data.relationIdentifier () := by
  exact holds

end NightstreamFPrime.Export.Stage1.VerifierContext
