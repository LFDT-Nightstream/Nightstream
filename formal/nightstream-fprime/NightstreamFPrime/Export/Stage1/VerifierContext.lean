import NightstreamFPrime.Export.Stage1.VerifierContextCandidate
import NightstreamFPrime.Export.Stage1.Data
import NightstreamFPrime.Lifecycle.VerifierContext

/-!
Owns the conditional link from the reference-prefix context fixture to `Data`.
`PackageIdentityHolds` is an explicit premise for these two prefix lemmas.
The selected wide application package derives its production context through
`PerApplicationCanonicalPackage`; these fixture identities are not its pins.
-/

namespace NightstreamFPrime.Export.Stage1.VerifierContext

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-- Reference-prefix identity condition. It is not a semantic axiom and is
not used to prove row soundness. -/
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
