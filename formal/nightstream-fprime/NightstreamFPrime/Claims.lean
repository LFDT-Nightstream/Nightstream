import NightstreamFPrime.Claims.Foundations
import NightstreamFPrime.Claims.Encodings
import NightstreamFPrime.Claims.Fold
import NightstreamFPrime.Claims.Step
import NightstreamFPrime.Claims.Ivc
import NightstreamFPrime.Claims.Nebula
import NightstreamFPrime.Claims.Compression
import NightstreamFPrime.Claims.Akita
import NightstreamFPrime.Claims.AkitaKey20
import NightstreamFPrime.Claims.AkitaKey30
import NightstreamFPrime.Claims.Final
import NightstreamFPrime.Claims.Shipped

/-! Nightstream claims exactly the theorems of `NightstreamFPrime.Claims`. Each other theorem of
the package is a lemma. A claim states one result with each hypothesis in its signature, and its
proof is one application of the theorem that proves it. `tests/PremiseCensus.lean` reads this
namespace: it audits the axioms of each claim, checks the form of each proof, and lists each
hypothesis for `SECURITY_MODEL.md`, section Premises. -/
