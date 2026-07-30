import Nightstream.Implementation.R1CS.Canonical.KMulHonest
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKMulHonest

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KMulHonest.lcEval_congr' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KMulHonest.lcEval_congr

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KMulHonest.canonical_distinct' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KMulHonest.canonical_distinct

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KMulHonest.witness_satisfies' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KMulHonest.witness_satisfies

end NightstreamTests.Axioms.CanonicalKMulHonest
