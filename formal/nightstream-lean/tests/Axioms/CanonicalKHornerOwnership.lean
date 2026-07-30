import Nightstream.Implementation.R1CS.Canonical.KHornerOwnership
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKHornerOwnership

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KHornerOwnership.receipts_length' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KHornerOwnership.receipts_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KHornerOwnership.hornerRows_eq_map_receipts' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KHornerOwnership.hornerRows_eq_map_receipts

end NightstreamTests.Axioms.CanonicalKHornerOwnership
