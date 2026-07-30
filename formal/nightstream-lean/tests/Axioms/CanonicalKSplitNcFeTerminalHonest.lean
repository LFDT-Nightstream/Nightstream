import Nightstream.Implementation.R1CS.Canonical.KSplitNcFeTerminalProductsHonest
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.CanonicalKSplitNcFeTerminalHonest

open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcFeTerminalHonest.afterCarriedRow_off_source' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcFeTerminalHonest.afterCarriedRow_off_source

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcFeTerminalHonest.pointPrefix_honest' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcFeTerminalHonest.pointPrefix_honest

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcFeTerminalProductsHonest.computedRows_honest' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcFeTerminalProductsHonest.computedRows_honest

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcFeTerminalProductsHonest.rows_eq_computedRows_append_terminal' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcFeTerminalProductsHonest.rows_eq_computedRows_append_terminal

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcFeTerminalProductsHonest.rows_honest_of_binding' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcFeTerminalProductsHonest.rows_honest_of_binding

end NightstreamTests.Axioms.CanonicalKSplitNcFeTerminalHonest
