import Nightstream.Implementation.R1CS.Canonical.KStrictNormSequentialHonest
import Nightstream.Implementation.R1CS.Canonical.KSplitNcNcArithmeticHonest
import Nightstream.Implementation.R1CS.Canonical.KSplitNcNcEndpointHonest
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.CanonicalKSplitNcNcHonest

open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KStrictNormSequentialHonest.rowsFrom_honest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KStrictNormSequentialHonest.rowsFrom_honest

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcNcArithmeticHonest.rows_honest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcNcArithmeticHonest.rows_honest

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcNcEndpointHonest.computedRows_honest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcNcEndpointHonest.computedRows_honest

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcNcEndpointHonest.rows_honest_of_binding' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcNcEndpointHonest.rows_honest_of_binding

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcNcEndpointHonest.rows_eq_initial_append_computed_append_terminal' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms
  KSplitNcNcEndpointHonest.rows_eq_initial_append_computed_append_terminal

end NightstreamTests.Axioms.CanonicalKSplitNcNcHonest
