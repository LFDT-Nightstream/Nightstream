import Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
import tests.Axioms.Support

/-!
Fail-closed dependency gate for the canonical Poseidon2 permutation encoding.

No theorem here may acquire `Lean.trustCompiler`: the point of the canonical
encoding is a cost that does not inherit the generated artifact's number.
-/

namespace NightstreamTests.Axioms.CanonicalPoseidon2Core

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Core.sboxRows_chain' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2Core.sboxRows_chain

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Core.sboxRows_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2Core.sboxRows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Core.applyMatrix_emits_no_rows' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Core.applyMatrix_emits_no_rows

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Core.terminalBindingRows_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2Core.terminalBindingRows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Core.sboxCount_eq' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Core.sboxCount_eq

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Core.nonlinearRows_eq' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Core.nonlinearRows_eq

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Core.permutationRows_eq' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Core.permutationRows_eq



/-! Derived subtotals and forecasts.  The auxiliary-column count is
deliberately absent: it is unresolved pending the assembled receipt. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Core.nonlinearRowSubtotal' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Core.nonlinearRowSubtotal

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Core.totalRowForecast' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Core.totalRowForecast

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Core.nonlinearRows_lt_permutationRows' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Core.nonlinearRows_lt_permutationRows

end NightstreamTests.Axioms.CanonicalPoseidon2Core
