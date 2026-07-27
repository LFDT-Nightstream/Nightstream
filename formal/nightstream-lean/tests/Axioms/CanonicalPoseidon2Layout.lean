import Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout
import tests.Axioms.Support

/-!
Fail-closed dependency gate for Poseidon2 layout well-formedness and
row ownership uniqueness.
No theorem may acquire `Lean.trustCompiler`.
-/

namespace NightstreamTests.Axioms.CanonicalPoseidon2Layout

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout.auxiliaryColumns_ge' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Layout.auxiliaryColumns_ge

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout.inputPort_not_auxiliary' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Layout.inputPort_not_auxiliary

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout.outputPort_not_auxiliary' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Layout.outputPort_not_auxiliary

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout.constantWire_not_auxiliary' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Layout.constantWire_not_auxiliary

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout.canonicalLayout_wellFormed' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Layout.canonicalLayout_wellFormed

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout.canonicalColumnTotal_eq' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Layout.canonicalColumnTotal_eq

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout.canonicalLayout_contiguous' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Layout.canonicalLayout_contiguous

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout.sboxRows_target' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2Layout.sboxRows_target

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout.sboxRows_disjoint' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Layout.sboxRows_disjoint

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout.bindRow_not_sboxRow' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Layout.bindRow_not_sboxRow

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout.bindRow_injective' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2Layout.bindRow_injective

/-! Shifted layouts and per-call disjointness. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout.shiftedLayout_wellFormed' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Layout.shiftedLayout_wellFormed

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout.shiftedLayout_aux_disjoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Layout.shiftedLayout_aux_disjoint

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout.canonicalLayout_eq_shifted' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Layout.canonicalLayout_eq_shifted

end NightstreamTests.Axioms.CanonicalPoseidon2Layout
