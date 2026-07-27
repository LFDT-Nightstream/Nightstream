import Nightstream.Implementation.R1CS.Canonical.Poseidon2Conservation
import tests.Axioms.Support

/-!
Fail-closed dependency gate for column conservation.
No theorem may acquire `Lean.trustCompiler`.
-/

namespace NightstreamTests.Axioms.CanonicalPoseidon2Conservation

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Conservation.scheduleOf_conservation' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Conservation.scheduleOf_conservation

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Conservation.finalState_conservation' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Conservation.finalState_conservation

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Conservation.scheduleOf_columns' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Conservation.scheduleOf_columns

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Conservation.initialState_columns' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Conservation.initialState_columns

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Conservation.partialState_columns' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Conservation.partialState_columns

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Conservation.terminalState_columns' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Conservation.terminalState_columns

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Conservation.canonicalLayout_sboxColumn_lt' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Conservation.canonicalLayout_sboxColumn_lt

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Conservation.canonicalLayout_sboxOutput_lt' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Conservation.canonicalLayout_sboxOutput_lt

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Conservation.canonicalLayout_inputPort_lt' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Conservation.canonicalLayout_inputPort_lt

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Conservation.canonicalLayout_outputPort_lt' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Conservation.canonicalLayout_outputPort_lt

/-! Carried-entry column classification. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Conservation.scheduleOfFrom_columns' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Conservation.scheduleOfFrom_columns

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Conservation.initialStateFrom_columns' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Conservation.initialStateFrom_columns

end NightstreamTests.Axioms.CanonicalPoseidon2Conservation
