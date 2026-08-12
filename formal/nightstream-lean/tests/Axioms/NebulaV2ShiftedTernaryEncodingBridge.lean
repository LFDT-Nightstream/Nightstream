import Nightstream.Implementation.NebulaV2.ShiftedTernaryEncodingBridge
import tests.Axioms.Support

open Nightstream.Implementation.NebulaV2

/-- info: 'Nightstream.Implementation.NebulaV2.ShiftedTernaryEncodingBridge.trits_getD_eq_quotient' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ShiftedTernaryEncodingBridge.trits_getD_eq_quotient

/-- info: 'Nightstream.Implementation.NebulaV2.ShiftedTernaryEncodingBridge.canonicalDigit_eq_fieldDigit_tritAt' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ShiftedTernaryEncodingBridge.canonicalDigit_eq_fieldDigit_tritAt

/-- info: 'Nightstream.Implementation.NebulaV2.ShiftedTernaryEncodingBridge.productionDigit_eq_protocolDigit' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ShiftedTernaryEncodingBridge.productionDigit_eq_protocolDigit
