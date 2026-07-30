import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRawProgram
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.ConcreteNifsRawProgram

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRawProgram.rawRows_length' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsRawProgram.rawRows_length

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRawProgram.rows_owned' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsRawProgram.rows_owned

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRawProgram.rowIds_nodup' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsRawProgram.rowIds_nodup

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRawProgram.allocation_length' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsRawProgram.allocation_length

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRawProgram.rawRows_supported' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsRawProgram.rawRows_supported

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRawProgram.rows_supported' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsRawProgram.rows_supported

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRawProgram.allocation_eq_temporaryPrefix' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsRawProgram.allocation_eq_temporaryPrefix

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRawProgram.allocation_nodup' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsRawProgram.allocation_nodup

end NightstreamTests.Axioms.ConcreteNifsRawProgram
