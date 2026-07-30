import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsActivatedProgram
import tests.Axioms.Support

namespace NightstreamTests.Axioms.ConcreteNifsActivatedProgram

open NightstreamTests.Axioms
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsActivatedProgram.selected_footprint_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsActivatedProgram.selected_footprint_exact

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsActivatedProgram.allocation_coverage' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsActivatedProgram.allocation_coverage

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsActivatedProgram.rows_length' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsActivatedProgram.rows_length

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsActivatedProgram.rows_owned' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsActivatedProgram.rows_owned

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsActivatedProgram.rowIds_nodup' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsActivatedProgram.rowIds_nodup

end NightstreamTests.Axioms.ConcreteNifsActivatedProgram
