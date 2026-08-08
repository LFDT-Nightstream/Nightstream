import Nightstream.Implementation.Rust.NifsProductionGolden
import tests.Axioms.Support

namespace NightstreamTests.Axioms.NifsProductionGolden

open Nightstream.Implementation.Rust.NifsProductionGolden

/-- info: 'Nightstream.Implementation.Rust.NifsProductionGolden.Poseidon2Trace.output_eq_reference' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Trace.output_eq_reference

/-- info: 'Nightstream.Implementation.Rust.NifsProductionGolden.PiCcsReplay.replay?_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiCcsReplay.replay?_sound

/-- info: 'Nightstream.Implementation.Rust.NifsProductionGolden.PiRlcReplay.handoff?_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcReplay.handoff?_sound

/-- info: 'Nightstream.Implementation.Rust.NifsProductionGolden.PiRlcReplay.sample?_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcReplay.sample?_sound

/-- info: 'Nightstream.Implementation.Rust.NifsProductionGolden.PiCcsChecker.checkReceipt_sound' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiCcsChecker.checkReceipt_sound

/-- info: 'Nightstream.Implementation.Rust.NifsProductionGolden.PiRlcChecker.checkReceipt_sound' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcChecker.checkReceipt_sound

/-- info: 'Nightstream.Implementation.Rust.NifsProductionGolden.PiDecChecker.checkReceipt_sound' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiDecChecker.checkReceipt_sound

/-- info: 'Nightstream.Implementation.Rust.NifsProductionGolden.ExecutionChecker.checkReceipt_sound' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ExecutionChecker.checkReceipt_sound

end NightstreamTests.Axioms.NifsProductionGolden
