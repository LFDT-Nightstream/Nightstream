import Nightstream.Implementation.NebulaV2.ProductNifsRunningParserCorrect
import tests.Axioms.Support

open Nightstream.Implementation.NebulaV2.ProductNifsRunningParser

/-- info: 'Nightstream.Implementation.NebulaV2.ProductNifsRunningParser.exact_section_counts' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms exact_section_counts

/-- info: 'Nightstream.Implementation.NebulaV2.ProductNifsRunningParser.bundleCodec_getD' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms bundleCodec_getD

/-- info: 'Nightstream.Implementation.NebulaV2.ProductNifsRunningParser.runningOfFields_fieldsOfRunning' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms runningOfFields_fieldsOfRunning

/-- info: 'Nightstream.Implementation.NebulaV2.ProductNifsRunningParser.parse_blockOfRunning' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms parse_blockOfRunning

/-- info: 'Nightstream.Implementation.NebulaV2.ProductNifsRunningParser.parse_rejects_modulus_word' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms parse_rejects_modulus_word
