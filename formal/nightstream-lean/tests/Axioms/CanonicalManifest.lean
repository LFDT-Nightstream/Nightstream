import Nightstream.Implementation.Lowering.Goldilocks.CanonicalManifest
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalManifest

open NightstreamTests.Axioms
open Nightstream.Implementation.Lowering.Goldilocks.CanonicalManifest

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.CanonicalManifest.eval_normalizeCombination' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms eval_normalizeCombination

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.CanonicalManifest.normalizeRow_holds_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms normalizeRow_holds_iff

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.CanonicalManifest.Program.decode_ofEncoding' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Program.decode_ofEncoding

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.CanonicalManifest.Program.decoded_satisfies_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Program.decoded_satisfies_iff

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.CanonicalManifest.Program.all_coefficients_nonzero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Program.all_coefficients_nonzero

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.CanonicalManifest.Program.all_coefficients_canonical' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Program.all_coefficients_canonical

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.CanonicalManifest.Program.all_combination_columns_nodup' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Program.all_combination_columns_nodup

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.CanonicalManifest.Program.columns_ofEncoding' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Program.columns_ofEncoding

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.CanonicalManifest.Program.rows_length_ofEncoding' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Program.rows_length_ofEncoding

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.CanonicalManifest.Program.cost_ofEncoding' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Program.cost_ofEncoding

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.CanonicalManifest.Program.cost_recurringRows' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Program.cost_recurringRows

end NightstreamTests.Axioms.CanonicalManifest
