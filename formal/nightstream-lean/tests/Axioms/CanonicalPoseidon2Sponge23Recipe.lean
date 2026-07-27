import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalPoseidon2Sponge23RecipeAudit
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalPoseidon2Sponge23Recipe

open NightstreamTests.Axioms
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalPoseidon2Sponge23Recipe.rows_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalPoseidon2Sponge23Recipe.rows_length

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalPoseidon2Sponge23Recipe.rowIds_nodup' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalPoseidon2Sponge23Recipe.rowIds_nodup

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalPoseidon2Sponge23Recipe.active_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalPoseidon2Sponge23Recipe.active_sound

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalPoseidon2Sponge23Recipe.active_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalPoseidon2Sponge23Recipe.active_complete

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalPoseidon2Sponge23Recipe.inactive_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalPoseidon2Sponge23Recipe.inactive_complete

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalPoseidon2Sponge23Recipe.rows_supported' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalPoseidon2Sponge23Recipe.rows_supported

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalPoseidon2Sponge23Recipe.receipt_row_column_conservation' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalPoseidon2Sponge23Recipe.receipt_row_column_conservation

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalPoseidon2Sponge23Recipe.standaloneCost_matches_receipt' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalPoseidon2Sponge23Recipe.standaloneCost_matches_receipt

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalPoseidon2Sponge23Recipe.exact_nonzero_coefficient_count' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalPoseidon2Sponge23Recipe.exact_nonzero_coefficient_count

end NightstreamTests.Axioms.CanonicalPoseidon2Sponge23Recipe
