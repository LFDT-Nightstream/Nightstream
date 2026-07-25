import tests.FPrimeProductionEncodeInstanceRecipe
import tests.Axioms.Support

/-!
Fail-closed guards for the concrete six-row fixed-one `encodeInstance`
recipe.
-/

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodeInstanceRecipe.row_count' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodeInstanceRecipe.row_count

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodeInstanceRecipe.footprint_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodeInstanceRecipe.footprint_exact

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodeInstanceRecipe.rows_owned' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodeInstanceRecipe.rows_owned

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodeInstanceRecipe.row_ids_nodup' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodeInstanceRecipe.row_ids_nodup

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodeInstanceRecipe.rows_supported' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodeInstanceRecipe.rows_supported

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodeInstanceRecipe.active_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodeInstanceRecipe.active_sound

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodeInstanceRecipe.active_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodeInstanceRecipe.active_complete

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodeInstanceRecipe.inactive_complete' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodeInstanceRecipe.inactive_complete
