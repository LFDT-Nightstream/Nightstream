import Nightstream.Implementation.R1CS.Canonical.PiDecRecipe
import tests.Axioms.Support

/-!
Fail-closed axiom guard for the assembled Pi_DEC recipe.

Every report below is measured, not asserted: the expected text was produced by
running the audit and copying its output, so any drift fails the build.
-/

namespace NightstreamTests.Axioms.CanonicalPiDecRecipe

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiDecRecipe.lcEval_fresh' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms PiDecRecipe.lcEval_fresh

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiDecRecipe.carriedValue_fresh' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms PiDecRecipe.carriedValue_fresh

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiDecRecipe.inactiveRows_length' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms PiDecRecipe.inactiveRows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiDecRecipe.rows_length' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms PiDecRecipe.rows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiDecRecipe.columns_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms PiDecRecipe.columns_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiDecRecipe.columns_nodup' does not depend on any axioms -/
#guard_msgs in
#audit_axioms PiDecRecipe.columns_nodup

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiDecRecipe.satisfies_parts' does not depend on any axioms -/
#guard_msgs in
#audit_axioms PiDecRecipe.satisfies_parts

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiDecRecipe.rows_sound_recomposition' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiDecRecipe.rows_sound_recomposition

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiDecRecipe.rows_sound_lowNorm' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms PiDecRecipe.rows_sound_lowNorm

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiDecRecipe.rows_sound_inactive' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms PiDecRecipe.rows_sound_inactive

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiDecRecipe.rows_sound_padding' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms PiDecRecipe.rows_sound_padding

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiDecRecipe.rows_sound_consistency' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms PiDecRecipe.rows_sound_consistency

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiDecRecipe.rows_honest' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms PiDecRecipe.rows_honest

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiDecRecipe.cost_rows' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms PiDecRecipe.cost_rows

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiDecRecipe.cost_columns' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms PiDecRecipe.cost_columns

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiDecRecipe.recompositions_length_without_sidecars' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms PiDecRecipe.recompositions_length_without_sidecars

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiDecRecipe.rows_length_without_sidecars' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiDecRecipe.rows_length_without_sidecars

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiDecRecipe.rows_owner_not_unique' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms PiDecRecipe.rows_owner_not_unique


/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiDecRecipe.rows_conservation' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms PiDecRecipe.rows_conservation


/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiDecRecipe.rows_use_columns' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms PiDecRecipe.rows_use_columns

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiDecRecipe.columns_exact' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms PiDecRecipe.columns_exact

end NightstreamTests.Axioms.CanonicalPiDecRecipe
