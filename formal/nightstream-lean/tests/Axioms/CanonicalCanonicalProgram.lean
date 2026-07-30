import Nightstream.Implementation.R1CS.Canonical.CanonicalProgram
import tests.Axioms.Support

/-!
Fail-closed axiom guard for the canonical program assembly.

Every report below is measured, not asserted: the expected text was produced by
running the audit and copying its output, so any drift fails the build.
-/

namespace NightstreamTests.Axioms.CanonicalCanonicalProgram

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.rows_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.rows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.rows_length_built_only' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.rows_length_built_only

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.satisfies_piDec' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.satisfies_piDec

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.satisfies_foldDigest' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.satisfies_foldDigest

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.satisfies_mixer' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.satisfies_mixer

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.satisfies_transcript' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.satisfies_transcript

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.windows_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.windows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.windows_no_collision' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.windows_no_collision

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.windows_exclude_constantWire' does not depend on any axioms -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.windows_exclude_constantWire

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.satisfies_step' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.satisfies_step

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.satisfies_nifsVerify' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.satisfies_nifsVerify

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.satisfies_runningCheck' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.satisfies_runningCheck

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.satisfies_freshCheck' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.satisfies_freshCheck

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.satisfies_every_part' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.satisfies_every_part

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.rows_honest' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.rows_honest

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.rows_iff_every_part' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.rows_iff_every_part

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.committedColumns_from_selections' does not depend on any axioms -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.committedColumns_from_selections

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.publicColumns_from_selections' does not depend on any axioms -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.publicColumns_from_selections

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.built_recipes_allocate_no_public_columns' does not depend on any axioms -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.built_recipes_allocate_no_public_columns

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.N_canonical_components' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.N_canonical_components

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.selection_may_duplicate_built_rows' does not depend on any axioms -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.selection_may_duplicate_built_rows

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.rows_owner_not_unique' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.rows_owner_not_unique

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.auxiliaryColumns_eq_counts_sum' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.auxiliaryColumns_eq_counts_sum

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.foldDigest_placed' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.foldDigest_placed

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.mixer_placed' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.mixer_placed

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.transcript_placed' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.transcript_placed

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.selection_placed' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.selection_placed

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.piDec_placed' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.piDec_placed

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.parts_widths' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.parts_widths

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.placeAll_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.placeAll_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.placedRows_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.placedRows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.placedRows_length_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.placedRows_length_eq

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.placed_satisfies_piDec' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.placed_satisfies_piDec

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.placed_satisfies_foldDigest' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.placed_satisfies_foldDigest

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.placed_satisfies_at' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.placed_satisfies_at

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.placed_satisfies_mixer' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.placed_satisfies_mixer

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.placed_satisfies_transcript' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.placed_satisfies_transcript

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.placed_satisfies_step' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.placed_satisfies_step

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.placed_satisfies_nifsVerify' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.placed_satisfies_nifsVerify

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.placed_satisfies_runningCheck' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.placed_satisfies_runningCheck

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.placed_satisfies_freshCheck' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.placed_satisfies_freshCheck

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.placed_satisfies_every_part' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.placed_satisfies_every_part

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.placedRows_honest' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.placedRows_honest

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.foldDigest_placed_at_its_base' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.foldDigest_placed_at_its_base

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.mixer_placed_at_its_base' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.mixer_placed_at_its_base

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.transcript_placed_at_its_base' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.transcript_placed_at_its_base

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.piDec_placed_at_its_base' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.piDec_placed_at_its_base

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.selection_placed_at_its_base' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.selection_placed_at_its_base


/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.builtRows_conservation' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.builtRows_conservation

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.rows_eq_builtRows_append_selections' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.rows_eq_builtRows_append_selections


/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.allocations_disjoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.allocations_disjoint

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.foldDigest_and_mixer_allocate_nothing' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.foldDigest_and_mixer_allocate_nothing


/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalProgram.Recipes.builtRows_use_piDec_columns' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalProgram.Recipes.builtRows_use_piDec_columns

end NightstreamTests.Axioms.CanonicalCanonicalProgram
