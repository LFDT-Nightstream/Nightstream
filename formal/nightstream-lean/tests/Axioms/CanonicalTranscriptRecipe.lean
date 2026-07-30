import Nightstream.Implementation.R1CS.Canonical.TranscriptRecipe
import tests.Axioms.Support

/-!
Fail-closed axiom guard for TranscriptRecipe.

Every report below is measured, not asserted: the expected text was produced by
running the audit and copying its output, so any drift fails the build.
-/

namespace NightstreamTests.Axioms.CanonicalTranscriptRecipe

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TranscriptRecipe.transcriptRows_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TranscriptRecipe.transcriptRows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TranscriptRecipe.transcriptRows_length_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TranscriptRecipe.transcriptRows_length_eq

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TranscriptRecipe.satisfies_round' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms TranscriptRecipe.satisfies_round

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TranscriptRecipe.round_computes_reference' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TranscriptRecipe.round_computes_reference

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TranscriptRecipe.entry_carried' does not depend on any axioms -/
#guard_msgs in
#audit_axioms TranscriptRecipe.entry_carried

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TranscriptRecipe.entry_overwritten' does not depend on any axioms -/
#guard_msgs in
#audit_axioms TranscriptRecipe.entry_overwritten

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TranscriptRecipe.entry_initial' does not depend on any axioms -/
#guard_msgs in
#audit_axioms TranscriptRecipe.entry_initial

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TranscriptRecipe.chain_value' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms TranscriptRecipe.chain_value

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TranscriptRecipe.absorbed_value' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms TranscriptRecipe.absorbed_value

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TranscriptRecipe.transcriptCost_rows' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TranscriptRecipe.transcriptCost_rows

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TranscriptRecipe.transcriptRows_honest' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TranscriptRecipe.transcriptRows_honest

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TranscriptRecipe.transcriptRows_conservation' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TranscriptRecipe.transcriptRows_conservation

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TranscriptRecipe.canonicalLayouts_eq_spongeCall' does not depend on any axioms -/
#guard_msgs in
#audit_axioms TranscriptRecipe.canonicalLayouts_eq_spongeCall

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TranscriptRecipe.canonicalLayouts_wellFormed' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TranscriptRecipe.canonicalLayouts_wellFormed

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TranscriptRecipe.canonicalLayouts_disjoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TranscriptRecipe.canonicalLayouts_disjoint

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TranscriptRecipe.canonicalTranscriptRows_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TranscriptRecipe.canonicalTranscriptRows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TranscriptRecipe.canonicalLayouts_column_window' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TranscriptRecipe.canonicalLayouts_column_window

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TranscriptRecipe.canonicalLayouts_windows_disjoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TranscriptRecipe.canonicalLayouts_windows_disjoint

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TranscriptRecipe.canonicalLayouts_no_shared_column' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TranscriptRecipe.canonicalLayouts_no_shared_column

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TranscriptRecipe.rawRow_target_in_window' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TranscriptRecipe.rawRow_target_in_window

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TranscriptRecipe.row_target_in_window' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TranscriptRecipe.row_target_in_window

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TranscriptRecipe.transcriptRows_owner_unique' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TranscriptRecipe.transcriptRows_owner_unique

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TranscriptRecipe.transcriptColumns_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TranscriptRecipe.transcriptColumns_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TranscriptRecipe.transcriptColumns_length_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TranscriptRecipe.transcriptColumns_length_eq

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TranscriptRecipe.transcriptColumns_nonzero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TranscriptRecipe.transcriptColumns_nonzero

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TranscriptRecipe.transcriptColumns_in_window' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TranscriptRecipe.transcriptColumns_in_window


/-- info: 'Nightstream.Implementation.R1CS.Canonical.TranscriptRecipe.transcriptColumns_eq_canonical_sbox' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TranscriptRecipe.transcriptColumns_eq_canonical_sbox

/-- info: 'Nightstream.Implementation.R1CS.Canonical.TranscriptRecipe.transcriptColumns_not_layout_generic' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TranscriptRecipe.transcriptColumns_not_layout_generic


/-- info: 'Nightstream.Implementation.R1CS.Canonical.TranscriptRecipe.transcriptColumns_written' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TranscriptRecipe.transcriptColumns_written

end NightstreamTests.Axioms.CanonicalTranscriptRecipe
