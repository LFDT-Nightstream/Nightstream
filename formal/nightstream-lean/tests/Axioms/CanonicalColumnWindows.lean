import Nightstream.Implementation.R1CS.Canonical.ColumnWindows
import tests.Axioms.Support

/-!
Fail-closed axiom guard for ColumnWindows.

Every report below is measured, not asserted: the expected text was produced by
running the audit and copying its output, so any drift fails the build.
-/

namespace NightstreamTests.Axioms.CanonicalColumnWindows

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.ColumnWindows.windowsOf_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms ColumnWindows.windowsOf_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.ColumnWindows.mem_windowsOf_base_ge' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms ColumnWindows.mem_windowsOf_base_ge

/-- info: 'Nightstream.Implementation.R1CS.Canonical.ColumnWindows.windowsOf_no_collision' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ColumnWindows.windowsOf_no_collision

/-- info: 'Nightstream.Implementation.R1CS.Canonical.ColumnWindows.windowsOf_span' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms ColumnWindows.windowsOf_span

/-- info: 'Nightstream.Implementation.R1CS.Canonical.ColumnWindows.constantWire_unowned' does not depend on any axioms -/
#guard_msgs in
#audit_axioms ColumnWindows.constantWire_unowned

/-- info: 'Nightstream.Implementation.R1CS.Canonical.ColumnWindows.relocate_pos' does not depend on any axioms -/
#guard_msgs in
#audit_axioms ColumnWindows.relocate_pos

/-- info: 'Nightstream.Implementation.R1CS.Canonical.ColumnWindows.relocate_owns' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms ColumnWindows.relocate_owns

/-- info: 'Nightstream.Implementation.R1CS.Canonical.ColumnWindows.mentions_renameTerms_relocate' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ColumnWindows.mentions_renameTerms_relocate

/-- info: 'Nightstream.Implementation.R1CS.Canonical.ColumnWindows.placeAll_columns' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms ColumnWindows.placeAll_columns

/-- info: 'Nightstream.Implementation.R1CS.Canonical.ColumnWindows.placeAll_targets' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms ColumnWindows.placeAll_targets

/-- info: 'Nightstream.Implementation.R1CS.Canonical.ColumnWindows.allocationPlaced_nil' does not depend on any axioms -/
#guard_msgs in
#audit_axioms ColumnWindows.allocationPlaced_nil

/-- info: 'Nightstream.Implementation.R1CS.Canonical.ColumnWindows.allocationPlaced_of_bounded' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ColumnWindows.allocationPlaced_of_bounded

/-- info: 'Nightstream.Implementation.R1CS.Canonical.ColumnWindows.spanOf_nil' does not depend on any axioms -/
#guard_msgs in
#audit_axioms ColumnWindows.spanOf_nil

/-- info: 'Nightstream.Implementation.R1CS.Canonical.ColumnWindows.le_spanOf' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms ColumnWindows.le_spanOf

/-- info: 'Nightstream.Implementation.R1CS.Canonical.ColumnWindows.allocationPlaced_spanOf' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ColumnWindows.allocationPlaced_spanOf

/-- info: 'Nightstream.Implementation.R1CS.Canonical.ColumnWindows.placeAll_satisfies_head' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ColumnWindows.placeAll_satisfies_head

/-- info: 'Nightstream.Implementation.R1CS.Canonical.ColumnWindows.placeAll_satisfies_tail' does not depend on any axioms -/
#guard_msgs in
#audit_axioms ColumnWindows.placeAll_satisfies_tail

/-- info: 'Nightstream.Implementation.R1CS.Canonical.ColumnWindows.placeAll_satisfies_index' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ColumnWindows.placeAll_satisfies_index

/-- info: 'Nightstream.Implementation.R1CS.Canonical.ColumnWindows.placeAll_honest' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms ColumnWindows.placeAll_honest


/-- info: 'Nightstream.Implementation.R1CS.Canonical.ColumnWindows.placed_allocations_disjoint' does not depend on any axioms -/
#guard_msgs in
#audit_axioms ColumnWindows.placed_allocations_disjoint

end NightstreamTests.Axioms.CanonicalColumnWindows
