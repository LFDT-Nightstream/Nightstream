import Nightstream.Implementation.R1CS.Canonical.Link270Production
import tests.Axioms.Support

/-!
Fail-closed dependency gate for the canonical 270-coordinate link.

Owns: dependency expectations for the Phase-1 canonical encoding and the Phase-1b
comparison surface.

No theorem here may acquire `Lean.trustCompiler`: the canonical encoding must
stay kernel-derived, since its whole purpose is to produce a count that does
not come from a generated artifact.
-/

namespace NightstreamTests.Axioms.CanonicalLink270

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-! Phase 1 — canonical encoding. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Link270.canonicalRows_holds_iff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Link270.canonicalRows_holds_iff

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Link270.canonicalRows_length_eq' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Link270.canonicalRows_length_eq

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Link270.affine_zero_iff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Link270.affine_zero_iff

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Link270.dropCoordinate_admits_violation' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Link270.dropCoordinate_admits_violation

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Link270.nonzeroTail_linked' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Link270.nonzeroTail_linked

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Link270.canonicalRows_owned' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Link270.canonicalRows_owned

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Link270.coordinateRow_injective' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Link270.coordinateRow_injective

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Link270.columns_disjoint' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Link270.columns_disjoint

/-! Phase 1b — comparison surface. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Link270Production.copies_not_pinsZero' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Link270Production.copies_not_pinsZero

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Link270Production.tailPinsZero_not_agrees' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Link270Production.tailPinsZero_not_agrees

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Link270Production.tail_count' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Link270Production.tail_count

/-! Phase 1 — complete cost tuple and conservation. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Link270.link270Cost_eq' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Link270.link270Cost_eq

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Link270.cost_references_without_allocating' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Link270.cost_references_without_allocating

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Link270.referencedColumns_length_eq' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Link270.referencedColumns_length_eq

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Link270.allocation_exact' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Link270.allocation_exact

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Link270.rowColumns_accounted' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Link270.rowColumns_accounted

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Link270.constantWire_not_allocated' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Link270.constantWire_not_allocated

/-! Phase 1b — column alignment and exhaustive classification. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Link270Production.agreesAt_of_aligned' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Link270Production.agreesAt_of_aligned

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Link270Production.capture_eq_canonical' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Link270Production.capture_eq_canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Link270Production.classify_exhaustive' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Link270Production.classify_exhaustive

end NightstreamTests.Axioms.CanonicalLink270
