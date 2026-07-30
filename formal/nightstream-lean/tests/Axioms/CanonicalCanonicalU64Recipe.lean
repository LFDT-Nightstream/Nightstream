import Nightstream.Implementation.R1CS.Canonical.CanonicalU64Recipe
import tests.Axioms.Support

/-!
Fail-closed axiom guards for the Lean-owned canonical-u64 row constructor.
-/

namespace NightstreamTests.Axioms.CanonicalCanonicalU64Recipe

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalU64Recipe.allocation_nodup' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalU64Recipe.allocation_nodup

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalU64Recipe.rows_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms CanonicalU64Recipe.rows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalU64Recipe.cost_rows' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms CanonicalU64Recipe.cost_rows

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalU64Recipe.rows_conservation' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalU64Recipe.rows_conservation

end NightstreamTests.Axioms.CanonicalCanonicalU64Recipe
