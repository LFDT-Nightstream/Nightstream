import Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeHonest
import tests.Axioms.Support

/-!
Fail-closed axiom guards for honest completeness of the Lean-owned
canonical-u64 recipe.
-/

namespace NightstreamTests.Axioms.CanonicalCanonicalU64RecipeHonest

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeHonest.witness_input' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalU64RecipeHonest.witness_input

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeHonest.witness_canonical' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalU64RecipeHonest.witness_canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeHonest.complete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalU64RecipeHonest.complete

end NightstreamTests.Axioms.CanonicalCanonicalU64RecipeHonest
