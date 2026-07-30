import Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeSound
import tests.Axioms.Support

/-!
Fail-closed axiom guards for semantic soundness of the Lean-owned
canonical-u64 recipe.
-/

namespace NightstreamTests.Axioms.CanonicalCanonicalU64RecipeSound

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeSound.bitValue_le_one' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalU64RecipeSound.bitValue_le_one

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeSound.bitsValue_lt_modulus' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalU64RecipeSound.bitsValue_lt_modulus

/-- info: 'Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeSound.sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalU64RecipeSound.sound

end NightstreamTests.Axioms.CanonicalCanonicalU64RecipeSound
