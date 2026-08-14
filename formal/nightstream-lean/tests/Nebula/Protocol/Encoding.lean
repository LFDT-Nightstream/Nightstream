import Nightstream.Protocol.Nebula.Encoding
import Nightstream.Protocol.Nebula.Profile

set_option autoImplicit false

namespace tests.NebulaEncoding

open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.ShiftedTernary41V1

def zero : CanonicalGoldilocks := ⟨0, by norm_num [modulus]⟩

def maximum : CanonicalGoldilocks :=
  ⟨modulus - 1, by norm_num [modulus]⟩

example : decode (trits zero) = 0 := decode_encode zero

example : decode (trits maximum) = modulus - 1 := decode_encode maximum

example : trits zero ≠ trits maximum := by
  intro equal
  have equalValues := congrArg Subtype.val (trits_injective equal)
  change 0 = modulus - 1 at equalValues
  exact (by norm_num [modulus] : 0 ≠ modulus - 1) equalValues

/- An E=8 relation that reuses the V2 name and version is still a different
record, but SPEC.md forbids issuing it as V2. No production constructor for
this value exists. -/
def forbiddenSameVersionFactorEight : Profile.Identity :=
  { Profile.v2 with checkedStepsPerFreshClaim := 8 }

example : forbiddenSameVersionFactorEight ≠ Profile.v2 := by
  decide

end tests.NebulaEncoding
