import Nightstream.Implementation.NebulaV2.ShiftedTernaryEncodingBridge

set_option autoImplicit false

namespace tests.NebulaV2ShiftedTernaryEncodingBridge

open Nightstream.Implementation.NebulaV2.ShiftedTernaryEncodingBridge
open Nightstream.Protocol.NebulaV2.ShiftedTernary41V1

def maximum : CanonicalGoldilocks :=
  ⟨modulus - 1, by norm_num [modulus]⟩

example (index : Fin digitCount) :
    (trits maximum).getD index.val 0 =
      target maximum / 3 ^ index.val % 3 :=
  trits_getD_eq_quotient maximum index.val index.isLt

example (index : Fin digitCount) :
    Nightstream.Implementation.R1CS.ShiftedTernaryCanonicalWord.canonicalDigit
        maximum.val index.val =
      fieldDigit
        (Nightstream.Protocol.NebulaV2.CompactCommit.tritAt maximum index) :=
  canonicalDigit_eq_fieldDigit_tritAt maximum index

end tests.NebulaV2ShiftedTernaryEncodingBridge
