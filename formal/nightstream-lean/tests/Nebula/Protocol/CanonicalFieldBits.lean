import Nightstream.Protocol.Nebula.CanonicalFieldBits

set_option autoImplicit false

namespace Nightstream.Tests.NebulaCanonicalFieldBits

open Nightstream.Protocol.Nebula.CanonicalFieldBits

example : Function.Injective decode := decode_injective

example : ¬ Canonical modulusWord := modulusWord_not_canonical

example : zeroWord ≠ modulusWord := zeroWord_ne_modulusWord

example : decode zeroWord %
      Nightstream.Protocol.Nebula.ShiftedTernary41V1.modulus =
    decode modulusWord %
      Nightstream.Protocol.Nebula.ShiftedTernary41V1.modulus :=
  zero_and_modulus_are_modulo_aliases

end Nightstream.Tests.NebulaCanonicalFieldBits
