import Nightstream.Implementation.NebulaV2.Core.FieldCodec

/-! Focused regressions for the exact V2 public-field codec bridge. -/

namespace NightstreamTests.NebulaV2FieldCodec

open Nightstream.Implementation.NebulaV2.FieldCodec
open Nightstream.Protocol.NebulaV2

example :
    (∃ value, nativeDecode CanonicalFieldBits.zeroWord = some value ∧
        value.val = 0) ∧
      nativeDecode CanonicalFieldBits.modulusWord = none :=
  rejects_zero_modulus_alias

example (field :
    Nightstream.Implementation.R1CS.CanonicalU64Complete.FieldInverse) :
    let assignment :=
      Nightstream.Implementation.R1CS.CanonicalU64Complete.interpret field
        (sourceOfWord CanonicalFieldBits.zeroWord)
    Nightstream.Implementation.R1CS.Satisfies
      Nightstream.Implementation.R1CS.CanonicalU64.rows assignment := by
  exact (local_complete field CanonicalFieldBits.zeroWord
    (CanonicalFieldBits.encode_is_canonical CanonicalFieldBits.zero)).1

end NightstreamTests.NebulaV2FieldCodec
