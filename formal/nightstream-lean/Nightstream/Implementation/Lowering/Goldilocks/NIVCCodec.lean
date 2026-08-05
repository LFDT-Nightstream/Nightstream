import Nightstream.HyperNova.NIVCCompatibility
import Nightstream.Implementation.Lowering.Goldilocks.Codec

/-!
Contract: adapt an admissible fixed-width Goldilocks codec to the canonical
codec required by corrected HyperNova Definition 12.

Owns: the exact admissible subtype, its fixed-width encoding, and the proof
that this encoding has both inverse directions and is prefix-free.

Does not own: a complete NIVC compiler, transcript hashing, verifier-key
projection, Rust, R1CS rows, or constraint counts.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.Goldilocks.NIVCCodec

universe u

/-- Values for which the source codec proves its round-trip contract. -/
abbrev AdmissibleValue
    {Value : Type u}
    (codec : Nightstream.Implementation.Lowering.Goldilocks.Codec Value) :=
  { value : Value // codec.Admissible value }

private theorem encoding_injective
    {Value : Type u}
    (codec : Nightstream.Implementation.Lowering.Goldilocks.Codec Value) :
    Function.Injective
      (fun value : AdmissibleValue codec => codec.encode value.1) := by
  intro left right sameEncoding
  apply Subtype.ext
  exact codec.encode_injective_of_admissible
    left.property right.property sameEncoding

/-- The HyperNova-facing codec uses the source codec's exact coordinate order
and a canonical partial inverse. -/
noncomputable def toNivcCodec
    {Value : Type u}
    (codec : Nightstream.Implementation.Lowering.Goldilocks.Codec Value) :
    Nightstream.HyperNova.NIVCCompatibility.Codec
      (AdmissibleValue codec) Field :=
  Nightstream.HyperNova.NIVCCompatibility.Codec.withClassicalDecoder
    (fun value => codec.encode value.1)

/-- Fixed source width and admissible-domain injectivity discharge the
HyperNova canonical and prefix-free codec contract. -/
theorem toNivcCodec_canonical
    {Value : Type u}
    (codec : Nightstream.Implementation.Lowering.Goldilocks.Codec Value) :
    (toNivcCodec codec).Canonical := by
  exact
    Nightstream.HyperNova.NIVCCompatibility.Codec.fixedWidthInjective_canonical
      codec.width (fun value : AdmissibleValue codec => codec.encode value.1)
      (fun value => codec.encode_length value.1)
      (encoding_injective codec)

/-- Adapt a source codec whose complete value type is admissible. This form
keeps the original value type and coordinate order. -/
noncomputable def toTotalNivcCodec
    {Value : Type u}
    (codec : Nightstream.Implementation.Lowering.Goldilocks.Codec Value) :
    Nightstream.HyperNova.NIVCCompatibility.Codec Value Field :=
  Nightstream.HyperNova.NIVCCompatibility.Codec.withClassicalDecoder
    codec.encode

/-- A fixed-width source codec that accepts every value is a canonical
Definition 12 codec on the original value type. -/
theorem toTotalNivcCodec_canonical
    {Value : Type u}
    (codec : Nightstream.Implementation.Lowering.Goldilocks.Codec Value)
    (allAdmissible : forall value, codec.Admissible value) :
    (toTotalNivcCodec codec).Canonical := by
  exact
    Nightstream.HyperNova.NIVCCompatibility.Codec.fixedWidthInjective_canonical
      codec.width codec.encode codec.encode_length
      (fun left right equal =>
        codec.encode_injective_of_admissible
          (allAdmissible left) (allAdmissible right) equal)

end Nightstream.Implementation.Lowering.Goldilocks.NIVCCodec
