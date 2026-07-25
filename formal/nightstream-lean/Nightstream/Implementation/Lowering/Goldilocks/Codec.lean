import Nightstream.Implementation.Lowering.Typed.Signature
import Nightstream.SuperNeo.Concrete.Algebra

/-!
Contract: artifact-independent fixed-width codecs from typed semantic values to
canonical Goldilocks coordinates.

Owns:
- the partial-codec laws and their exact-width consequences;
- canonical one-coordinate codecs for Goldilocks values and Booleans;
- the explicitly bounded one-coordinate codec for natural numbers;
- typed codec families and schema-local port-width agreement.

Does not own: physical columns, row emission, Rust layouts, generated
artifacts, or a claim that all natural numbers fit in a fixed-width encoding.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Lowering.Goldilocks

open Nightstream.Implementation.Lowering

universe u

/-- The canonical field used by the concrete SuperNeo instantiation. -/
abbrev Field := Nightstream.SuperNeo.Concrete.F

/-- A fixed-width partial semantic codec.

`encode` is total so a lowering pass never has to invent coordinates, but its
round-trip guarantee applies only to `Admissible` values.  Conversely, every
successful decode is admissible and is represented by exactly the coordinates
that were decoded. -/
structure Codec (α : Type u) where
  width : Nat
  Admissible : α -> Prop
  encode : α -> List Field
  decode : List Field -> Option α
  encode_length : ∀ value, (encode value).length = width
  decode_encode :
    ∀ value, Admissible value -> decode (encode value) = some value
  encode_decode :
    ∀ coordinates value, decode coordinates = some value ->
      Admissible value ∧ encode value = coordinates

namespace Codec

/-- A successfully decoded value is inside the codec's explicit domain. -/
theorem admissible_of_decode
    {α : Type u}
    (codec : Codec α)
    {coordinates : List Field}
    {value : α}
    (decoded : codec.decode coordinates = some value) :
    codec.Admissible value :=
  (codec.encode_decode coordinates value decoded).1

/-- A successful decode can only consume the codec's exact coordinate width. -/
theorem length_eq_width_of_decode
    {α : Type u}
    (codec : Codec α)
    {coordinates : List Field}
    {value : α}
    (decoded : codec.decode coordinates = some value) :
    coordinates.length = codec.width := by
  have encoded := (codec.encode_decode coordinates value decoded).2
  rw [← encoded]
  exact codec.encode_length value

/-- Successful decoding is injective at a fixed coordinate string. -/
theorem decoded_value_unique
    {α : Type u}
    (codec : Codec α)
    {coordinates : List Field}
    {left right : α}
    (leftDecoded : codec.decode coordinates = some left)
    (rightDecoded : codec.decode coordinates = some right) :
    left = right := by
  rw [leftDecoded] at rightDecoded
  exact Option.some.inj rightDecoded

end Codec

/-! ## Canonical one-coordinate codecs -/

private def decodeField : List Field -> Option Field
  | [value] => some value
  | _ => none

/-- A field element is already one canonical Goldilocks coordinate. -/
def fieldCodec : Codec Field where
  width := 1
  Admissible := fun _ => True
  encode := fun value => [value]
  decode := decodeField
  encode_length := by
    intro value
    rfl
  decode_encode := by
    intro value admissible
    rfl
  encode_decode := by
    intro coordinates value decoded
    cases coordinates with
    | nil =>
        simp [decodeField] at decoded
    | cons head tail =>
        cases tail with
        | nil =>
            simp only [decodeField] at decoded
            have equal : head = value := Option.some.inj decoded
            subst value
            exact ⟨True.intro, rfl⟩
        | cons next rest =>
            simp [decodeField] at decoded

private def encodeBool : Bool -> List Field
  | false => [0]
  | true => [1]

private def decodeBool : List Field -> Option Bool
  | [value] =>
      if value = 0 then
        some false
      else if value = 1 then
        some true
      else
        none
  | _ => none

private theorem field_zero_ne_one : (0 : Field) ≠ 1 := by
  decide

private theorem field_one_ne_zero : (1 : Field) ≠ 0 := by
  decide

/-- Booleans use the canonical field coordinates zero and one. -/
def boolCodec : Codec Bool where
  width := 1
  Admissible := fun _ => True
  encode := encodeBool
  decode := decodeBool
  encode_length := by
    intro value
    cases value <;> rfl
  decode_encode := by
    intro value admissible
    cases value with
    | false => rfl
    | true =>
        simp only [encodeBool, decodeBool, if_neg field_one_ne_zero,
          if_pos True.intro]
  encode_decode := by
    intro coordinates value decoded
    cases coordinates with
    | nil =>
        simp [decodeBool] at decoded
    | cons head tail =>
        cases tail with
        | nil =>
            by_cases isZero : head = 0
            · simp only [decodeBool, if_pos isZero] at decoded
              have valueEq : false = value := Option.some.inj decoded
              subst value
              exact ⟨True.intro, by simp [encodeBool, isZero]⟩
            · by_cases isOne : head = 1
              · simp only [decodeBool, if_neg isZero, if_pos isOne] at decoded
                have valueEq : true = value := Option.some.inj decoded
                subst value
                exact ⟨True.intro, by simp [encodeBool, isOne]⟩
              · simp [decodeBool, isZero, isOne] at decoded
        | cons next rest =>
            simp [decodeBool] at decoded

private def encodeNat (value : Nat) : List Field :=
  if admissible : value < Nightstream.SuperNeo.Concrete.goldilocksModulus then
    [⟨value, admissible⟩]
  else
    [0]

private def decodeNat : List Field -> Option Nat
  | [value] => some value.val
  | _ => none

/-- One-coordinate natural-number encoding with an explicit finite domain.

This is intentionally not a total fixed-width injection of `Nat`: values at or
above the Goldilocks modulus are outside `Admissible` and receive no round-trip
guarantee. -/
def boundedNatCodec : Codec Nat where
  width := 1
  Admissible :=
    fun value => value < Nightstream.SuperNeo.Concrete.goldilocksModulus
  encode := encodeNat
  decode := decodeNat
  encode_length := by
    intro value
    by_cases admissible :
        value < Nightstream.SuperNeo.Concrete.goldilocksModulus
    · simp [encodeNat, admissible]
    · simp [encodeNat, admissible]
  decode_encode := by
    intro value admissible
    simp [encodeNat, admissible, decodeNat]
  encode_decode := by
    intro coordinates value decoded
    cases coordinates with
    | nil =>
        simp [decodeNat] at decoded
    | cons head tail =>
        cases tail with
        | nil =>
            simp only [decodeNat] at decoded
            have equal : head.val = value := Option.some.inj decoded
            subst value
            constructor
            · exact head.isLt
            · simp only [encodeNat, dif_pos head.isLt]
        | cons next rest =>
            simp [decodeNat] at decoded

/-- Decoding one canonical natural-number coordinate exposes its exact
integer representative.  This theorem is the public one-coordinate boundary;
call recipes never unfold the private decoder. -/
theorem boundedNatCodec_decode_singleton_iff
    (coordinate : Field)
    (value : Nat) :
    boundedNatCodec.decode [coordinate] = some value ↔
      coordinate.val = value := by
  constructor
  · intro decoded
    exact Option.some.inj decoded
  · intro equal
    simpa [boundedNatCodec, decodeNat] using congrArg some equal

/-! ## Typed codec profiles -/

/-- One semantic codec for every kind in an artifact-independent type system.
Data codecs are selected by semantic tag, never by Rust column metadata. -/
structure Family (types : Typed.TypeSystem.{u}) where
  field : Codec types.Field
  bit : Codec types.Bit
  data : (tag : types.Data) -> Codec (types.dataValue tag)

namespace Family

/-- Select the semantic codec for one typed IR kind. -/
def codecFor
    {types : Typed.TypeSystem.{u}}
    (family : Family types) :
    (kind : types.Kind) -> Codec (types.Value kind)
  | .field => family.field
  | .bit => family.bit
  | .data tag => family.data tag

end Family

/-- A logical port allocates exactly the coordinates required by its semantic
codec.  Ownership classes may differ between ports; only width is constrained
here. -/
def PortWidthAgrees
    {types : Typed.TypeSystem.{u}}
    (family : Family types)
    (port : Typed.Port types) : Prop :=
  (family.codecFor port.kind).width = port.layout.owners.length

/-- Width agreement for every port in one exact typed schema. -/
def SchemaWidthAgrees
    {types : Typed.TypeSystem.{u}}
    (family : Family types)
    (schema : Typed.Schema types) : Prop :=
  ∀ port, port ∈ schema -> PortWidthAgrees family port

/-- A schema-local concrete encoding profile.

The profile does not prescribe ownership or physical addresses.  It only
connects each typed port's already-declared logical layout to the width of the
independently selected semantic codec. -/
structure Profile
    (types : Typed.TypeSystem.{u})
    (schema : Typed.Schema types) where
  family : Family types
  widthsAgree : SchemaWidthAgrees family schema

namespace Profile

/-- Public elimination rule for port-layout width agreement. -/
theorem codec_width_eq_layout_width
    {types : Typed.TypeSystem.{u}}
    {schema : Typed.Schema types}
    (profile : Profile types schema)
    (port : Typed.Port types)
    (member : port ∈ schema) :
    (profile.family.codecFor port.kind).width = port.layout.owners.length :=
  profile.widthsAgree port member

end Profile

end Nightstream.Implementation.Lowering.Goldilocks
