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

universe u v

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

/-! ## Codecs from explicit injective encodings -/

/-- Recover an admissible value from an explicit fixed-width encoding.

This constructor is for proof-owned semantic layouts whose field-coordinate
order is explicit but whose decoder is not used as an executable protocol
algorithm.  The chosen value is unique because `encodeInjective` applies
inside the stated admissible domain.  The construction uses classical choice
only to implement that inverse; the emitted coordinate order remains exactly
the caller-supplied `encode`.
-/
noncomputable def ofInjectiveEncoding
    {α : Type u}
    (width : Nat)
    (Admissible : α → Prop)
    (encode : α → List Field)
    (encodeLength : ∀ value, (encode value).length = width)
    (encodeInjective :
      ∀ {left right},
        Admissible left →
        Admissible right →
        encode left = encode right →
        left = right) :
    Codec α where
  width := width
  Admissible := Admissible
  encode := encode
  decode := fun coordinates =>
    letI :
        Decidable
          (∃ value, Admissible value ∧ encode value = coordinates) :=
      Classical.propDecidable _
    if existsValue :
        ∃ value, Admissible value ∧ encode value = coordinates then
      some (Classical.choose existsValue)
    else
      none
  encode_length := encodeLength
  decode_encode := by
    intro value admissible
    have existsValue :
        ∃ candidate, Admissible candidate ∧
          encode candidate = encode value :=
      ⟨value, admissible, rfl⟩
    rw [dif_pos existsValue]
    apply congrArg some
    exact encodeInjective
      (Classical.choose_spec existsValue).1 admissible
      (Classical.choose_spec existsValue).2
  encode_decode := by
    intro coordinates value decoded
    by_cases existsValue :
        ∃ candidate, Admissible candidate ∧
          encode candidate = coordinates
    · rw [dif_pos existsValue] at decoded
      have chosenEq :
          Classical.choose existsValue = value :=
        Option.some.inj decoded
      subst value
      exact Classical.choose_spec existsValue
    · rw [dif_neg existsValue] at decoded
      contradiction

/-- Canonical encoding is injective inside a codec's admissible domain. -/
theorem encode_injective_of_admissible
    {α : Type u}
    (codec : Codec α)
    {left right : α}
    (leftAdmissible : codec.Admissible left)
    (rightAdmissible : codec.Admissible right)
    (encodedEqual : codec.encode left = codec.encode right) :
    left = right := by
  have leftDecoded := codec.decode_encode left leftAdmissible
  have rightDecoded := codec.decode_encode right rightAdmissible
  rw [encodedEqual, rightDecoded] at leftDecoded
  exact Option.some.inj leftDecoded.symm

/-- Product encoding in left-to-right coordinate order. -/
noncomputable def product
    {α : Type u}
    {β : Type v}
    (left : Codec α)
    (right : Codec β) :
    Codec (α × β) :=
  ofInjectiveEncoding
    (left.width + right.width)
    (fun value =>
      left.Admissible value.1 ∧ right.Admissible value.2)
    (fun value => left.encode value.1 ++ right.encode value.2)
    (by
      intro value
      simp [left.encode_length, right.encode_length])
    (by
      intro first second firstAdmissible secondAdmissible encodedEqual
      have leftEncoded :
          left.encode first.1 = left.encode second.1 := by
        have selected :=
          congrArg (List.take left.width) encodedEqual
        simpa [left.encode_length] using selected
      have rightEncoded :
          right.encode first.2 = right.encode second.2 := by
        have selected :=
          congrArg (List.drop left.width) encodedEqual
        simpa [left.encode_length] using selected
      exact Prod.ext
        (left.encode_injective_of_admissible
          firstAdmissible.1 secondAdmissible.1 leftEncoded)
        (right.encode_injective_of_admissible
          firstAdmissible.2 secondAdmissible.2 rightEncoded))

/-- Pull a codec back through one injective semantic projection.

The target codec determines the complete coordinate order.  `toInjective`
proves that those coordinates retain every part of the source value.
-/
noncomputable def pullback
    {α : Type u}
    {β : Type v}
    (target : Codec β)
    (toTarget : α → β)
    (toInjective : Function.Injective toTarget) :
    Codec α :=
  ofInjectiveEncoding
    target.width
    (fun value => target.Admissible (toTarget value))
    (fun value => target.encode (toTarget value))
    (fun value => target.encode_length (toTarget value))
    (by
      intro left right leftAdmissible rightAdmissible encodedEqual
      apply toInjective
      exact target.encode_injective_of_admissible
        leftAdmissible rightAdmissible encodedEqual)

/-- Pull a target codec back through an injective projection on an explicit
source domain.  This form is used when the source has fixed setup fields that
are omitted from the coordinate string and checked by `Admissible`. -/
noncomputable def pullbackOn
    {α : Type u}
    {β : Type v}
    (target : Codec β)
    (sourceAdmissible : α → Prop)
    (toTarget : α → β)
    (targetAdmissible :
      ∀ value, sourceAdmissible value →
        target.Admissible (toTarget value))
    (toInjective :
      ∀ {left right},
        sourceAdmissible left →
        sourceAdmissible right →
        toTarget left = toTarget right →
        left = right) :
    Codec α :=
  ofInjectiveEncoding
    target.width
    sourceAdmissible
    (fun value => target.encode (toTarget value))
    (fun value => target.encode_length (toTarget value))
    (by
      intro left right leftAdmissible rightAdmissible encodedEqual
      apply toInjective leftAdmissible rightAdmissible
      exact target.encode_injective_of_admissible
        (targetAdmissible left leftAdmissible)
        (targetAdmissible right rightAdmissible)
        encodedEqual)

/-- Field coordinates for a fixed finite function, in increasing `Fin`
index order. -/
def encodeFin
    {α : Type u}
    (codec : Codec α) :
    (count : Nat) → (Fin count → α) → List Field
  | 0, _ => []
  | count + 1, values =>
      codec.encode (values 0) ++
        encodeFin codec count (fun index => values index.succ)

@[simp] theorem encodeFin_length
    {α : Type u}
    (codec : Codec α)
    (count : Nat)
    (values : Fin count → α) :
    (encodeFin codec count values).length = count * codec.width := by
  induction count with
  | zero => simp [encodeFin]
  | succ count inductionHypothesis =>
      simp only [encodeFin, List.length_append, codec.encode_length,
        inductionHypothesis, Nat.succ_mul]
      omega

private theorem getD_append_left
    {α : Type u}
    (left right : List α)
    (index : Nat)
    (default : α)
    (indexLt : index < left.length) :
    (left ++ right).getD index default = left.getD index default := by
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem (by
      rw [List.length_append]
      omega),
    List.getElem?_eq_getElem indexLt,
    List.getElem_append_left]

private theorem getD_append_right
    {α : Type u}
    (left right : List α)
    (index : Nat)
    (default : α) :
    (left ++ right).getD (left.length + index) default =
      right.getD index default := by
  rw [List.getD_eq_getElem?_getD,
    List.getElem?_append_right (by omega)]
  simp only [Nat.add_sub_cancel_left]
  rfl

/-- Select one coordinate from one element block of a finite-function
encoding.  This theorem is the index law used by semantic codec views. -/
theorem encodeFin_getD
    {α : Type u}
    (codec : Codec α)
    (count : Nat)
    (values : Fin count → α)
    (element : Fin count)
    (coordinate : Fin codec.width)
    (default : Field) :
    (encodeFin codec count values).getD
        (element.val * codec.width + coordinate.val) default =
      (codec.encode (values element)).getD coordinate.val default := by
  induction count with
  | zero =>
      exact Fin.elim0 element
  | succ count inductionHypothesis =>
      refine Fin.cases ?_ (fun tail => ?_) element
      · simp only [encodeFin, Fin.val_zero, Nat.zero_mul, Nat.zero_add]
        exact getD_append_left _ _ _ _
          (by
            rw [codec.encode_length]
            exact coordinate.isLt)
      · simp only [encodeFin, Fin.val_succ]
        rw [Nat.add_mul, Nat.one_mul]
        have indexShape :
            tail.val * codec.width + codec.width + coordinate.val =
              (codec.encode (values 0)).length +
                (tail.val * codec.width + coordinate.val) := by
          rw [codec.encode_length]
          omega
        rw [indexShape, getD_append_right]
        exact inductionHypothesis
          (fun index => values index.succ) tail

private theorem encodeFin_injective
    {α : Type u}
    (codec : Codec α)
    (count : Nat)
    {left right : Fin count → α}
    (leftAdmissible : ∀ index, codec.Admissible (left index))
    (rightAdmissible : ∀ index, codec.Admissible (right index))
    (encodedEqual :
      encodeFin codec count left = encodeFin codec count right) :
    left = right := by
  induction count with
  | zero =>
      funext index
      exact Fin.elim0 index
  | succ count inductionHypothesis =>
      have headEncoded :
          codec.encode (left 0) = codec.encode (right 0) := by
        have selected :=
          congrArg (List.take codec.width) encodedEqual
        simpa [encodeFin, codec.encode_length] using selected
      have headEqual : left 0 = right 0 :=
        codec.encode_injective_of_admissible
          (leftAdmissible 0) (rightAdmissible 0) headEncoded
      have tailEncoded :
          encodeFin codec count
              (fun index : Fin count => left index.succ) =
            encodeFin codec count
              (fun index : Fin count => right index.succ) := by
        have selected :=
          congrArg (List.drop codec.width) encodedEqual
        simpa [encodeFin, codec.encode_length] using selected
      have tailEqual :
          (fun index : Fin count => left index.succ) =
            (fun index : Fin count => right index.succ) :=
        inductionHypothesis
          (fun index : Fin count => leftAdmissible index.succ)
          (fun index : Fin count => rightAdmissible index.succ)
          tailEncoded
      funext index
      exact Fin.cases headEqual
        (fun tail => congrFun tailEqual tail) index

/-- Fixed finite functions use the index-major concatenation of their element
codecs. -/
noncomputable def finFunction
    {α : Type u}
    (count : Nat)
    (codec : Codec α) :
    Codec (Fin count → α) :=
  ofInjectiveEncoding
    (count * codec.width)
    (fun values => ∀ index, codec.Admissible (values index))
    (encodeFin codec count)
    (encodeFin_length codec count)
    (encodeFin_injective codec count)

/-- A list with an exact semantic length, serialized in list order.  The
default is used only by the total encoding outside the admissible domain. -/
noncomputable def fixedList
    {α : Type u}
    (count : Nat)
    (default : α)
    (codec : Codec α) :
    Codec (List α) :=
  ofInjectiveEncoding
    (count * codec.width)
    (fun values =>
      values.length = count ∧
        ∀ index : Fin count,
          codec.Admissible (values.getD index.val default))
    (fun values =>
      encodeFin codec count
        (fun index => values.getD index.val default))
    (fun values => encodeFin_length codec count _)
    (by
      intro left right leftAdmissible rightAdmissible encodedEqual
      have valuesEqual :
          (fun index : Fin count =>
              left.getD index.val default) =
            (fun index : Fin count =>
              right.getD index.val default) :=
        encodeFin_injective codec count
          leftAdmissible.2 rightAdmissible.2 encodedEqual
      apply List.ext_get
      · exact leftAdmissible.1.trans rightAdmissible.1.symm
      · intro index leftLt rightLt
        let typed : Fin count :=
          ⟨index, by
            rw [← leftAdmissible.1]
            exact leftLt⟩
        have selected := congrFun valuesEqual typed
        change left.getD index default = right.getD index default at selected
        rw [List.getD_eq_getElem?_getD,
          List.getElem?_eq_getElem leftLt] at selected
        rw [List.getD_eq_getElem?_getD,
          List.getElem?_eq_getElem rightLt] at selected
        exact selected)

/-- An array with an exact semantic size, serialized in array order.  The
default is used only by the total encoding outside the admissible domain. -/
noncomputable def fixedArray
    {α : Type u}
    (count : Nat)
    (default : α)
    (codec : Codec α) :
    Codec (Array α) :=
  ofInjectiveEncoding
    (count * codec.width)
    (fun values =>
      values.size = count ∧
        ∀ index : Fin count,
          codec.Admissible (values.getD index.val default))
    (fun values =>
      encodeFin codec count
        (fun index => values.getD index.val default))
    (fun values => encodeFin_length codec count _)
    (by
      intro left right leftAdmissible rightAdmissible encodedEqual
      have valuesEqual :
          (fun index : Fin count =>
              left.getD index.val default) =
            (fun index : Fin count =>
              right.getD index.val default) :=
        encodeFin_injective codec count
          leftAdmissible.2 rightAdmissible.2 encodedEqual
      apply Array.ext
      · exact leftAdmissible.1.trans rightAdmissible.1.symm
      · intro index leftLt rightLt
        let typed : Fin count :=
          ⟨index, by
            rw [← leftAdmissible.1]
            exact leftLt⟩
        have selected := congrFun valuesEqual typed
        change left.getD index default = right.getD index default at selected
        rw [Array.getD_eq_getD_getElem?,
          Array.getElem?_eq_getElem leftLt] at selected
        rw [Array.getD_eq_getD_getElem?,
          Array.getElem?_eq_getElem rightLt] at selected
        exact selected)

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
  [⟨value % Nightstream.SuperNeo.Concrete.goldilocksModulus,
    Nat.mod_lt _ (by decide)⟩]

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
    simp [encodeNat]
  decode_encode := by
    intro value admissible
    simp [encodeNat, decodeNat, Nat.mod_eq_of_lt admissible]
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
            · simp [encodeNat, Nat.mod_eq_of_lt head.isLt]
        | cons next rest =>
            simp [decodeNat] at decoded

/-- The total natural-coordinate encoding commutes with successor in the
Goldilocks field.  Its admissible semantic domain remains the strict interval
below the modulus. -/
theorem boundedNatCodec_encode_succ (value : Nat) :
    (boundedNatCodec.encode (value + 1)).getD 0 0 =
      (boundedNatCodec.encode value).getD 0 0 + 1 := by
  apply Fin.ext
  change
    (value + 1) % Nightstream.SuperNeo.Concrete.goldilocksModulus =
      (value % Nightstream.SuperNeo.Concrete.goldilocksModulus + 1) %
        Nightstream.SuperNeo.Concrete.goldilocksModulus
  rw [Nat.add_mod]
  simp [Nightstream.SuperNeo.Concrete.goldilocksModulus]

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
