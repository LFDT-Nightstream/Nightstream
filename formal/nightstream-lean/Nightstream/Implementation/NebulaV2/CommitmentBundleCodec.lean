import Nightstream.Implementation.NebulaV2.WasmStateCodec
import Nightstream.Protocol.NebulaV2.CommitmentBundle
import Nightstream.Protocol.NebulaV2.Digest
import Nightstream.Protocol.NebulaV2.MemoryWireGeometry

/-!
Contract: canonical bit codec for the mandatory V2 commitment bundle.

Assurance tier: implementation model.

Owns the exact component order `full, operations, initial snapshot, final
snapshot`, all 972 canonical Goldilocks coordinates per component, the
248,832-bit flattened length, and encoding injectivity.

Does not own Ajtai evaluation, bundle binding, recursive public-column
placement, generated rows, Rust parsing, or terminal same-witness opening.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.CommitmentBundleCodec

open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.CommitmentBundle
open Nightstream.Protocol.NebulaV2.MemoryWireGeometry

abbrev Coordinate := Fin commitmentFieldCount
abbrev ComponentCommitment := Coordinate → ShiftedTernary41V1.CanonicalGoldilocks
abbrev Value := Bundle ComponentCommitment

structure FieldTag where
  component : Component
  coordinate : Coordinate
deriving DecidableEq, Repr

def componentOrder : List Component :=
  [.full, .operations, .initialSnapshot, .finalSnapshot]

def schema : List FieldTag :=
  (componentOrder.map fun component =>
    List.ofFn fun coordinate : Coordinate =>
      { component := component, coordinate := coordinate }).flatten

def FieldTag.bitWidth (_tag : FieldTag) : Nat := baseFieldBitCount

def Value.fieldValue (value : Value) (tag : FieldTag) : Nat :=
  ((value tag.component) tag.coordinate).val

theorem Value.fieldValue_lt_width (value : Value) (tag : FieldTag) :
    value.fieldValue tag < 2 ^ tag.bitWidth := by
  exact ((value tag.component) tag.coordinate).property.trans (by
    norm_num [ShiftedTernary41V1.modulus, FieldTag.bitWidth,
      baseFieldBitCount])

theorem Value.fieldValue_injective :
    Function.Injective Value.fieldValue := by
  intro left right equal
  funext component coordinate
  apply Subtype.ext
  exact congrFun equal { component := component, coordinate := coordinate }

def encodeWord :=
  Nightstream.Implementation.NebulaV2.WasmStateCodec.encodeWord

def encodeFields (value : Value) (tag : FieldTag) : List Nat :=
  encodeWord tag.bitWidth (value.fieldValue tag)

theorem encodeFields_length (value : Value) (tag : FieldTag) :
    (encodeFields value tag).length = tag.bitWidth :=
  Nightstream.Implementation.NebulaV2.WasmStateCodec.encodeWord_length _ _

theorem encodeFields_binary
    (value : Value) (tag : FieldTag) (digit : Nat)
    (member : digit ∈ encodeFields value tag) :
    digit < 2 :=
  Nightstream.Implementation.NebulaV2.WasmStateCodec.encodeWord_binary
    _ _ _ member

def encodeFor (tags : List FieldTag) (value : Value) : List Nat :=
  tags.flatMap (encodeFields value)

def encode (value : Value) : List Nat :=
  encodeFor schema value

theorem encodeFor_length (tags : List FieldTag) (value : Value) :
    (encodeFor tags value).length =
      (tags.map FieldTag.bitWidth).sum := by
  induction tags with
  | nil => rfl
  | cons tag rest inductionHypothesis =>
      simp [encodeFor, encodeFields_length]

/-- The 64-bit word at the first occurrence of a listed field tag is the
exact canonical encoding of that bundle coordinate. -/
theorem encodeFor_word_at
    (value : Value) (tag : FieldTag) :
    ∀ {tags : List FieldTag}, tag ∈ tags →
      ((encodeFor tags value).drop
          (baseFieldBitCount * tags.idxOf tag)).take baseFieldBitCount =
        encodeFields value tag := by
  intro tags member
  induction tags with
  | nil => simp at member
  | cons head tail inductionHypothesis =>
      simp only [List.mem_cons] at member
      by_cases same : head = tag
      · subst head
        simp [encodeFor, FieldTag.bitWidth, baseFieldBitCount,
          encodeFields_length]
      · have different : tag ≠ head := fun equal => same equal.symm
        have tailMember : tag ∈ tail := member.resolve_left different
        rw [List.idxOf_cons_ne (a := tag) (b := head) tail same]
        rw [show baseFieldBitCount * (List.idxOf tag tail + 1) =
            baseFieldBitCount +
              baseFieldBitCount * List.idxOf tag tail by
          simp [baseFieldBitCount, Nat.mul_add, Nat.add_comm]]
        simp only [encodeFor, List.flatMap_cons]
        rw [← List.drop_drop]
        rw [show baseFieldBitCount = (encodeFields value head).length by
          exact (encodeFields_length value head).symm]
        rw [List.drop_left]
        simpa [encodeFields_length, FieldTag.bitWidth, baseFieldBitCount] using
          inductionHypothesis tailMember

theorem schema_length :
    schema.length = bundleComponentCount * commitmentFieldCount := by
  simp [schema, componentOrder, bundleComponentCount]
  omega

private theorem sum_bitWidths (tags : List FieldTag) :
    (tags.map FieldTag.bitWidth).sum =
      tags.length * baseFieldBitCount := by
  induction tags with
  | nil => simp
  | cons head tail inductionHypothesis =>
      simp [FieldTag.bitWidth, inductionHypothesis, baseFieldBitCount]
      omega

theorem schema_width_exact :
    (schema.map FieldTag.bitWidth).sum = mandatoryBundleBits := by
  rw [sum_bitWidths, schema_length]
  rfl

theorem encode_length (value : Value) :
    (encode value).length = mandatoryBundleBits := by
  rw [encode, encodeFor_length, schema_width_exact]

theorem encode_exact_length (value : Value) :
    (encode value).length = 248832 := by
  rw [encode_length, mandatoryBundleBits_exact]

theorem encode_binary (value : Value) (digit : Nat)
    (member : digit ∈ encode value) :
    digit < 2 := by
  simp only [encode, encodeFor, List.mem_flatMap] at member
  obtain ⟨tag, _tagMember, digitMember⟩ := member
  exact encodeFields_binary value tag digit digitMember

theorem FieldTag.mem_schema (tag : FieldTag) : tag ∈ schema := by
  rcases tag with ⟨component, coordinate⟩
  cases component <;> simp [schema, componentOrder]

private theorem encodeFor_equal_at_member
    {left right : Value}
    {tags : List FieldTag}
    (equal : encodeFor tags left = encodeFor tags right)
    {tag : FieldTag} (member : tag ∈ tags) :
    left.fieldValue tag = right.fieldValue tag := by
  induction tags with
  | nil => simp at member
  | cons head tail inductionHypothesis =>
      rcases List.mem_cons.mp member with tagEqual | tailMember
      · subst tag
        have headEqual := congrArg (List.take head.bitWidth) equal
        have wordEqual :
            encodeFields left head = encodeFields right head := by
          simpa [encodeFor, encodeFields_length] using headEqual
        exact
          Nightstream.Implementation.NebulaV2.WasmStateCodec.encodeWord_injective_of_bound
            (left.fieldValue_lt_width head)
            (right.fieldValue_lt_width head)
            wordEqual
      · have tailEqual := congrArg (List.drop head.bitWidth) equal
        have exactTail : encodeFor tail left = encodeFor tail right := by
          simpa [encodeFor, encodeFields_length] using tailEqual
        exact inductionHypothesis exactTail tailMember

/-- The complete mandatory bundle has one canonical bit encoding. -/
theorem encode_injective : Function.Injective encode := by
  intro left right equal
  apply Value.fieldValue_injective
  funext tag
  exact encodeFor_equal_at_member equal tag.mem_schema

end Nightstream.Implementation.NebulaV2.CommitmentBundleCodec
