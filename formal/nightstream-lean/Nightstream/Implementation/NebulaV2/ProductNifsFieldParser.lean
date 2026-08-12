import Nightstream.Implementation.NebulaV2.ProductNifsCodec
import Nightstream.Implementation.NebulaV2.TaggedBitSlices
import Nightstream.Protocol.NebulaV2.CanonicalFieldBits

/-!
Contract: executable canonical-field parser for the V2 paper NIFS carrier.

Assurance tier: implementation model.

Owns exact 64-bit chunking, strict Goldilocks canonicality, field-vector
order, rejection of modulo aliases, and honest encode/parse completeness.

Does not own the structured running-claim layout, byte-container framing,
generated parser rows, Rust conformance, or NIFS verification.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.NebulaV2.ProductNifsFieldParser

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.ProductNifsCodec
open Nightstream.Protocol.NebulaV2
open Nightstream.SuperNeo.Concrete

abbrev Block (count : Nat) := FixedBits.Word (count * fieldBitWidth)

/-- The exact little-endian word at one field-vector position. -/
def fieldWord {count : Nat} (block : Block count) (index : Fin count) :
    CanonicalFieldBits.Word :=
  FixedBits.slice block (index.val * fieldBitWidth) fieldBitWidth (by
    have indexLt := index.isLt
    change index.val * 64 + 64 ≤ count * 64
    omega)

def AllCanonical {count : Nat} (block : Block count) : Prop :=
  ∀ index, CanonicalFieldBits.Canonical (fieldWord block index)

def decodedFields {count : Nat} (block : Block count)
    (canonical : AllCanonical block) : Fin count → F :=
  fun index =>
    ⟨CanonicalFieldBits.decode (fieldWord block index), by
      simpa [CanonicalFieldBits.Canonical,
        ShiftedTernary41V1.modulus, goldilocksModulus] using canonical index⟩

/-- Executable fail-closed parser. The finite universal check is decidable
and rejects as soon as any 64-bit limb is at least the field modulus. -/
def parse {count : Nat} (block : Block count) : Option (Fin count → F) :=
  letI : DecidablePred
      (fun index : Fin count =>
        CanonicalFieldBits.Canonical (fieldWord block index)) :=
    fun _ => Nat.decLt _ _
  letI : Decidable (AllCanonical block) :=
    Fintype.decidableForallFintype
  if canonical : AllCanonical block then
    some (decodedFields block canonical)
  else
    none

theorem parse_success_canonical
    {count : Nat} {block : Block count} {fields : Fin count → F}
    (accepted : parse block = some fields) : AllCanonical block := by
  unfold parse at accepted
  split at accepted
  next canonical => exact canonical
  next notCanonical => contradiction

theorem parse_success_fields
    {count : Nat} {block : Block count} {fields : Fin count → F}
    (accepted : parse block = some fields) :
    ∀ index, (fields index).val =
      CanonicalFieldBits.decode (fieldWord block index) := by
  unfold parse at accepted
  split at accepted
  next canonical =>
    have equal := Option.some.inj accepted
    intro index
    rw [← equal]
    rfl
  next notCanonical => contradiction

theorem parse_rejects_noncanonical
    {count : Nat} (block : Block count)
    (notCanonical : ¬ AllCanonical block) : parse block = none := by
  simp [parse, notCanonical]

theorem parse_rejects_modulus_word
    {count : Nat} (block : Block count) (index : Fin count)
    (modulusAt : fieldWord block index = CanonicalFieldBits.modulusWord) :
    parse block = none := by
  apply parse_rejects_noncanonical
  intro canonical
  have atIndex := canonical index
  rw [modulusAt] at atIndex
  exact CanonicalFieldBits.modulusWord_not_canonical atIndex

def valuesList {count : Nat} (values : Fin count → F) : List F :=
  List.ofFn values

theorem valuesList_eq {count : Nat} (values : Fin count → F) :
    valuesList values = List.ofFn values :=
  rfl

def encode {count : Nat} (values : Fin count → F) : Block count :=
  ⟨encodeFieldBits (valuesList values), by
    rw [encodeFieldBits_length]
    simp [valuesList],
    fun digit member => encodeFieldBits_binary _ digit member⟩

theorem encode_value {count : Nat} (values : Fin count → F) :
    (encode values).val = encodeFieldBits (valuesList values) :=
  rfl

def wordOfField (value : F) : CanonicalFieldBits.Word :=
  ⟨fieldBits value, fieldBits_length value,
    fun digit member => fieldBits_binary value digit member⟩

private theorem sum_map_constant
    {Alpha : Type} (values : List Alpha) (width : Nat) :
    (values.map fun _ => width).sum = values.length * width := by
  induction values with
  | nil => simp
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, List.sum_cons, List.length_cons,
        inductionHypothesis, Nat.succ_mul]
      omega

private theorem flatMap_eq_map_flatten
    {Alpha Beta : Type} (function : Alpha → List Beta) (values : List Alpha) :
    values.flatMap function = (values.map function).flatten := by
  induction values with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp [List.flatMap, inductionHypothesis]

private theorem constantOffset
    {count : Nat} (values : Fin count → F) (index : Fin count) :
    TaggedBitSlices.offsetAt (fun _ : F => fieldBitWidth)
        (valuesList values) index.val =
      index.val * fieldBitWidth := by
  rw [TaggedBitSlices.offsetAt, sum_map_constant]
  rw [List.length_take]
  rw [show (valuesList values).length = count by simp [valuesList]]
  rw [Nat.min_eq_left (Nat.le_of_lt index.isLt)]

theorem fieldWord_encode
    {count : Nat} (values : Fin count → F) (index : Fin count) :
    fieldWord (encode values) index = wordOfField (values index) := by
  apply Subtype.ext
  change
    ((encodeFieldBits (valuesList values)).drop
      (index.val * fieldBitWidth)).take fieldBitWidth =
        fieldBits (values index)
  have sliced := TaggedBitSlices.slice_flatten_at
    fieldBits (fun _ : F => fieldBitWidth) fieldBits_length
    (valuesList values) index.val (by simp [valuesList])
  rw [constantOffset values index] at sliced
  unfold TaggedBitSlices.flatten at sliced
  rw [flatMap_eq_map_flatten fieldBits (valuesList values)] at sliced
  simpa [encodeFieldBits, fieldBlocks, TaggedBitSlices.flatten,
    valuesList] using sliced

theorem encode_allCanonical
    {count : Nat} (values : Fin count → F) :
    AllCanonical (encode values) := by
  intro index
  rw [fieldWord_encode]
  change Nat.ofDigits 2 (fieldBits (values index)) <
    ShiftedTernary41V1.modulus
  change Nat.ofDigits 2
    (WasmStateCodec.encodeWord fieldBitWidth (values index).val) <
      ShiftedTernary41V1.modulus
  rw [WasmStateCodec.ofDigits_encodeWord_of_bound]
  · simpa [ShiftedTernary41V1.modulus, goldilocksModulus] using
      (values index).isLt
  · exact Nat.lt_trans (values index).isLt (by
      norm_num [goldilocksModulus, fieldBitWidth])

theorem parse_encode
    {count : Nat} (values : Fin count → F) :
    parse (encode values) = some values := by
  rw [parse, dif_pos (encode_allCanonical values)]
  apply congrArg some
  funext index
  apply Fin.ext
  change CanonicalFieldBits.decode (fieldWord (encode values) index) =
    (values index).val
  rw [fieldWord_encode]
  change Nat.ofDigits 2
    (WasmStateCodec.encodeWord fieldBitWidth (values index).val) =
      (values index).val
  exact WasmStateCodec.ofDigits_encodeWord_of_bound
    (Nat.lt_trans (values index).isLt (by
      norm_num [goldilocksModulus, fieldBitWidth]))

end Nightstream.Implementation.NebulaV2.ProductNifsFieldParser
