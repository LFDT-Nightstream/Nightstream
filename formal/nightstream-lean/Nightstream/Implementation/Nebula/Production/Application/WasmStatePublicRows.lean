import Nightstream.Implementation.Nebula.Core.BoundedWordRows
import Nightstream.Implementation.Nebula.Production.Application.WasmStateFields
import Nightstream.Implementation.Nebula.Core.TaggedBitSlices

/-!
Contract: exact public-bit to field-native rows for one V2 WASM state.

The source is the normative 2,293-bit `WasmStateCodec` image. The target is
the exact 85-field `ProductionWasmStateFields` image. Each target field is
derived by one integer-safe recomposition row. A 64-bit state word is split
into two 32-bit limbs before field arithmetic.

`BitsPlaced` is the only parser boundary. It states exact placement of the
2,293 verifier-owned source bits. It contains no recombined field equality.

Assurance tier: implementation-to-codec bridge.

Does not own byte parsing, the generated public-column allocation, WASM
transition rows, or Rust refinement.

Emits constraints: 85 rows.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Nebula.ProductionWasmStatePublicRows

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.BoundedWordRows
open Nightstream.Implementation.Nebula.TaggedBitSlices
open Nightstream.Implementation.Nebula.WasmStateCodec
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Protocol.Nebula.WasmStateEncoding

inductive Limb where
  | whole
  | low
  | high
deriving DecidableEq, Repr

/-- One field-native coordinate and the source word that determines it. -/
structure Piece where
  tag : FieldTag
  limb : Limb
deriving DecidableEq, Repr

def piecesForTag (tag : FieldTag) : List Piece :=
  if tag.bitWidth = 64 then
    [{ tag := tag, limb := .low }, { tag := tag, limb := .high }]
  else
    [{ tag := tag, limb := .whole }]

def piecesFor (tags : List FieldTag) : List Piece :=
  tags.flatMap piecesForTag

/-- Pieces in the exact production state schema order. -/
def pieces : List Piece :=
  piecesFor schema

def Piece.width (piece : Piece) : Nat :=
  match piece.limb with
  | .whole => piece.tag.bitWidth
  | .low | .high => 32

def Piece.value (piece : Piece) (image : Image) : Nat :=
  match piece.limb with
  | .whole => image.fieldValue piece.tag
  | .low => image.fieldValue piece.tag % (2 ^ 32)
  | .high => image.fieldValue piece.tag / (2 ^ 32)

def encodePiece (image : Image) (piece : Piece) : List Nat :=
  encodeWord piece.width (piece.value image)

def encodePiecesFor (tags : List FieldTag) (image : Image) : List Nat :=
  (piecesFor tags).flatMap (encodePiece image)

def pieceValuesFor (tags : List FieldTag) (image : Image) : List Nat :=
  (piecesFor tags).map (fun piece => piece.value image)

theorem encodeWord_sixtyFour_split
    {value : Nat} (bounded : value < 2 ^ 64) :
    encodeWord 64 value =
      encodeWord 32 (value % (2 ^ 32)) ++
        encodeWord 32 (value / (2 ^ 32)) := by
  apply Nat.ofDigits_inj_of_len_eq (by decide : 1 < 2)
  · simp [encodeWord_length]
  · intro digit member
    exact encodeWord_binary _ _ _ member
  · intro digit member
    rw [List.mem_append] at member
    rcases member with low | high
    · exact encodeWord_binary _ _ _ low
    · exact encodeWord_binary _ _ _ high
  · rw [ofDigits_encodeWord_of_bound bounded, Nat.ofDigits_append,
      ofDigits_encodeWord_of_bound (Nat.mod_lt _ (by positivity))]
    have highBound : value / (2 ^ 32) < 2 ^ 32 := by
      norm_num at bounded ⊢
      omega
    rw [ofDigits_encodeWord_of_bound highBound, encodeWord_length]
    exact (Nat.mod_add_div value (2 ^ 32)).symm

private theorem encodePiecesForTag_eq
    {image : Image} (canonical : image.Canonical) (tag : FieldTag) :
    (piecesForTag tag).flatMap (encodePiece image) =
      encodeFields image tag := by
  by_cases wide : tag.bitWidth = 64
  · have bounded : image.fieldValue tag < 2 ^ 64 := by
      simpa [wide] using fieldValue_lt_width canonical tag
    simp [piecesForTag, wide, encodePiece, Piece.width, Piece.value,
      encodeFields, encodeWord_sixtyFour_split bounded]
  · simp [piecesForTag, wide, encodePiece, Piece.width, Piece.value,
      encodeFields]

/-- The piece codec is exactly the normative 2,293-bit state codec. -/
theorem encodePiecesFor_eq
    {image : Image} (canonical : image.Canonical) (tags : List FieldTag) :
    encodePiecesFor tags image = WasmStateCodec.encodeFor tags image := by
  induction tags with
  | nil => rfl
  | cons tag rest inductionHypothesis =>
      simp only [encodePiecesFor, piecesFor, List.flatMap_cons,
        List.flatMap_append, WasmStateCodec.encodeFor]
      rw [encodePiecesForTag_eq canonical tag]
      exact congrArg (List.append (encodeFields image tag))
        inductionHypothesis

theorem encodePieces_eq
    {image : Image} (canonical : image.Canonical) :
    pieces.flatMap (encodePiece image) = WasmStateCodec.encode image := by
  exact encodePiecesFor_eq canonical schema

private theorem pieceValuesForTag_eq (image : Image) (tag : FieldTag) :
    (piecesForTag tag).map (fun piece => piece.value image) =
      ProductionWasmStateFields.encodeTag image tag := by
  by_cases wide : tag.bitWidth = 64
  · simp only [piecesForTag, wide, if_pos, List.map_cons, List.map_nil,
      Piece.value, ProductionWasmStateFields.encodeTag]
    change
      [image.fieldValue tag % (2 ^ 32),
        image.fieldValue tag / (2 ^ 32)] =
      Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.u64Halves
        (image.fieldValue tag)
    rfl
  · simp [piecesForTag, wide, Piece.value,
      ProductionWasmStateFields.encodeTag]

/-- Piece values are exactly the 85 field-native state coordinates. -/
theorem pieceValuesFor_eq (tags : List FieldTag) (image : Image) :
    pieceValuesFor tags image =
      ProductionWasmStateFields.encodeFor tags image := by
  induction tags with
  | nil => rfl
  | cons tag rest inductionHypothesis =>
      simp only [pieceValuesFor, piecesFor, List.flatMap_cons,
        List.map_append, ProductionWasmStateFields.encodeFor]
      rw [pieceValuesForTag_eq image tag]
      exact congrArg
        (List.append (ProductionWasmStateFields.encodeTag image tag))
        inductionHypothesis

theorem pieceValues_eq (image : Image) :
    pieces.map (fun piece => piece.value image) =
      ProductionWasmStateFields.encode image := by
  exact pieceValuesFor_eq schema image

theorem pieces_length_exact : pieces.length = 85 := by
  decide

theorem pieces_width_sum_exact : (pieces.map Piece.width).sum = 2293 := by
  decide

private theorem member_piecesForTag_shape
    {tag : FieldTag} {piece : Piece}
    (member : piece ∈ piecesForTag tag) :
    (tag.bitWidth = 64 ∧
      (piece = { tag := tag, limb := .low } ∨
        piece = { tag := tag, limb := .high })) ∨
      (tag.bitWidth ≠ 64 ∧
        piece = { tag := tag, limb := .whole }) := by
  by_cases wide : tag.bitWidth = 64
  · left
    constructor
    · exact wide
    · simpa [piecesForTag, wide] using member
  · right
    constructor
    · exact wide
    · simpa [piecesForTag, wide] using member

private theorem tag_width_positive (tag : FieldTag) :
    0 < tag.bitWidth := by
  cases tag <;> norm_num [FieldTag.bitWidth]

private theorem tag_width_le_32_of_ne_64
    (tag : FieldTag) (notWide : tag.bitWidth ≠ 64) :
    tag.bitWidth ≤ 32 := by
  cases tag <;> simp_all [FieldTag.bitWidth]

theorem piece_width_positive {piece : Piece} (member : piece ∈ pieces) :
    0 < piece.width := by
  rw [pieces, piecesFor, List.mem_flatMap] at member
  obtain ⟨tag, _, inTag⟩ := member
  rcases member_piecesForTag_shape inTag with
    ⟨_, rfl | rfl⟩ | ⟨_, rfl⟩
  · norm_num [Piece.width]
  · norm_num [Piece.width]
  · exact tag_width_positive tag

theorem piece_width_le_32 {piece : Piece} (member : piece ∈ pieces) :
    piece.width ≤ 32 := by
  rw [pieces, piecesFor, List.mem_flatMap] at member
  obtain ⟨tag, _, inTag⟩ := member
  rcases member_piecesForTag_shape inTag with
    ⟨_, rfl | rfl⟩ | ⟨notWide, rfl⟩
  · norm_num [Piece.width]
  · norm_num [Piece.width]
  · simpa [Piece.width] using tag_width_le_32_of_ne_64 tag notWide

theorem piece_width_fits_goldilocks
    {piece : Piece} (member : piece ∈ pieces) :
    2 ^ piece.width ≤ goldilocksP := by
  have width := piece_width_le_32 member
  have power : 2 ^ piece.width ≤ 2 ^ 32 :=
    Nat.pow_le_pow_right (by decide) width
  exact power.trans (by norm_num [goldilocksP])

theorem piece_value_bound
    {image : Image} (canonical : image.Canonical)
    {piece : Piece} (member : piece ∈ pieces) :
    piece.value image < 2 ^ piece.width := by
  rw [pieces, piecesFor, List.mem_flatMap] at member
  obtain ⟨tag, _, inTag⟩ := member
  rcases member_piecesForTag_shape inTag with
    ⟨wide, rfl | rfl⟩ | ⟨notWide, rfl⟩
  · simp only [Piece.value, Piece.width]
    exact Nat.mod_lt _ (by positivity)
  · simp only [Piece.value, Piece.width]
    have bounded := fieldValue_lt_width canonical tag
    rw [wide] at bounded
    norm_num at bounded ⊢
    omega
  · simpa [Piece.value, Piece.width] using fieldValue_lt_width canonical tag

/-- Absolute columns for one state conversion. -/
structure Layout where
  publicBitStart : Nat
  fieldColumn : Fin 85 -> Nat

def pieceAt (index : Fin 85) : Piece :=
  pieces.get (Fin.cast pieces_length_exact.symm index)

def pieceOffset (index : Fin 85) : Nat :=
  TaggedBitSlices.offsetAt Piece.width pieces index.val

def Layout.word (layout : Layout) (index : Fin 85) :
    BoundedWordRows.Layout where
  width := (pieceAt index).width
  valueColumn := layout.fieldColumn index
  bitStart := layout.publicBitStart + pieceOffset index

def rows (layout : Layout) : List Row :=
  List.ofFn fun index : Fin 85 => (layout.word index).recompositionRow

theorem rows_length_exact (layout : Layout) : (rows layout).length = 85 := by
  simp [rows]

/-- Exact verifier-owned bit placement. No field-native value is assumed. -/
def BitsPlaced (layout : Layout) (assignment : Nat -> Nat)
    (image : Image) : Prop :=
  forall offset (bound : offset < 2293),
    assignment (layout.publicBitStart + offset) =
      (WasmStateCodec.encode image).get
        ⟨offset, by simpa [WasmStateCodec.encode_exact_length] using bound⟩

private theorem pieceAt_mem (index : Fin 85) : pieceAt index ∈ pieces := by
  exact List.get_mem pieces (Fin.cast pieces_length_exact.symm index)

private theorem pieceOffset_add_width_le (index : Fin 85) :
    pieceOffset index + (pieceAt index).width ≤ 2293 := by
  have bounded : (Fin.cast pieces_length_exact.symm index).val < pieces.length :=
    (Fin.cast pieces_length_exact.symm index).isLt
  have sliceLength := congrArg List.length
    (TaggedBitSlices.slice_flatten_at
      (fun piece => List.replicate piece.width ()) Piece.width
      (fun piece => by simp) pieces index.val (by simpa using bounded))
  have totalLength :
      (TaggedBitSlices.flatten
        (fun piece => List.replicate piece.width ()) pieces).length = 2293 := by
    change (pieces.flatMap (fun piece => List.replicate piece.width ())).length =
      2293
    simpa using pieces_width_sum_exact
  simp only [List.length_take, List.length_drop, List.length_replicate]
    at sliceLength
  rw [totalLength] at sliceLength
  have remaining := min_eq_left_iff.mp sliceLength
  have positive := piece_width_positive (pieceAt_mem index)
  have widthEqual :
      (pieceAt index).width =
        (pieces.get ⟨index.val, bounded⟩).width := by
    rfl
  rw [widthEqual] at positive
  change
    TaggedBitSlices.offsetAt Piece.width pieces index.val +
        (pieces.get ⟨index.val, bounded⟩).width ≤ 2293
  omega

private theorem word_digits_eq_encodePiece
    {layout : Layout} {assignment : Nat -> Nat} {image : Image}
    (canonicalImage : image.Canonical)
    (placed : BitsPlaced layout assignment image) (index : Fin 85) :
    (layout.word index).digits assignment = encodePiece image (pieceAt index) := by
  have codecEqual := encodePieces_eq canonicalImage
  have selected := TaggedBitSlices.slice_flatten_at
    (encodePiece image) Piece.width
    (fun piece => encodeWord_length piece.width (piece.value image))
    pieces index.val (by
      simpa [pieces_length_exact] using index.isLt)
  unfold TaggedBitSlices.flatten at selected
  rw [codecEqual] at selected
  have selectedExact :
      ((WasmStateCodec.encode image).drop (pieceOffset index)).take
          (pieceAt index).width = encodePiece image (pieceAt index) := by
    simpa [pieceAt, pieceOffset] using selected
  rw [← selectedExact]
  apply List.ext_get
  · rw [BoundedWordRows.Layout.digits_length, List.length_take,
      List.length_drop, WasmStateCodec.encode_exact_length]
    have bound := pieceOffset_add_width_le index
    have leSub : (pieceAt index).width ≤ 2293 - pieceOffset index := by
      omega
    exact (min_eq_left leSub).symm
  · intro offset leftBound rightBound
    have offsetBound : offset < (pieceAt index).width := by
      simpa [Layout.word, BoundedWordRows.Layout.digits] using leftBound
    have globalBound : pieceOffset index + offset < 2293 := by
      have := pieceOffset_add_width_le index
      omega
    have source := placed (pieceOffset index + offset) globalBound
    simpa [Layout.word, BoundedWordRows.Layout.digits,
      BoundedWordRows.Layout.bitColumn, pieceOffset,
      List.getElem_drop, Nat.add_assoc] using source

private theorem recompositionRow_mem
    (layout : Layout) (index : Fin 85) :
    (layout.word index).recompositionRow ∈ rows layout := by
  exact List.mem_ofFn.mpr ⟨index, rfl⟩

theorem fieldColumn_eq_pieceValue
    {layout : Layout} {assignment : Nat -> Nat} {image : Image}
    (canonicalImage : image.Canonical)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : BitsPlaced layout assignment image)
    (holds : Satisfies (rows layout) assignment)
    (index : Fin 85) :
    assignment (layout.fieldColumn index) = (pieceAt index).value image := by
  have rowHolds := holds _ (recompositionRow_mem layout index)
  have decodedBound :
      BoundedWordRows.decoded (layout.word index) assignment <
        2 ^ (layout.word index).width := by
    unfold BoundedWordRows.decoded
    rw [word_digits_eq_encodePiece canonicalImage placed index]
    change Nat.ofDigits 2
      (encodeWord (pieceAt index).width ((pieceAt index).value image)) <
        2 ^ (pieceAt index).width
    rw [ofDigits_encodeWord_of_bound
      (piece_value_bound canonicalImage (pieceAt_mem index))]
    exact piece_value_bound canonicalImage (pieceAt_mem index)
  have exact := BoundedWordRows.recompositionRow_sound_of_decoded_bound
    (piece_width_fits_goldilocks (pieceAt_mem index)) canonical one
    decodedBound rowHolds
  change assignment (layout.word index).valueColumn =
    (pieceAt index).value image
  rw [exact]
  unfold BoundedWordRows.decoded
  rw [word_digits_eq_encodePiece canonicalImage placed index]
  exact ofDigits_encodeWord_of_bound
    (piece_value_bound canonicalImage (pieceAt_mem index))

/-- The 85 row-derived target columns equal the exact field-native state
encoding. -/
theorem fields_exact
    {layout : Layout} {assignment : Nat -> Nat} {image : Image}
    (canonicalImage : image.Canonical)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : BitsPlaced layout assignment image)
    (holds : Satisfies (rows layout) assignment) :
    List.ofFn (fun index : Fin 85 => assignment (layout.fieldColumn index)) =
      ProductionWasmStateFields.encode image := by
  rw [← pieceValues_eq image]
  calc
    List.ofFn
        (fun index : Fin 85 => assignment (layout.fieldColumn index)) =
        List.ofFn
          (fun index : Fin 85 => (pieceAt index).value image) := by
      congr 1
      funext index
      exact fieldColumn_eq_pieceValue canonicalImage canonical one placed
        holds index
    _ = pieces.map (fun piece => piece.value image) := by
      have exactPieces : List.ofFn pieceAt = pieces := by
        apply List.ext_get
        · rw [List.length_ofFn, pieces_length_exact]
        · intro index leftBound rightBound
          rw [List.get_ofFn]
          rfl
      calc
        List.ofFn (fun index : Fin 85 => (pieceAt index).value image) =
            (List.ofFn pieceAt).map (fun piece => piece.value image) := by
          simpa [Function.comp_def] using
            (List.ofFn_comp' pieceAt (fun piece => piece.value image))
        _ = pieces.map (fun piece => piece.value image) := by
          rw [exactPieces]

end Nightstream.Implementation.Nebula.ProductionWasmStatePublicRows
