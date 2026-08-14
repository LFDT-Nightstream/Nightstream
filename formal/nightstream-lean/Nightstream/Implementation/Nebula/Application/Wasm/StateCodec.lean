import Mathlib.Data.Nat.Digits.Lemmas
import Nightstream.Protocol.Nebula.WasmStateEncoding

/-!
Contract: canonical bit codec for the complete V2 WASM public state.

Assurance tier: implementation model.

Owns the exact fixed-width little-endian binary encoding of all 55 tagged
state fields, the 2,293-bit concatenation order, canonical-domain bounds, and
injectivity of both the tagged and flattened encodings.

Does not own byte-container framing, Rust parsing, public-column placement,
generated Boolean rows, or the verifier-key manifest digest. Those layers
must refine this codec exactly.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.WasmStateCodec

open Nightstream.Protocol.Nebula.Fingerprint
open Nightstream.Protocol.Nebula.WasmState
open Nightstream.Protocol.Nebula.WasmStateEncoding

/-- Fixed-width little-endian binary encoding. The modulo makes encoding
total; all soundness theorems restrict it to values below `2^width`. -/
def encodeWord (width value : Nat) : List Nat :=
  Nat.digitsAppend 2 width (value % 2 ^ width)

theorem encodeWord_length (width value : Nat) :
    (encodeWord width value).length = width := by
  exact Nat.length_digitsAppend (by decide) width
    (Nat.mod_lt _ (by positivity))

theorem encodeWord_binary (width value digit : Nat)
    (member : digit ∈ encodeWord width value) :
    digit < 2 := by
  exact Nat.lt_of_mem_digitsAppend (by decide) width digit member

theorem ofDigits_encodeWord (width value : Nat) :
    Nat.ofDigits 2 (encodeWord width value) = value % 2 ^ width := by
  unfold encodeWord Nat.digitsAppend
  rw [Nat.ofDigits_append_replicate_zero, Nat.ofDigits_digits]

theorem ofDigits_encodeWord_of_bound
    {width value : Nat} (bounded : value < 2 ^ width) :
    Nat.ofDigits 2 (encodeWord width value) = value := by
  rw [ofDigits_encodeWord, Nat.mod_eq_of_lt bounded]

theorem encodeWord_injective_of_bound
    {width left right : Nat}
    (leftBound : left < 2 ^ width)
    (rightBound : right < 2 ^ width)
    (equal : encodeWord width left = encodeWord width right) :
    left = right := by
  have decoded := congrArg (Nat.ofDigits 2) equal
  simpa [ofDigits_encodeWord_of_bound leftBound,
    ofDigits_encodeWord_of_bound rightBound] using decoded

/-- Exact tagged words before concatenation. -/
def encodeFields (image : Image) (tag : FieldTag) : List Nat :=
  encodeWord tag.bitWidth (image.fieldValue tag)

theorem encodeFields_length (image : Image) (tag : FieldTag) :
    (encodeFields image tag).length = tag.bitWidth :=
  encodeWord_length _ _

theorem encodeFields_binary
    (image : Image) (tag : FieldTag) (digit : Nat)
    (member : digit ∈ encodeFields image tag) :
    digit < 2 :=
  encodeWord_binary _ _ _ member

/-- Every canonical semantic field fits its exact declared bit width. -/
theorem fieldValue_lt_width
    {image : Image} (canonical : image.Canonical) (tag : FieldTag) :
    image.fieldValue tag < 2 ^ tag.bitWidth := by
  cases tag with
  | pc =>
      simpa [Image.Canonical, decode, Image.fieldValue,
        FieldTag.bitWidth, u64Limit] using canonical.pcBound
  | operandStackPointer =>
      simpa [Image.Canonical, decode, Image.fieldValue,
        FieldTag.bitWidth, u64Limit] using
        canonical.operandStackPointerBound
  | stackFrameBase =>
      simpa [Image.Canonical, decode, Image.fieldValue,
        FieldTag.bitWidth, u64Limit] using canonical.stackFrameBaseBound
  | outputEnabled => exact Bool.toNat_lt _
  | outputLow =>
      simpa [Image.Canonical, decode, Image.fieldValue,
        FieldTag.bitWidth, u32Limit, OutputState.Canonical] using
        canonical.outputCanonical.1
  | outputHigh =>
      simpa [Image.Canonical, decode, Image.fieldValue,
        FieldTag.bitWidth, u32Limit, OutputState.Canonical] using
        canonical.outputCanonical.2.1
  | callStackDepth =>
      simpa [Image.Canonical, decode, Image.fieldValue,
        FieldTag.bitWidth, u64Limit] using canonical.callStackDepthBound
  | memoryPagesPresent => exact Bool.toNat_lt _
  | memoryPagesValue =>
      simpa [Image.Canonical, decode, Image.fieldValue,
        FieldTag.bitWidth, u32Limit, OptionalU32.Canonical] using
        canonical.memoryPagesCanonical.1
  | maximumMemoryPagesPresent => exact Bool.toNat_lt _
  | maximumMemoryPagesValue =>
      simpa [Image.Canonical, decode, Image.fieldValue,
        FieldTag.bitWidth, u32Limit, OptionalU32.Canonical] using
        canonical.maximumMemoryPagesCanonical.1
  | localsFrameBase =>
      simpa [Image.Canonical, decode, Image.fieldValue,
        FieldTag.bitWidth, u64Limit] using canonical.localsFrameBaseBound
  | halted => exact Bool.toNat_lt _
  | trapped => exact Bool.toNat_lt _
  | trapCode =>
      simpa [Image.Canonical, decode, Image.fieldValue,
        FieldTag.bitWidth, u32Limit] using canonical.trapCodeBound
  | parameterInitializationActive => exact Bool.toNat_lt _
  | parameterInitializationRemaining =>
      simpa [Image.Canonical, decode, Image.fieldValue,
        FieldTag.bitWidth, u32Limit, Countdown.Canonical] using
        canonical.parameterInitializationCanonical.1
  | tailCallPending => exact Bool.toNat_lt _
  | hostArgumentsActive => exact Bool.toNat_lt _
  | hostArgumentsRemaining =>
      simpa [Image.Canonical, decode, Image.fieldValue,
        FieldTag.bitWidth, u32Limit, Countdown.Canonical] using
        canonical.hostArgumentsCanonical.1
  | hostResultPending => exact Bool.toNat_lt _
  | hostCalleeFunction =>
      simpa [Image.Canonical, decode, Image.fieldValue,
        FieldTag.bitWidth, u32Limit] using canonical.hostCalleeFunctionBound
  | hostEventChain index =>
      exact Nat.lt_trans (canonical.hostEventChainCanonical index)
        (by norm_num [goldilocksModulus, FieldTag.bitWidth])
  | eventBuffer index =>
      exact Nat.lt_trans (canonical.eventBufferCanonical index)
        (by norm_num [goldilocksModulus, FieldTag.bitWidth])
  | eventBufferSlot =>
      simpa [Image.Canonical, decode, Image.fieldValue,
        FieldTag.bitWidth, eventBufferSlots] using
        canonical.eventBufferSlotBound
  | permutationPending => exact Bool.toNat_lt _
  | permutationRound =>
      have bound := canonical.permutationRoundBound
      simp only [decode] at bound
      simp only [Image.fieldValue, FieldTag.bitWidth]
      norm_num [eventPermutationRows] at bound ⊢
      omega
  | permutationState index =>
      exact Nat.lt_trans (canonical.permutationStateCanonical index)
        (by norm_num [goldilocksModulus, FieldTag.bitWidth])
  | grammarMode => exact Bool.toNat_lt _
  | grammarTurnExportFunction =>
      simpa [Image.Canonical, decode, Image.fieldValue,
        FieldTag.bitWidth, u32Limit] using
        canonical.grammarTurnFunctionBound
  | grammarEventsRemaining =>
      simpa [Image.Canonical, decode, Image.fieldValue,
        FieldTag.bitWidth, u32Limit] using
        canonical.grammarEventsRemainingBound
  | grammarEventIndex =>
      simpa [Image.Canonical, decode, Image.fieldValue,
        FieldTag.bitWidth, u32Limit] using canonical.grammarEventIndexBound
  | grammarArgumentsBase =>
      simpa [Image.Canonical, decode, Image.fieldValue,
        FieldTag.bitWidth, u64Limit] using
        canonical.grammarArgumentsBaseBound
  | grammarSlotCursor =>
      simpa [Image.Canonical, decode, Image.fieldValue,
        FieldTag.bitWidth, grammarSlots] using
        canonical.grammarSlotCursorBound

theorem encodeFields_injective_on_canonical
    {left right : Image}
    (leftCanonical : left.Canonical)
    (rightCanonical : right.Canonical)
    (equal : encodeFields left = encodeFields right) :
    left = right := by
  apply Image.fieldValue_injective
  funext tag
  exact encodeWord_injective_of_bound
    (fieldValue_lt_width leftCanonical tag)
    (fieldValue_lt_width rightCanonical tag)
    (congrFun equal tag)

/-- Concatenate the 55 field words in the protocol-owned schema order. -/
def encodeFor (tags : List FieldTag) (image : Image) : List Nat :=
  tags.flatMap (encodeFields image)

def encode (image : Image) : List Nat :=
  encodeFor schema image

theorem encodeFor_length (tags : List FieldTag) (image : Image) :
    (encodeFor tags image).length =
      (tags.map FieldTag.bitWidth).sum := by
  induction tags with
  | nil => rfl
  | cons tag rest inductionHypothesis =>
      simp [encodeFor, encodeFields_length]

theorem encode_length (image : Image) :
    (encode image).length = serializedBitCount := by
  exact encodeFor_length schema image

theorem encode_exact_length (image : Image) :
    (encode image).length = 2293 := by
  rw [encode_length, serializedBitCount_eq]

theorem encode_binary (image : Image) (digit : Nat)
    (member : digit ∈ encode image) :
    digit < 2 := by
  simp only [encode, encodeFor, List.mem_flatMap] at member
  obtain ⟨tag, tagMember, digitMember⟩ := member
  exact encodeFields_binary image tag digit digitMember

private theorem encodeFor_equal_at_member
    {left right : Image}
    (leftCanonical : left.Canonical)
    (rightCanonical : right.Canonical)
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
        exact encodeWord_injective_of_bound
          (fieldValue_lt_width leftCanonical head)
          (fieldValue_lt_width rightCanonical head)
          wordEqual
      · have tailEqual := congrArg (List.drop head.bitWidth) equal
        have exactTail :
            encodeFor tail left = encodeFor tail right := by
          simpa [encodeFor, encodeFields_length] using tailEqual
        exact inductionHypothesis exactTail tailMember

theorem FieldTag.mem_schema (tag : FieldTag) : tag ∈ schema := by
  cases tag <;> simp [schema]
  case hostEventChain index => fin_cases index <;> simp
  case eventBuffer index => fin_cases index <;> simp
  case permutationState index => fin_cases index <;> simp

/-- The exact flattened 2,293-bit public state has one canonical semantic
preimage. -/
theorem encode_injective_on_canonical
    {left right : Image}
    (leftCanonical : left.Canonical)
    (rightCanonical : right.Canonical)
    (equal : encode left = encode right) :
    left = right := by
  apply Image.fieldValue_injective
  funext tag
  exact encodeFor_equal_at_member leftCanonical rightCanonical
    equal (FieldTag.mem_schema tag)

end Nightstream.Implementation.Nebula.WasmStateCodec
