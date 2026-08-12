import Nightstream.Implementation.NebulaV2.WasmStateCodec
import Nightstream.Protocol.NebulaV2.WasmStatement

/-!
Contract: canonical bit codec for the complete V2 WASM result image.

Assurance tier: implementation model.

Owns the exact 2,665-bit result order, four-lane digest encoding, fixed block
framing, and injectivity on images accepted by the independent terminal-result
relation.

Does not own container bytes, Rust parsing, generated public columns, terminal
proof verification, Poseidon2 security, or proof-system extraction.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.WasmResultCodec

open Nightstream.Implementation.NebulaV2.WasmStateCodec
open Nightstream.Protocol.NebulaV2.Completion
open Nightstream.Protocol.NebulaV2.Digest
open Nightstream.Protocol.NebulaV2.WasmState
open Nightstream.Protocol.NebulaV2.WasmStateEncoding
open Nightstream.Protocol.NebulaV2.WasmStatement

def encodeDigestAux :
    (count : Nat) →
      (Fin count →
        Nightstream.Protocol.NebulaV2.ShiftedTernary41V1.CanonicalGoldilocks) →
        List Nat
  | 0, _ => []
  | count + 1, lanes =>
      encodeWord laneBitWidth (lanes 0).val ++
        encodeDigestAux count (fun lane => lanes lane.succ)

def encodeDigest
    (digest : Nightstream.Protocol.NebulaV2.Digest.Value) : List Nat :=
  encodeDigestAux laneCount digest.lanes

theorem encodeDigestAux_length
    (count : Nat)
    (lanes : Fin count →
      Nightstream.Protocol.NebulaV2.ShiftedTernary41V1.CanonicalGoldilocks) :
    (encodeDigestAux count lanes).length = count * laneBitWidth := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp [encodeDigestAux, encodeWord_length, inductionHypothesis,
        Nat.succ_mul]
      omega

theorem encodeDigest_length
    (digest : Nightstream.Protocol.NebulaV2.Digest.Value) :
    (encodeDigest digest).length =
      Nightstream.Protocol.NebulaV2.Digest.serializedBitCount := by
  exact encodeDigestAux_length laneCount digest.lanes

theorem encodeDigest_exact_length
    (digest : Nightstream.Protocol.NebulaV2.Digest.Value) :
    (encodeDigest digest).length = 256 := by
  rw [encodeDigest_length,
    Nightstream.Protocol.NebulaV2.Digest.serializedBitCount_eq]

theorem encodeDigestAux_binary
    (count : Nat)
    (lanes : Fin count →
      Nightstream.Protocol.NebulaV2.ShiftedTernary41V1.CanonicalGoldilocks)
    (digit : Nat)
    (member : digit ∈ encodeDigestAux count lanes) :
    digit < 2 := by
  induction count with
  | zero => simp [encodeDigestAux] at member
  | succ count inductionHypothesis =>
      simp only [encodeDigestAux, List.mem_append] at member
      rcases member with head | tail
      · exact encodeWord_binary _ _ _ head
      · exact inductionHypothesis (fun lane => lanes lane.succ) tail

theorem encodeDigest_binary
    (digest : Nightstream.Protocol.NebulaV2.Digest.Value)
    (digit : Nat) (member : digit ∈ encodeDigest digest) :
    digit < 2 :=
  encodeDigestAux_binary laneCount digest.lanes digit member

private theorem encodeDigestAux_injective
    (count : Nat)
    {left right :
      Fin count →
        Nightstream.Protocol.NebulaV2.ShiftedTernary41V1.CanonicalGoldilocks}
    (equal : encodeDigestAux count left = encodeDigestAux count right) :
    left = right := by
  induction count with
  | zero =>
      funext lane
      exact Fin.elim0 lane
  | succ count inductionHypothesis =>
      have headWords := congrArg (List.take laneBitWidth) equal
      have headEqualValue : (left 0).val = (right 0).val := by
        apply encodeWord_injective_of_bound (width := laneBitWidth)
          (Nat.lt_trans (left 0).property
            (by norm_num
              [Nightstream.Protocol.NebulaV2.ShiftedTernary41V1.modulus,
                laneBitWidth]))
          (Nat.lt_trans (right 0).property
            (by norm_num
              [Nightstream.Protocol.NebulaV2.ShiftedTernary41V1.modulus,
                laneBitWidth]))
        simpa [encodeDigestAux, encodeWord_length] using headWords
      have headEqual : left 0 = right 0 :=
        Subtype.ext headEqualValue
      have tailWords := congrArg (List.drop laneBitWidth) equal
      have tailEncoded :
          encodeDigestAux count (fun lane => left lane.succ) =
            encodeDigestAux count (fun lane => right lane.succ) := by
        simpa [encodeDigestAux, encodeWord_length] using tailWords
      have tailEqual := inductionHypothesis tailEncoded
      funext lane
      exact Fin.cases headEqual
        (fun tail => congrFun tailEqual tail) lane

theorem encodeDigest_injective : Function.Injective encodeDigest := by
  intro left right equal
  apply Nightstream.Protocol.NebulaV2.Digest.Value.ext
  exact encodeDigestAux_injective laneCount equal

def modeValue : TerminationMode → Nat
  | .returned => 0
  | .trapped => 1

theorem modeValue_injective : Function.Injective modeValue := by
  intro left right equal
  cases left <;> cases right <;> simp_all [modeValue]

def resultBlockWidths : List Nat :=
  [18, 2293, 1, 32, 1, 32, 32, 256]

/-- Field blocks in the exact order of `SPEC.md` section 17. -/
def blocks (image : ProductionResultImage) : List (List Nat) :=
  [ encodeWord 18 image.realApplicationRowCount
  , WasmStateCodec.encode image.finalApplicationState
  , encodeWord 1 (modeValue image.terminationMode)
  , encodeWord 32 image.exitCode
  , encodeWord 1 image.outputPresent.toNat
  , encodeWord 32 image.outputValueLow
  , encodeWord 32 image.outputValueHigh
  , encodeDigest image.finalMemoryRoot
  ]

def encode (image : ProductionResultImage) : List Nat :=
  (blocks image).flatten

theorem blocks_lengths (image : ProductionResultImage) :
    (blocks image).map List.length = resultBlockWidths := by
  simp [blocks, resultBlockWidths, encodeWord_length,
    WasmStateCodec.encode_exact_length, encodeDigest_exact_length]

theorem encode_length (image : ProductionResultImage) :
    (encode image).length = 2665 := by
  simp [encode, List.length_flatten, blocks_lengths, resultBlockWidths]

theorem encode_binary
    (image : ProductionResultImage) (digit : Nat)
    (member : digit ∈ encode image) :
    digit < 2 := by
  rcases List.mem_flatten.mp member with ⟨block, blockMember, digitMember⟩
  simp only [blocks, List.mem_cons, List.not_mem_nil, or_false] at blockMember
  rcases blockMember with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · exact encodeWord_binary _ _ _ digitMember
  · exact WasmStateCodec.encode_binary _ _ digitMember
  · exact encodeWord_binary _ _ _ digitMember
  · exact encodeWord_binary _ _ _ digitMember
  · exact encodeWord_binary _ _ _ digitMember
  · exact encodeWord_binary _ _ _ digitMember
  · exact encodeWord_binary _ _ _ digitMember
  · exact encodeDigest_binary _ _ digitMember

theorem flatten_injective_of_lengths
    {Element : Type}
    {left right : List (List Element)} {widths : List Nat}
    (leftLengths : left.map List.length = widths)
    (rightLengths : right.map List.length = widths)
    (flattened : left.flatten = right.flatten) :
    left = right := by
  induction widths generalizing left right with
  | nil =>
      have leftEmpty : left = [] := by
        simpa using congrArg List.length leftLengths
      have rightEmpty : right = [] := by
        simpa using congrArg List.length rightLengths
      rw [leftEmpty, rightEmpty]
  | cons width widths inductionHypothesis =>
      cases left with
      | nil => simp at leftLengths
      | cons leftHead leftTail =>
          cases right with
          | nil => simp at rightLengths
          | cons rightHead rightTail =>
              simp only [List.map_cons, List.cons.injEq] at leftLengths rightLengths
              have headEqual : leftHead = rightHead := by
                have selected := congrArg (List.take width) flattened
                simpa [leftLengths.1, rightLengths.1] using selected
              have tailFlattened :
                  leftTail.flatten = rightTail.flatten := by
                have selected := congrArg (List.drop width) flattened
                simpa [leftLengths.1, rightLengths.1] using selected
              have tailEqual := inductionHypothesis
                leftLengths.2 rightLengths.2 tailFlattened
              rw [headEqual, tailEqual]

theorem blocks_equal_of_encode_equal
    {left right : ProductionResultImage}
    (equal : encode left = encode right) :
    blocks left = blocks right :=
  flatten_injective_of_lengths
    (blocks_lengths left) (blocks_lengths right) equal

private theorem component_words_equal
    {left right : ProductionResultImage}
    (equal : blocks left = blocks right) :
    encodeWord 18 left.realApplicationRowCount =
        encodeWord 18 right.realApplicationRowCount ∧
      WasmStateCodec.encode left.finalApplicationState =
        WasmStateCodec.encode right.finalApplicationState ∧
      encodeWord 1 (modeValue left.terminationMode) =
        encodeWord 1 (modeValue right.terminationMode) ∧
      encodeWord 32 left.exitCode = encodeWord 32 right.exitCode ∧
      encodeWord 1 left.outputPresent.toNat =
        encodeWord 1 right.outputPresent.toNat ∧
      encodeWord 32 left.outputValueLow =
        encodeWord 32 right.outputValueLow ∧
      encodeWord 32 left.outputValueHigh =
        encodeWord 32 right.outputValueHigh ∧
      encodeDigest left.finalMemoryRoot = encodeDigest right.finalMemoryRoot := by
  simpa [blocks] using equal

private theorem boolToNat_injective : Function.Injective Bool.toNat := by
  intro left right equal
  cases left <;> cases right <;> simp_all

private theorem exitCode_lt
    {image : ProductionResultImage}
    {result : ExecutionResult AppStateVector
      Nightstream.Protocol.NebulaV2.Digest.Value}
    (decoded : image.Decodes result) :
    image.exitCode < 2 ^ 32 := by
  have status := decoded.mode_exit_and_flags_exact
  rcases status.2.2 with returned | trapped
  · rw [returned.2.2]
    decide
  · have := trapped.2.2.2
    norm_num at this ⊢
    omega

private theorem outputLow_lt
    {image : ProductionResultImage}
    {result : ExecutionResult AppStateVector
      Nightstream.Protocol.NebulaV2.Digest.Value}
    (decoded : image.Decodes result) :
    image.outputValueLow < 2 ^ 32 := by
  have outputEqual := decoded.output_fields_equal_state
  have valid := decoded.terminal.valid.outputCanonical.1
  calc
    image.outputValueLow = result.finalApplicationState.output.low :=
      outputEqual.2.1
    _ < 2 ^ 32 := by
      simpa [OutputState.Canonical, u32Limit] using valid

private theorem outputHigh_lt
    {image : ProductionResultImage}
    {result : ExecutionResult AppStateVector
      Nightstream.Protocol.NebulaV2.Digest.Value}
    (decoded : image.Decodes result) :
    image.outputValueHigh < 2 ^ 32 := by
  have outputEqual := decoded.output_fields_equal_state
  have valid := decoded.terminal.valid.outputCanonical.2.1
  calc
    image.outputValueHigh = result.finalApplicationState.output.high :=
      outputEqual.2.2
    _ < 2 ^ 32 := by
      simpa [OutputState.Canonical, u32Limit] using valid

/-- The complete flattened production result has one accepted typed preimage.
No digest or state component can be omitted or reduced modulo Goldilocks. -/
theorem encode_injective_of_decodes
    {left right : ProductionResultImage}
    {leftResult rightResult : ExecutionResult AppStateVector
      Nightstream.Protocol.NebulaV2.Digest.Value}
    (leftDecoded : left.Decodes leftResult)
    (rightDecoded : right.Decodes rightResult)
    (equal : encode left = encode right) :
    left = right := by
  have words := component_words_equal (blocks_equal_of_encode_equal equal)
  have leftRowBound : left.realApplicationRowCount < 2 ^ 18 := by
    rw [leftDecoded.exactImage]
    simpa [ResultImage.ofResult, realApplicationRowLimit] using
      leftDecoded.realRowCountBound
  have rightRowBound : right.realApplicationRowCount < 2 ^ 18 := by
    rw [rightDecoded.exactImage]
    simpa [ResultImage.ofResult, realApplicationRowLimit] using
      rightDecoded.realRowCountBound
  apply ResultImage.ext
  · exact encodeWord_injective_of_bound
      leftRowBound rightRowBound words.1
  · exact WasmStateCodec.encode_injective_on_canonical
      leftDecoded.final_state_canonical rightDecoded.final_state_canonical
      words.2.1
  · apply modeValue_injective
    exact encodeWord_injective_of_bound (by cases left.terminationMode <;> decide)
      (by cases right.terminationMode <;> decide) words.2.2.1
  · exact encodeWord_injective_of_bound
      (exitCode_lt leftDecoded) (exitCode_lt rightDecoded) words.2.2.2.1
  · apply boolToNat_injective
    exact encodeWord_injective_of_bound (Bool.toNat_lt _)
      (Bool.toNat_lt _) words.2.2.2.2.1
  · exact encodeWord_injective_of_bound
      (outputLow_lt leftDecoded) (outputLow_lt rightDecoded)
      words.2.2.2.2.2.1
  · exact encodeWord_injective_of_bound
      (outputHigh_lt leftDecoded) (outputHigh_lt rightDecoded)
      words.2.2.2.2.2.2.1
  · exact encodeDigest_injective words.2.2.2.2.2.2.2

end Nightstream.Implementation.NebulaV2.WasmResultCodec
