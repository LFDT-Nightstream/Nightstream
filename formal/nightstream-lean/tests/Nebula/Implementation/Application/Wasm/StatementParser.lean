import Nightstream.Implementation.Nebula.Application.Wasm.StatementParser

set_option autoImplicit false

namespace tests.NebulaWasmStatementParser

open Nightstream.Implementation.Nebula.WasmStatementBytes
open Nightstream.Implementation.Nebula.WasmStatementParser
open Nightstream.Protocol.Nebula.WasmPublicStatementEncoding

example (image : PublicImage) :
    parse (encode image) = some (statementWord image) :=
  parse_encode image

example : parse [] = none := by
  apply rejects_wrong_length
  decide

def nonByteSection : List Nat :=
  List.ofFn fun index : Fin statementByteCount =>
    if index.val = 0 then 256 else 0

example : parse nonByteSection = none := by
  have lengthExact : nonByteSection.length = statementByteCount := by
    simp [nonByteSection]
  refine rejects_non_byte nonByteSection lengthExact ?_
  · intro range
    have every := List.all_eq_true.mp range
    have member : 256 ∈ nonByteSection := by
      apply List.mem_ofFn.mpr
      exact ⟨⟨0, by decide⟩, rfl⟩
    have impossible := every 256 member
    simp at impossible

def highPaddingSection : List Nat :=
  List.ofFn fun index : Fin statementByteCount =>
    if index.val = statementByteCount - 1 then 16 else 0

example : parse highPaddingSection = none := by
  have lengthExact : highPaddingSection.length = statementByteCount := by
    simp [highPaddingSection]
  refine rejects_nonzero_high_padding highPaddingSection lengthExact ?_ ?_
  · unfold bytesInRange
    rw [List.all_eq_true]
    intro byte member
    rcases List.mem_ofFn.mp member with ⟨index, rfl⟩
    split <;> simp
  · have finalValue :
        (words highPaddingSection lengthExact
          ⟨statementByteCount - 1, by decide⟩).toNat = 16 := by
      simp only [words, highPaddingSection, List.get_ofFn]
      simp [statementByteCount, byteBitCount]
    omega

end tests.NebulaWasmStatementParser
