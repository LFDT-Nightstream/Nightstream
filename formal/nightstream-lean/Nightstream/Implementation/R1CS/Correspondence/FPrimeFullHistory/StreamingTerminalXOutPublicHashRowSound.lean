import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutPublicHash
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField

/-!
Contract: the exact recursive-terminal XOut public-hash source rows force each
verifier-owned public 64-bit word to decode to its matching Poseidon2 lane.

Owns the structural adapter from one generated `PublicWord` to the shared
canonical-u64 `FieldCodec.CallSite`, extraction of its typed public word, and
composition with the self-owned Poseidon2 trace theorem.

Does not own final selective-row transport, terminal lifecycle composition,
or collision resistance.

Assurance tier: artifact-checked for the Nightstream b2/k16 terminal profile.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPublicHashRowSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.CanonicalU64
open Nightstream.Implementation.R1CS.Canonical.GoldilocksField
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.Nebula
open Nightstream.Protocol.Nebula
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutPublicHash.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutPublicHash

private theorem rowsIncluded_append_left (left right : List Row) :
    rowsIncluded left (left ++ right) = true := by
  unfold rowsIncluded
  apply List.all_eq_true.mpr
  intro row member
  exact decide_eq_true (List.mem_append_left right member)

private theorem rowsIncluded_append_right (left right : List Row) :
    rowsIncluded right (left ++ right) = true := by
  unfold rowsIncluded
  apply List.all_eq_true.mpr
  intro row member
  exact decide_eq_true (List.mem_append_right left member)

private theorem getD_append_right_exact
    {alpha : Type} (left right : List alpha) (index : Nat) (default : alpha) :
    (left ++ right).getD (left.length + index) default =
      right.getD index default := by
  rw [List.getD_eq_getElem?_getD,
    List.getElem?_append_right (by omega)]
  simp only [Nat.add_sub_cancel_left]
  rfl

private theorem mapped_bit_column
    (word : PublicWord) (valid : word.Valid)
    (index : Nat) (bounded : index < CanonicalFieldBits.bitCount) :
    Relabel.column word.columnMap (bitCol index) =
      word.canonicalBitColumns.getD index 0 := by
  have canonicalBound : index < word.canonicalBitColumns.length := by
    rw [valid.1]
    simpa [CanonicalFieldBits.bitCount] using bounded
  unfold Relabel.column PublicWord.columnMap bitCol
  rw [List.append_assoc]
  have indexShape :
      index + 2 = [0, word.fieldColumn].length + index := by
    simp [Nat.add_comm]
  rw [indexShape, getD_append_right_exact]
  exact List.getD_append _ _ _ _ canonicalBound

private theorem callSite_linkPairs_eq
    (word : PublicWord) (valid : word.Valid) :
    (List.range CanonicalFieldBits.bitCount).map (fun index =>
      (word.publicBitColumns.getD index 0,
        Relabel.column word.columnMap (bitCol index))) = word.linkPairs := by
  unfold PublicWord.linkPairs
  rw [show CanonicalFieldBits.bitCount = 64 by rfl]
  apply List.map_congr_left
  intro index member
  apply Prod.ext
  · rfl
  · exact mapped_bit_column word valid index (List.mem_range.mp member)

private def publicWordCallSite (word : PublicWord) (valid : word.Valid) :
    FieldCodec.CallSite word.rows where
  columnMap := word.columnMap
  rawColumns := word.publicBitColumns
  rawColumnsLength := by
    simpa [CanonicalFieldBits.bitCount] using valid.2.1
  mapsConstantOne := by
    rfl
  canonicalRowsIncluded := by
    change rowsIncluded word.canonicalProgram word.rows = true
    unfold PublicWord.rows
    exact rowsIncluded_append_left _ _
  linkRowsIncluded := by
    rw [callSite_linkPairs_eq word valid]
    change rowsIncluded word.linkProgram word.rows = true
    unfold PublicWord.rows
    exact rowsIncluded_append_right _ _

private theorem word_rows_satisfied
    {assignment : Nat → Nat}
    (satisfied : rawArtifact.Satisfied assignment)
    {word : PublicWord} (member : word ∈ publicWords) :
    Satisfies word.rows assignment := by
  unfold RawArtifact.Satisfied at satisfied
  intro row rowMember
  apply satisfied row
  unfold RawArtifact.rows
  apply List.mem_append_right
  exact List.mem_flatMap.mpr ⟨word, member, rowMember⟩

private theorem trace_rows_satisfied
    {assignment : Nat → Nat}
    (satisfied : rawArtifact.Satisfied assignment) :
    Satisfies trace.rows assignment := by
  unfold RawArtifact.Satisfied at satisfied
  intro row rowMember
  apply satisfied row
  unfold RawArtifact.rows
  exact List.mem_append_left _ rowMember

/-- The typed public word is derived from the assignment and exact emitted
rows. Its value list is exactly the verifier-owned public-bit placement. -/
def publicWordOfRows
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : rawArtifact.Satisfied assignment)
    (word : PublicWord) (member : word ∈ publicWords)
    (valid : word.Valid) : CanonicalFieldBits.Word :=
  (publicWordCallSite word valid).wordOfRows goldilocks_euclidPrime
    canonical one (word_rows_satisfied satisfied member)

@[simp] theorem publicWordOfRows_val
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : rawArtifact.Satisfied assignment)
    (word : PublicWord) (member : word ∈ publicWords)
    (valid : word.Valid) :
    (publicWordOfRows assignment canonical one satisfied word member valid).val =
      (List.range CanonicalFieldBits.bitCount).map (fun index =>
        assignment (word.publicBitColumns.getD index 0)) := rfl

private theorem mapped_field_column (word : PublicWord) :
    Relabel.column word.columnMap varCol = word.fieldColumn := by
  rfl

theorem publicWord_field_sound
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : rawArtifact.Satisfied assignment)
    (word : PublicWord) (member : word ∈ publicWords)
    (valid : word.Valid) :
    CanonicalFieldBits.decode
        (publicWordOfRows assignment canonical one satisfied word member valid) =
      assignment word.fieldColumn := by
  let derivedWord : CanonicalFieldBits.Word :=
    publicWordOfRows assignment canonical one satisfied word member valid
  have placed :
      (publicWordCallSite word valid).Places assignment derivedWord := by
    rfl
  rcases FieldCodec.CallSite.sound goldilocks_euclidPrime
      (publicWordCallSite word valid) canonical one
      (word_rows_satisfied satisfied member) placed with
    ⟨value, decoded, valueExact⟩
  have decodedValue :
      CanonicalFieldBits.Canonical derivedWord ∧
        value.val = CanonicalFieldBits.decode derivedWord :=
    (FieldCodec.nativeDecode_some_iff derivedWord value).mp decoded
  change value.val =
    assignment (Relabel.column word.columnMap varCol) at valueExact
  rw [mapped_field_column] at valueExact
  change CanonicalFieldBits.decode derivedWord = assignment word.fieldColumn
  exact decodedValue.2.symm.trans valueExact

/-- One generated word and its generated lane binding derive the pure
Poseidon2 hash value from the exact ordered 32-field XOut input. -/
theorem publicWord_hash_sound
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : rawArtifact.Satisfied assignment)
    {lane : Nat} (laneLt : lane < 4)
    {word : PublicWord} (member : word ∈ publicWords)
    (binding : word.fieldColumn = trace.outputColumns.getD lane 0)
    (valid : word.Valid) :
    CanonicalFieldBits.decode
        (publicWordOfRows assignment canonical one satisfied word member valid) =
      runValueRounds trace.rounds (trace.inputColumns.map assignment)
        (fun _ => 0) lane := by
  calc
    _ = assignment word.fieldColumn :=
      publicWord_field_sound assignment canonical one satisfied word member valid
    _ = assignment (trace.outputColumns.getD lane 0) :=
      congrArg assignment binding
    _ = runValueRounds trace.rounds (trace.inputColumns.map assignment)
        (fun _ => 0) lane :=
      ownedTrace_values_sound trace_owned_valid canonical one
        (trace_rows_satisfied satisfied) lane laneLt

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPublicHashRowSound
