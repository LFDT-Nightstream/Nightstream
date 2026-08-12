import Nightstream.Implementation.R1CS.Correspondence.CanonicalU64.CanonicalU64Complete
import Nightstream.Implementation.R1CS.Core.EqualityPins
import Nightstream.Protocol.NebulaV2.CanonicalFieldBits
import Mathlib.Data.List.GetD

/-!
Contract: exact native and generated-row refinement for one V2 Goldilocks
public-input limb.

Assurance tier: implementation model.

Owns the fail-closed native decoder, the exact placement of 64 little-endian
public bits, inclusion of the generated canonical-u64 block and its bit-link
rows, extraction of the unique canonical integer from satisfying rows, and
local compiler completeness for every accepted word.

Does not own the final V2 generated program, Rust parser extraction, extension
field arithmetic, proof-system soundness, or cryptographic assumptions. A
call-site certificate contains row inclusion and input placement only; it does
not contain canonicality or the decoded equality that this file proves.

Emits constraints: no. It proves properties of exact generated constraints.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.FieldCodec

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.CanonicalU64
open Nightstream.Implementation.R1CS.CanonicalU64Complete
open Nightstream.Protocol.NebulaV2

namespace Bits

/-- Little-endian digit evaluation and the R1CS artifact use the same weighted
fold. The proof is valid for every length and does not use a 64-bit example. -/
theorem ofDigits_range_map (base count : Nat) (digit : Nat → Nat) :
    Nat.ofDigits base ((List.range count).map digit) =
      (List.range count).foldl
        (fun value index => value + base ^ index * digit index) 0 := by
  induction count with
  | zero => simp
  | succ count inductionHypothesis =>
      simp [List.range_succ, Nat.ofDigits_append, inductionHypothesis]

private theorem exact_list_eq_range_getD
    (values : List Nat) (default : Nat) :
    values =
      (List.range values.length).map (fun index => values.getD index default) := by
  apply List.ext_get
  · simp
  · intro index leftBound rightBound
    simp [List.get_eq_getElem, List.getD_eq_getElem?_getD, leftBound]

/-- If a protocol word is placed at the artifact bit columns, its independent
protocol decoder equals the integer reconstructed by the generated gadget. -/
theorem decode_eq_bitsValue_of_link
    (word : CanonicalFieldBits.Word) (assignment : Nat → Nat)
    (linked : ∀ index, index < CanonicalFieldBits.bitCount →
      word.val.getD index 0 = assignment (bitCol index)) :
    CanonicalFieldBits.decode word = bitsValue assignment := by
  have exactDigits :
      word.val =
        (List.range CanonicalFieldBits.bitCount).map
          (fun index => word.val.getD index 0) := by
    simpa [word.property.1] using
      exact_list_eq_range_getD word.val 0
  have linkedDigits :
      (List.range CanonicalFieldBits.bitCount).map
          (fun index => word.val.getD index 0) =
        (List.range CanonicalFieldBits.bitCount).map
          (fun index => assignment (bitCol index)) := by
    apply List.map_congr_left
    intro index member
    exact linked index (List.mem_range.mp member)
  calc
    CanonicalFieldBits.decode word = Nat.ofDigits 2 word.val := rfl
    _ = Nat.ofDigits 2
        ((List.range CanonicalFieldBits.bitCount).map
          (fun index => word.val.getD index 0)) := by rw [← exactDigits]
    _ = Nat.ofDigits 2
        ((List.range CanonicalFieldBits.bitCount).map
          (fun index => assignment (bitCol index))) := by rw [linkedDigits]
    _ = bitsValue assignment := by
      rw [ofDigits_range_map]
      rfl

end Bits

/-- Executable native language for one authority-bearing field limb. A raw
64-bit word is accepted only when its integer is strictly below Goldilocks. -/
def nativeDecode (word : CanonicalFieldBits.Word) :
    Option ShiftedTernary41V1.CanonicalGoldilocks :=
  if canonical :
      CanonicalFieldBits.decode word < ShiftedTernary41V1.modulus then
    some ⟨CanonicalFieldBits.decode word, canonical⟩
  else
    none

theorem nativeDecode_some_iff
    (word : CanonicalFieldBits.Word)
    (value : ShiftedTernary41V1.CanonicalGoldilocks) :
    nativeDecode word = some value ↔
      CanonicalFieldBits.Canonical word ∧
        value.val = CanonicalFieldBits.decode word := by
  unfold nativeDecode
  by_cases canonical :
      CanonicalFieldBits.decode word < ShiftedTernary41V1.modulus
  · rw [dif_pos canonical]
    constructor
    · intro equal
      have valueEqual :
          (⟨CanonicalFieldBits.decode word, canonical⟩ :
            ShiftedTernary41V1.CanonicalGoldilocks) = value :=
        Option.some.inj equal
      exact ⟨canonical, congrArg Subtype.val valueEqual.symm⟩
    · rintro ⟨_, value⟩
      apply congrArg some
      apply Subtype.ext
      exact value.symm
  · rw [dif_neg canonical]
    simp [CanonicalFieldBits.Canonical, canonical]

theorem nativeDecode_unique
    {word : CanonicalFieldBits.Word}
    {left right : ShiftedTernary41V1.CanonicalGoldilocks}
    (leftDecoded : nativeDecode word = some left)
    (rightDecoded : nativeDecode word = some right) :
    left = right := by
  rw [leftDecoded] at rightDecoded
  exact Option.some.inj rightDecoded

/-- Exact call-site rows for a 64-bit public word inside a larger generated
program. `rawColumns` are the authority-bearing public bits. `columnMap` maps
the independent canonical-u64 artifact into the program. -/
structure CallSite (programRows : List Row) where
  columnMap : List Nat
  rawColumns : List Nat
  rawColumnsLength : rawColumns.length = CanonicalFieldBits.bitCount
  mapsConstantOne : Relabel.column columnMap 0 = 0
  canonicalRowsIncluded :
    rowsIncluded (rows.map (Relabel.row columnMap)) programRows = true
  linkRowsIncluded :
    rowsIncluded
      (EqualityPins.rows
        ((List.range CanonicalFieldBits.bitCount).map fun index =>
          (rawColumns.getD index 0,
            Relabel.column columnMap (bitCol index))))
      programRows = true

namespace CallSite

def linkPairs {programRows : List Row} (site : CallSite programRows) :
    List (Nat × Nat) :=
  (List.range CanonicalFieldBits.bitCount).map fun index =>
    (site.rawColumns.getD index 0,
      Relabel.column site.columnMap (bitCol index))

def rawDigits {programRows : List Row} (site : CallSite programRows)
    (assignment : Nat → Nat) : List Nat :=
  (List.range CanonicalFieldBits.bitCount).map fun index =>
    assignment (site.rawColumns.getD index 0)

theorem rawDigits_length
    {programRows : List Row} (site : CallSite programRows)
    (assignment : Nat → Nat) :
    (site.rawDigits assignment).length = CanonicalFieldBits.bitCount := by
  simp [rawDigits]

/-- The exact canonical-u64 and link rows derive bitness of every raw public
column. -/
theorem rawDigits_binary
    (prime : EuclidPrime goldilocksP)
    {programRows : List Row} (site : CallSite programRows)
    {assignment : Nat → Nat}
    (canonicalAssignment : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (satisfies : Satisfies programRows assignment)
    (digit : Nat) (member : digit ∈ site.rawDigits assignment) :
    digit < 2 := by
  rcases List.mem_map.mp member with ⟨index, indexMember, rfl⟩
  have indexBound : index < CanonicalFieldBits.bitCount :=
    List.mem_range.mp indexMember
  have canonicalRows :
      Satisfies rows (Relabel.assignment site.columnMap assignment) :=
    Relabel.satisfies_of_included site.canonicalRowsIncluded satisfies
  have pulledCanonical :
      ∀ column,
        Relabel.assignment site.columnMap assignment column < goldilocksP :=
    Relabel.canonical canonicalAssignment
  have pulledOne :
      Relabel.assignment site.columnMap assignment 0 = 1 :=
    Relabel.constantOne site.mapsConstantOne constantOne
  have linkEqualities :
      ∀ pair ∈ site.linkPairs,
        assignment pair.1 = assignment pair.2 := by
    apply EqualityPins.sound site.linkRowsIncluded canonicalAssignment
      constantOne satisfies
  have pairMember :
      (site.rawColumns.getD index 0,
          Relabel.column site.columnMap (bitCol index)) ∈
        site.linkPairs := by
    exact List.mem_map.mpr ⟨index, indexMember, rfl⟩
  rw [linkEqualities _ pairMember]
  exact canonicalU64_bit_lt_two prime pulledCanonical pulledOne canonicalRows
    index (by simpa [CanonicalFieldBits.bitCount] using indexBound)

/-- The unique typed raw word derived from the assignment and exact rows. -/
def wordOfRows
    (prime : EuclidPrime goldilocksP)
    {programRows : List Row} (site : CallSite programRows)
    {assignment : Nat → Nat}
    (canonicalAssignment : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (satisfies : Satisfies programRows assignment) :
    CanonicalFieldBits.Word :=
  ⟨site.rawDigits assignment, site.rawDigits_length assignment,
    site.rawDigits_binary prime canonicalAssignment constantOne satisfies⟩

@[simp] theorem wordOfRows_val
    (prime : EuclidPrime goldilocksP)
    {programRows : List Row} (site : CallSite programRows)
    {assignment : Nat → Nat}
    (canonicalAssignment : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (satisfies : Satisfies programRows assignment) :
    (site.wordOfRows prime canonicalAssignment constantOne satisfies).val =
      site.rawDigits assignment := rfl

/-- Parser-to-assignment ownership boundary. It states only where the raw
bits were placed. It contains no decoded value or acceptance conclusion. -/
def Places {programRows : List Row} (site : CallSite programRows)
    (assignment : Nat → Nat) (word : CanonicalFieldBits.Word) : Prop :=
  word.val = site.rawDigits assignment

/-- Satisfying the exact generated block and exact public-bit links forces the
native decoder to accept one value equal to the circuit field wire. -/
theorem sound
    (prime : EuclidPrime goldilocksP)
    {programRows : List Row} (site : CallSite programRows)
    {assignment : Nat → Nat} {word : CanonicalFieldBits.Word}
    (canonicalAssignment : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (satisfies : Satisfies programRows assignment)
    (placed : site.Places assignment word) :
    ∃ value,
      nativeDecode word = some value ∧
        value.val =
          assignment (Relabel.column site.columnMap varCol) := by
  have canonicalRows :
      Satisfies rows (Relabel.assignment site.columnMap assignment) :=
    Relabel.satisfies_of_included site.canonicalRowsIncluded satisfies
  have pulledCanonical :
      ∀ column,
        Relabel.assignment site.columnMap assignment column < goldilocksP :=
    Relabel.canonical canonicalAssignment
  have pulledOne :
      Relabel.assignment site.columnMap assignment 0 = 1 :=
    Relabel.constantOne site.mapsConstantOne constantOne
  have artifact :=
    canonicalU64_sound prime pulledCanonical pulledOne canonicalRows
  have linkEqualities :
      ∀ pair ∈ site.linkPairs,
        assignment pair.1 = assignment pair.2 := by
    apply EqualityPins.sound site.linkRowsIncluded canonicalAssignment
      constantOne satisfies
  have linked :
      ∀ index, index < CanonicalFieldBits.bitCount →
        word.val.getD index 0 =
          Relabel.assignment site.columnMap assignment (bitCol index) := by
    intro index bounded
    have pairMember :
        (site.rawColumns.getD index 0,
            Relabel.column site.columnMap (bitCol index)) ∈
          site.linkPairs := by
      exact List.mem_map.mpr
        ⟨index, List.mem_range.mpr bounded, rfl⟩
    have equal := linkEqualities _ pairMember
    have placedAt :
        word.val.getD index 0 =
          assignment (site.rawColumns.getD index 0) := by
      have listsEqual :
          word.val =
            (List.range CanonicalFieldBits.bitCount).map fun position =>
              assignment (site.rawColumns.getD position 0) := placed
      have wordBound : index < word.val.length := by
        simpa [word.property.1] using bounded
      simpa [CallSite.rawDigits, wordBound, bounded] using
        congrArg (fun values => values.getD index 0) listsEqual
    exact placedAt.trans equal
  have decodedEqualsBits :
      CanonicalFieldBits.decode word =
        bitsValue (Relabel.assignment site.columnMap assignment) :=
    Bits.decode_eq_bitsValue_of_link word _ linked
  have canonicalWord : CanonicalFieldBits.Canonical word := by
    unfold CanonicalFieldBits.Canonical
    rw [decodedEqualsBits]
    simpa [ShiftedTernary41V1.modulus, goldilocksP] using artifact.2
  let value : ShiftedTernary41V1.CanonicalGoldilocks :=
    ⟨CanonicalFieldBits.decode word, canonicalWord⟩
  refine ⟨value, ?_, ?_⟩
  · change CanonicalFieldBits.decode word <
      ShiftedTernary41V1.modulus at canonicalWord
    simp [nativeDecode, canonicalWord, value]
  · change CanonicalFieldBits.decode word = _
    exact decodedEqualsBits.trans artifact.1.symm

end CallSite

/-- Deterministic source bits used by the native canonical-u64 compiler. -/
def sourceOfWord (word : CanonicalFieldBits.Word) : Source where
  bit := fun index => decide (word.val.getD index 0 = 1)

theorem sourceOfWord_bitValue
    (word : CanonicalFieldBits.Word) {index : Nat}
    (bounded : index < CanonicalFieldBits.bitCount) :
    bitValue (sourceOfWord word) index = word.val.getD index 0 := by
  have wordBound : index < word.val.length := by
    simpa [word.property.1] using bounded
  have digitBound : word.val.getD index 0 < 2 := by
    rw [List.getD_eq_getElem word.val 0 wordBound]
    exact word.property.2 _ (List.get_mem word.val ⟨index, wordBound⟩)
  have digitCases :
      word.val.getD index 0 = 0 ∨ word.val.getD index 0 = 1 := by
    omega
  rcases digitCases with digitZero | digitOne
  · change (decide (word.val.getD index 0 = 1)).toNat =
      word.val.getD index 0
    rw [digitZero]
    decide
  · change (decide (word.val.getD index 0 = 1)).toNat =
      word.val.getD index 0
    rw [digitOne]
    decide

/-- Every native-accepted protocol word has an honest assignment satisfying
all 69 exact generated canonical-u64 rows. The same assignment carries the
same 64 source bits and the same decoded integer. -/
theorem local_complete
    (field : FieldInverse) (word : CanonicalFieldBits.Word)
    (accepted : CanonicalFieldBits.Canonical word) :
    let assignment := interpret field (sourceOfWord word)
    Satisfies rows assignment ∧
      (∀ column, assignment column < goldilocksP) ∧
      CanonicalFieldBits.decode word = assignment varCol ∧
      (∀ index, index < CanonicalFieldBits.bitCount →
        word.val.getD index 0 = assignment (bitCol index)) := by
  let assignment := interpret field (sourceOfWord word)
  have linked :
      ∀ index, index < CanonicalFieldBits.bitCount →
        word.val.getD index 0 = assignment (bitCol index) := by
    intro index bounded
    rw [show assignment (bitCol index) =
        bitValue (sourceOfWord word) index by
      exact interpret_bit field (sourceOfWord word)
        (by simpa [CanonicalFieldBits.bitCount] using bounded)]
    exact (sourceOfWord_bitValue word bounded).symm
  have decodedEqualsBits :
      CanonicalFieldBits.decode word = bitsValue assignment :=
    Bits.decode_eq_bitsValue_of_link word assignment linked
  have decodedEqualsWordValue :
      CanonicalFieldBits.decode word = wordValue (sourceOfWord word) :=
    decodedEqualsBits.trans (bitsValue_interpret field _)
  have sourceCanonical : wordValue (sourceOfWord word) < goldilocksP := by
    rw [← decodedEqualsWordValue]
    simpa [CanonicalFieldBits.Canonical, ShiftedTernary41V1.modulus,
      goldilocksP] using accepted
  refine ⟨complete field (sourceOfWord word) sourceCanonical,
    interpret_canonical field (sourceOfWord word) sourceCanonical, ?_, linked⟩
  rw [interpret_var]
  exact decodedEqualsWordValue

/-- The exact modulo-alias regression: native decoding accepts canonical zero
and rejects the distinct 64-bit encoding of the modulus. -/
theorem rejects_zero_modulus_alias :
    (∃ value, nativeDecode CanonicalFieldBits.zeroWord = some value ∧
        value.val = 0) ∧
      nativeDecode CanonicalFieldBits.modulusWord = none := by
  constructor
  · refine ⟨CanonicalFieldBits.zero, ?_, rfl⟩
    have canonical : CanonicalFieldBits.Canonical
        CanonicalFieldBits.zeroWord :=
      CanonicalFieldBits.encode_is_canonical CanonicalFieldBits.zero
    change CanonicalFieldBits.decode CanonicalFieldBits.zeroWord <
      ShiftedTernary41V1.modulus at canonical
    unfold nativeDecode
    rw [dif_pos canonical]
    apply congrArg some
    apply Subtype.ext
    exact CanonicalFieldBits.decode_zeroWord
  · have notCanonical : ¬ CanonicalFieldBits.Canonical
        CanonicalFieldBits.modulusWord :=
      CanonicalFieldBits.modulusWord_not_canonical
    change ¬ CanonicalFieldBits.decode CanonicalFieldBits.modulusWord <
      ShiftedTernary41V1.modulus at notCanonical
    unfold nativeDecode
    rw [dif_neg notCanonical]

end Nightstream.Implementation.NebulaV2.FieldCodec
