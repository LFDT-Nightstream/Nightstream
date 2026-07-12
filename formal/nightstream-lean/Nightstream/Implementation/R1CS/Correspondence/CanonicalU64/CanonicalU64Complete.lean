import Nightstream.Implementation.R1CS.Correspondence.CanonicalU64.CanonicalU64Sound
import Nightstream.Implementation.R1CS.Core.Relabel

/-!
Contract: compiler completeness for the exact canonical-u64 artifact.

The production compiler receives a canonical 64-bit word, allocates its bits,
and evaluates one Goldilocks inverse while constructing the canonicity gate.
This module models that process as a deterministic interpreter.  Its public
execution witness contains source bits and interpreter equality only; it never
contains `Satisfies`, `RowHolds`, or a decoded acceptance equation.

Goldilocks inversion is an explicit runtime primitive.  `FieldInverse` states
the ordinary total-field contract once, independently of any assignment or
artifact row.  It is the same mathematical boundary as Rust's `F::inverse`.
-/

namespace Nightstream.Implementation.R1CS.CanonicalU64Complete

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.CanonicalU64

set_option maxRecDepth 65536
set_option maxHeartbeats 5000000

/-- Semantics of the field-inverse primitive used by the Rust witness
generator.  The law is global; it cannot be tailored to a circuit assignment. -/
structure FieldInverse where
  inverse : Nat → Nat
  canonical : ∀ value, inverse value < goldilocksP
  zero : inverse 0 = 0
  correct : ∀ value, value < goldilocksP → value ≠ 0 →
    value * inverse value % goldilocksP = 1

/-- Source-level 64-bit word.  Values outside the first 64 positions are
ignored, exactly as they are by a fixed-width Rust `u64`. -/
structure Source where
  bit : Nat → Bool

def bitValue (source : Source) (index : Nat) : Nat :=
  (source.bit index).toNat

def lowValue (source : Source) : Nat :=
  (List.range 32).foldl
    (fun value index => value + 2 ^ index * bitValue source index) 0

def highValue (source : Source) : Nat :=
  (List.range 32).foldl
    (fun value index => value + 2 ^ index * bitValue source (32 + index)) 0

/-- Integer denotation of the source word. -/
def wordValue (source : Source) : Nat :=
  lowValue source + 4294967296 * highValue source

private def oneCount (source : Source) : Nat :=
  (List.range 64).foldl
    (fun value index => value + bitValue source index) 0

def highMax : Nat := 4294967295

def highIsMax (source : Source) : Nat :=
  if highValue source = highMax then 1 else 0

theorem highIsMax_lt_modulus (source : Source) :
    highIsMax source < goldilocksP := by
  unfold highIsMax
  split <;> simp [goldilocksP]

/-- Canonical residue of `high - 0xffffffff`. -/
def highDifference (source : Source) : Nat :=
  (highValue source + (goldilocksP - highMax)) % goldilocksP

/-- Deterministic local witness constructed from a source word and the field
runtime.  Columns outside the isolated 68-column artifact are row-irrelevant
and use the constant-one value, matching `Relabel.column`'s finite-map
fallback. -/
def interpret (field : FieldInverse) (source : Source) : Nat → Nat :=
  fun column =>
    if column = 0 then 1
    else if column = varCol then wordValue source
    else if 2 ≤ column ∧ column < 66 then bitValue source (column - 2)
    else if column = 66 then highIsMax source
    else if column = 67 then field.inverse (highDifference source)
    else 1

private def bitRows : List Row :=
  (List.range 64).map fun index => bitRow (bitCol index)

private def lowTerms : List (Nat × Nat) :=
  (List.range 32).map fun index => (bitCol index, 2 ^ index)

private def highTerms : List (Nat × Nat) :=
  (List.range 32).map fun index => (bitCol (32 + index), 2 ^ index)

private def negativeBitTerms : List (Nat × Nat) :=
  (List.range 64).map fun index =>
    (bitCol index, goldilocksP - 2 ^ index)

private def recomposeRow : Row :=
  ⟨(varCol, 1) :: negativeBitTerms, [(0, 1)], []⟩

private def highDifferenceTerms : List (Nat × Nat) :=
  highTerms ++ [(0, goldilocksP - highMax)]

private def highDefinitionRow : Row :=
  ⟨[(66, 1)], highDifferenceTerms, []⟩

private def inverseRow : Row :=
  ⟨highDifferenceTerms, [(67, 1)],
    [(66, goldilocksP - 1), (0, 1)]⟩

private def canonicalityRow : Row :=
  ⟨[(66, 1)], lowTerms, []⟩

private theorem range32_shape : List.range 32 =
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
     31] := by decide

private theorem range64_shape : List.range 64 =
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
     31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
     46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
     61, 62, 63] := by decide

/-- The hand-readable compiler schedule is definitionally checked against all
69 generated rows.  Artifact drift therefore invalidates completeness. -/
theorem rows_shape :
    rows = bitRows ++
      [recomposeRow, bitRow 66, highDefinitionRow, inverseRow,
       canonicalityRow] := by
  native_decide

@[simp] theorem interpret_one (field : FieldInverse) (source : Source) :
    interpret field source 0 = 1 := by
  simp [interpret]

@[simp] theorem interpret_var (field : FieldInverse) (source : Source) :
    interpret field source varCol = wordValue source := by
  simp [interpret, varCol]

@[simp] theorem interpret_bit (field : FieldInverse) (source : Source)
    {index : Nat} (bounded : index < 64) :
    interpret field source (bitCol index) = bitValue source index := by
  have notVar : index + 2 ≠ 1 := by omega
  have lower : 2 ≤ index + 2 := by omega
  have upper : index + 2 < 66 := by omega
  simp [interpret, bitCol, varCol, notVar, lower, upper]

@[simp] theorem interpret_highFlag (field : FieldInverse) (source : Source) :
    interpret field source 66 = highIsMax source := by
  simp [interpret, varCol]

@[simp] theorem interpret_inverse (field : FieldInverse) (source : Source) :
    interpret field source 67 = field.inverse (highDifference source) := by
  simp [interpret, varCol]

theorem bitValue_le_one (source : Source) (index : Nat) :
    bitValue source index ≤ 1 := by
  cases h : source.bit index <;> simp [bitValue, h]

theorem bitValue_lt_modulus (source : Source) (index : Nat) :
    bitValue source index < goldilocksP := by
  have := bitValue_le_one source index
  simp only [goldilocksP]
  omega

theorem highValue_le_max (source : Source) :
    highValue source ≤ highMax := by
  have b0 := bitValue_le_one source 32
  have b1 := bitValue_le_one source 33
  have b2 := bitValue_le_one source 34
  have b3 := bitValue_le_one source 35
  have b4 := bitValue_le_one source 36
  have b5 := bitValue_le_one source 37
  have b6 := bitValue_le_one source 38
  have b7 := bitValue_le_one source 39
  have b8 := bitValue_le_one source 40
  have b9 := bitValue_le_one source 41
  have b10 := bitValue_le_one source 42
  have b11 := bitValue_le_one source 43
  have b12 := bitValue_le_one source 44
  have b13 := bitValue_le_one source 45
  have b14 := bitValue_le_one source 46
  have b15 := bitValue_le_one source 47
  have b16 := bitValue_le_one source 48
  have b17 := bitValue_le_one source 49
  have b18 := bitValue_le_one source 50
  have b19 := bitValue_le_one source 51
  have b20 := bitValue_le_one source 52
  have b21 := bitValue_le_one source 53
  have b22 := bitValue_le_one source 54
  have b23 := bitValue_le_one source 55
  have b24 := bitValue_le_one source 56
  have b25 := bitValue_le_one source 57
  have b26 := bitValue_le_one source 58
  have b27 := bitValue_le_one source 59
  have b28 := bitValue_le_one source 60
  have b29 := bitValue_le_one source 61
  have b30 := bitValue_le_one source 62
  have b31 := bitValue_le_one source 63
  simp [highValue, range32_shape, highMax] at *
  omega

theorem lowValue_le_max (source : Source) :
    lowValue source ≤ highMax := by
  have b0 := bitValue_le_one source 0
  have b1 := bitValue_le_one source 1
  have b2 := bitValue_le_one source 2
  have b3 := bitValue_le_one source 3
  have b4 := bitValue_le_one source 4
  have b5 := bitValue_le_one source 5
  have b6 := bitValue_le_one source 6
  have b7 := bitValue_le_one source 7
  have b8 := bitValue_le_one source 8
  have b9 := bitValue_le_one source 9
  have b10 := bitValue_le_one source 10
  have b11 := bitValue_le_one source 11
  have b12 := bitValue_le_one source 12
  have b13 := bitValue_le_one source 13
  have b14 := bitValue_le_one source 14
  have b15 := bitValue_le_one source 15
  have b16 := bitValue_le_one source 16
  have b17 := bitValue_le_one source 17
  have b18 := bitValue_le_one source 18
  have b19 := bitValue_le_one source 19
  have b20 := bitValue_le_one source 20
  have b21 := bitValue_le_one source 21
  have b22 := bitValue_le_one source 22
  have b23 := bitValue_le_one source 23
  have b24 := bitValue_le_one source 24
  have b25 := bitValue_le_one source 25
  have b26 := bitValue_le_one source 26
  have b27 := bitValue_le_one source 27
  have b28 := bitValue_le_one source 28
  have b29 := bitValue_le_one source 29
  have b30 := bitValue_le_one source 30
  have b31 := bitValue_le_one source 31
  simp [lowValue, range32_shape, highMax] at *
  omega

theorem highDifference_lt_modulus (source : Source) :
    highDifference source < goldilocksP := by
  unfold highDifference
  exact Nat.mod_lt _ (by decide)

private theorem bitRow_complete (field : FieldInverse) (source : Source)
    {index : Nat} (bounded : index < 64) :
    RowHolds (interpret field source) (bitRow (bitCol index)) := by
  rw [show bitRow (bitCol index) =
      ⟨[(bitCol index, 1)],
       [(bitCol index, 1), (0, goldilocksP - 1)], []⟩ by rfl]
  simp only [RowHolds, lcEval, List.foldl, interpret_bit field source bounded,
    interpret_one]
  cases h : source.bit index <;> simp [bitValue, h, goldilocksP]

private theorem bitRows_complete (field : FieldInverse) (source : Source) :
    Satisfies bitRows (interpret field source) := by
  intro row member
  rcases List.mem_map.mp member with ⟨index, indexMember, rfl⟩
  apply bitRow_complete field source
  simpa using List.mem_range.mp indexMember

private theorem lowTerms_eval (field : FieldInverse) (source : Source) :
    lcEval (interpret field source) lowTerms = lowValue source := by
  unfold lcEval
  have raw :
      lowTerms.foldl
          (fun acc term => acc + term.2 * interpret field source term.1) 0 =
        lowValue source := by
    simp [lowTerms, lowValue, range32_shape, interpret, bitCol, varCol]
  rw [raw]
  apply Nat.mod_eq_of_lt
  have bound := lowValue_le_max source
  simp only [highMax, goldilocksP] at bound ⊢
  omega

private theorem highTerms_eval (field : FieldInverse) (source : Source) :
    lcEval (interpret field source) highTerms = highValue source := by
  unfold lcEval
  have raw :
      highTerms.foldl
          (fun acc term => acc + term.2 * interpret field source term.1) 0 =
        highValue source := by
    simp [highTerms, highValue, range32_shape, interpret, bitCol, varCol]
  rw [raw]
  apply Nat.mod_eq_of_lt
  have bound := highValue_le_max source
  simp only [highMax, goldilocksP] at bound ⊢
  omega

private theorem highDifferenceTerms_eval
    (field : FieldInverse) (source : Source) :
    lcEval (interpret field source) highDifferenceTerms =
      highDifference source := by
  unfold lcEval
  have raw :
      highDifferenceTerms.foldl
          (fun acc term => acc + term.2 * interpret field source term.1) 0 =
        highValue source + (goldilocksP - highMax) := by
    simp [highDifferenceTerms, highTerms, highValue, range32_shape,
      interpret, bitCol, varCol]
  rw [raw]
  rfl

private theorem highFlagBit_complete
    (field : FieldInverse) (source : Source) :
    RowHolds (interpret field source) (bitRow 66) := by
  have flagEval :
      lcEval (interpret field source) [(66, 1)] = highIsMax source := by
    simp [lcEval, Nat.mod_eq_of_lt (highIsMax_lt_modulus source)]
  have minusOneEval :
      lcEval (interpret field source)
          [(66, 1), (0, goldilocksP - 1)] =
        (highIsMax source + (goldilocksP - 1)) % goldilocksP := by
    simp [lcEval]
  simp only [RowHolds, bitRow, flagEval, minusOneEval]
  unfold highIsMax
  split <;> simp [lcEval, goldilocksP]

private theorem highDefinition_complete
    (field : FieldInverse) (source : Source) :
    RowHolds (interpret field source) highDefinitionRow := by
  have difference := highDifferenceTerms_eval field source
  have flagEval :
      lcEval (interpret field source) [(66, 1)] = highIsMax source := by
    simp [lcEval, Nat.mod_eq_of_lt (highIsMax_lt_modulus source)]
  simp only [RowHolds, highDefinitionRow, flagEval, difference]
  unfold highIsMax
  split
  case isTrue equal =>
    simp [highDifference, equal, highMax, goldilocksP, lcEval]
  case isFalse notEqual => simp [lcEval]

private theorem inverse_complete
    (field : FieldInverse) (source : Source) :
    RowHolds (interpret field source) inverseRow := by
  have difference := highDifferenceTerms_eval field source
  have inverseEval :
      lcEval (interpret field source) [(67, 1)] =
        field.inverse (highDifference source) := by
    simp [lcEval, Nat.mod_eq_of_lt (field.canonical _)]
  have targetEval :
      lcEval (interpret field source)
          [(66, goldilocksP - 1), (0, 1)] =
        ((goldilocksP - 1) * highIsMax source + 1) % goldilocksP := by
    simp [lcEval]
  simp only [RowHolds, inverseRow, difference, inverseEval, targetEval]
  by_cases equal : highValue source = highMax
  · simp [highIsMax, equal, highDifference, highMax, goldilocksP]
  · have highBound := highValue_le_max source
    have highLt : highValue source < highMax := by omega
    have rawLt :
        highValue source + (goldilocksP - highMax) < goldilocksP := by
      simp only [goldilocksP, highMax] at highLt ⊢
      omega
    have rawPositive :
        0 < highValue source + (goldilocksP - highMax) := by
      simp only [goldilocksP, highMax]
      omega
    have differenceNonzero : highDifference source ≠ 0 := by
      rw [highDifference, Nat.mod_eq_of_lt rawLt]
      omega
    have inverseLaw := field.correct (highDifference source)
      (highDifference_lt_modulus source) differenceNonzero
    simpa [highIsMax, equal, goldilocksP] using inverseLaw

private theorem canonicality_complete
    (field : FieldInverse) (source : Source)
    (canonical : wordValue source < goldilocksP) :
    RowHolds (interpret field source) canonicalityRow := by
  have low := lowTerms_eval field source
  have flagEval :
      lcEval (interpret field source) [(66, 1)] = highIsMax source := by
    simp [lcEval, Nat.mod_eq_of_lt (highIsMax_lt_modulus source)]
  simp only [RowHolds, canonicalityRow, flagEval, low]
  by_cases equal : highValue source = highMax
  · have lowZero : lowValue source = 0 := by
      simp only [wordValue, equal, highMax, goldilocksP] at canonical
      omega
    simp [highIsMax, equal, lowZero, lcEval]
  · simp [highIsMax, equal, lcEval]

private theorem recompose_raw
    (field : FieldInverse) (source : Source) :
    ((varCol, 1) :: negativeBitTerms).foldl
        (fun acc term => acc + term.2 * interpret field source term.1) 0 =
      goldilocksP * oneCount source := by
  simp [negativeBitTerms, oneCount, wordValue, lowValue, highValue,
    range32_shape, range64_shape, interpret, bitCol, varCol, goldilocksP]
  omega

private theorem recompose_complete
    (field : FieldInverse) (source : Source) :
    RowHolds (interpret field source) recomposeRow := by
  have raw := recompose_raw field source
  have aEval :
      lcEval (interpret field source) ((varCol, 1) :: negativeBitTerms) = 0 := by
    unfold lcEval
    rw [raw]
    rw [Nat.mul_comm]
    exact Nat.mul_mod_left _ _
  have oneEval : lcEval (interpret field source) [(0, 1)] = 1 := by
    simp [lcEval, goldilocksP]
  change
    lcEval (interpret field source) ((varCol, 1) :: negativeBitTerms) *
          lcEval (interpret field source) [(0, 1)] % goldilocksP =
      lcEval (interpret field source) []
  rw [aEval, oneEval]
  simp [lcEval]

/-- Interpreting a canonical source word produces canonical field residues on
every local column. -/
theorem interpret_canonical
    (field : FieldInverse) (source : Source)
    (canonical : wordValue source < goldilocksP) :
    ∀ column, interpret field source column < goldilocksP := by
  intro column
  unfold interpret
  split
  · simp [goldilocksP]
  split
  · exact canonical
  split
  · exact bitValue_lt_modulus source _
  split
  · exact highIsMax_lt_modulus source
  split
  · exact field.canonical _
  · simp [goldilocksP]

/-- Any canonical source word, run through the compiler interpreter, satisfies
all 69 exact generated rows. -/
theorem complete
    (field : FieldInverse) (source : Source)
    (canonical : wordValue source < goldilocksP) :
    Satisfies rows (interpret field source) := by
  rw [rows_shape]
  intro row member
  simp only [List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with inBits | rfl | rfl | rfl | rfl | rfl
  · exact bitRows_complete field source row inBits
  · exact recompose_complete field source
  · exact highFlagBit_complete field source
  · exact highDefinition_complete field source
  · exact inverse_complete field source
  · exact canonicality_complete field source canonical

/-- Source/interpreter certificate for one native compiler invocation.  There
is deliberately no exact-row or decoded-acceptance field. -/
structure ExecutionWitness
    (field : FieldInverse) (assignment : Nat → Nat) where
  source : Source
  sourceCanonical : wordValue source < goldilocksP
  executed : interpret field source = assignment

/-- Compiler completeness phrased at the assignment returned by execution. -/
theorem native_complete
    {field : FieldInverse} {assignment : Nat → Nat}
    (witness : ExecutionWitness field assignment) :
    Satisfies rows assignment := by
  rw [← witness.executed]
  exact complete field witness.source witness.sourceCanonical

/-- Column-renamed compiler completeness for production call sites. -/
theorem mapped_complete
    (columnMap : List Nat)
    {field : FieldInverse} {assignment : Nat → Nat}
    (witness : ExecutionWitness field
      (Relabel.assignment columnMap assignment)) :
    Satisfies (rows.map (Relabel.row columnMap)) assignment := by
  apply (Relabel.satisfies_mapped_iff rows columnMap assignment).mpr
  exact native_complete witness

end Nightstream.Implementation.R1CS.CanonicalU64Complete
