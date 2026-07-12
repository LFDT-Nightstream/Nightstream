import Nightstream.Implementation.R1CS.Correspondence.CanonicalU64.CanonicalU64Sound
import Nightstream.Implementation.R1CS.Correspondence.CanonicalU64.CanonicalU64Complete
import Nightstream.Implementation.R1CS.Correspondence.U64.U64IncrementSound
import Nightstream.Implementation.R1CS.Correspondence.U64.U64AddSound
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Links

/-!
Contract: artifact-level soundness of the production-used recursive F' counter
block. Property `CIR-SOUND full-history counter`.

The 660 exact embedded rows bind source-image words to the incoming field counters,
enforce canonical decompositions of both outputs, pin the batch size to one,
and enforce no-wrap increment/addition. The proof below is quantified over
every canonical-residue assignment satisfying those exact rows. Its small
row-inclusion certificates project the large block onto the previously proved
canonical-u64, increment, and addition row programs.
-/

set_option maxRecDepth 32768
set_option maxHeartbeats 4000000

namespace Nightstream.Implementation.R1CS

namespace FPrimeFullHistoryCounterLocalSound

def columnMap (columns : List Nat) : Nat → Nat :=
  fun i => columns.getD i 0

def canonicalChunkInputMap : Nat → Nat := columnMap FPrimeFullHistoryCounter.chunkInputCanonicalMap
def canonicalStepInputMap : Nat → Nat := columnMap FPrimeFullHistoryCounter.stepInputCanonicalMap
def canonicalChunkOutputMap : Nat → Nat := columnMap FPrimeFullHistoryCounter.chunkOutputCanonicalMap
def canonicalStepOutputMap : Nat → Nat := columnMap FPrimeFullHistoryCounter.stepOutputCanonicalMap
def incrementColumnMap : Nat → Nat := columnMap FPrimeFullHistoryCounter.incrementMap
def addColumnMap : Nat → Nat := columnMap FPrimeFullHistoryCounter.addMap

def chunkOutputVarCol : Nat := canonicalChunkOutputMap CanonicalU64.varCol
def stepOutputVarCol : Nat := canonicalStepOutputMap CanonicalU64.varCol

def bitVectorValue (bits : Nat → Nat) : Nat :=
  (List.range 64).foldl (fun acc i => acc + 2 ^ i * bits i) 0

def chunkInputValue (z : Nat → Nat) : Nat :=
  bitVectorValue (fun i => z (incrementColumnMap (U64Increment.inputBitCol i)))

def chunkOutputValue (z : Nat → Nat) : Nat :=
  bitVectorValue (fun i => z (incrementColumnMap (U64Increment.outputBitCol i)))

def stepInputValue (z : Nat → Nat) : Nat :=
  bitVectorValue (fun i => z (addColumnMap (U64Add.lhsBitCol i)))

def rowsInChunkValue (z : Nat → Nat) : Nat :=
  bitVectorValue (fun i => z (addColumnMap (U64Add.rhsBitCol i)))

def stepOutputValue (z : Nat → Nat) : Nat :=
  bitVectorValue (fun i => z (addColumnMap (U64Add.outputBitCol i)))

private theorem foldl_bits_congr (xs : List Nat) (f g : Nat → Nat)
    (h : ∀ i ∈ xs, f i = g i) (initial : Nat) :
    xs.foldl (fun acc i => acc + 2 ^ i * f i) initial =
      xs.foldl (fun acc i => acc + 2 ^ i * g i) initial := by
  induction xs generalizing initial with
  | nil => rfl
  | cons head tail ih =>
      simp only [List.foldl_cons]
      rw [h head (by simp)]
      apply ih
      intro i hi
      exact h i (by simp [hi])

private theorem bitVectorValue_congr {f g : Nat → Nat}
    (h : ∀ i, i < 64 → f i = g i) :
    bitVectorValue f = bitVectorValue g := by
  unfold bitVectorValue
  apply foldl_bits_congr
  intro i hi
  exact h i (List.mem_range.mp hi)

def columnsAgree (f g : Nat → Nat) : Bool :=
  (List.range 64).all (fun i => decide (f i = g i))

private theorem columnsAgree_sound {f g : Nat → Nat}
    (h : columnsAgree f g = true) :
    ∀ i, i < 64 → f i = g i := by
  intro i hi
  have hdecide := (List.all_eq_true.mp h) i (List.mem_range.mpr hi)
  exact of_decide_eq_true hdecide

private theorem chunkInputCanonicalRowsIncluded :
    rowsIncluded
      (CanonicalU64.rows.map (renameRow canonicalChunkInputMap))
      FPrimeFullHistoryCounter.rows = true := by
  decide

private theorem stepInputCanonicalRowsIncluded :
    rowsIncluded
      (CanonicalU64.rows.map (renameRow canonicalStepInputMap))
      FPrimeFullHistoryCounter.rows = true := by
  decide

private theorem chunkOutputCanonicalRowsIncluded :
    rowsIncluded
      (CanonicalU64.rows.map (renameRow canonicalChunkOutputMap))
      FPrimeFullHistoryCounter.rows = true := by
  decide

private theorem stepOutputCanonicalRowsIncluded :
    rowsIncluded
      (CanonicalU64.rows.map (renameRow canonicalStepOutputMap))
      FPrimeFullHistoryCounter.rows = true := by
  decide

private theorem incrementRowsIncluded :
    rowsIncluded
      (U64Increment.rows.map (renameRow incrementColumnMap))
      FPrimeFullHistoryCounter.rows = true := by
  decide

private theorem addRowsIncluded :
    rowsIncluded
      (U64Add.rows.map (renameRow addColumnMap))
      FPrimeFullHistoryCounter.rows = true := by
  decide

private theorem chunkInputColumnsAgree :
    columnsAgree
      (fun i => incrementColumnMap (U64Increment.inputBitCol i))
      (fun i => canonicalChunkInputMap (CanonicalU64.bitCol i)) = true := by
  decide

private theorem chunkOutputColumnsAgree :
    columnsAgree
      (fun i => incrementColumnMap (U64Increment.outputBitCol i))
      (fun i => canonicalChunkOutputMap (CanonicalU64.bitCol i)) = true := by
  decide

private theorem stepInputColumnsAgree :
    columnsAgree
      (fun i => addColumnMap (U64Add.lhsBitCol i))
      (fun i => canonicalStepInputMap (CanonicalU64.bitCol i)) = true := by
  decide

private theorem stepOutputColumnsAgree :
    columnsAgree
      (fun i => addColumnMap (U64Add.outputBitCol i))
      (fun i => canonicalStepOutputMap (CanonicalU64.bitCol i)) = true := by
  decide

def expectedRowsBit (i : Nat) : Nat := if i < 1 then 1 else 0

def constantRow (column value : Nat) : Row :=
  if value = 0 then
    ⟨[(column, 1)], [(0, 1)], []⟩
  else
    ⟨[(column, 1), (0, goldilocksP - value)], [(0, 1)], []⟩

def rowsInChunkConstraintRows : List Row :=
  (List.range 64).flatMap (fun i =>
    [bitRow (addColumnMap (U64Add.rhsBitCol i)),
     constantRow (addColumnMap (U64Add.rhsBitCol i)) (expectedRowsBit i)])

private theorem rowsInChunkConstraintRowsIncluded :
    rowsIncluded rowsInChunkConstraintRows FPrimeFullHistoryCounter.rows = true := by
  decide

private theorem rowsInChunk_bit
    (hq : EuclidPrime goldilocksP) {z : Nat → Nat}
    (hcanon : ∀ i, z i < goldilocksP) (hone : z 0 = 1)
    (hsat : Satisfies FPrimeFullHistoryCounter.rows z) (i : Nat) (hi : i < 64) :
    z (addColumnMap (U64Add.rhsBitCol i)) = expectedRowsBit i := by
  let col := addColumnMap (U64Add.rhsBitCol i)
  have hbitMem : bitRow col ∈ rowsInChunkConstraintRows := by
    apply List.mem_flatMap.mpr
    exact ⟨i, List.mem_range.mpr hi, by simp [col]⟩
  have hconstMem : constantRow col (expectedRowsBit i) ∈
      rowsInChunkConstraintRows := by
    apply List.mem_flatMap.mpr
    exact ⟨i, List.mem_range.mpr hi, by simp [col]⟩
  have hbit : RowHolds z (bitRow col) :=
    hsat _ (rowsIncluded_sound rowsInChunkConstraintRowsIncluded _ hbitMem)
  have hconst : RowHolds z (constantRow col (expectedRowsBit i)) :=
    hsat _ (rowsIncluded_sound rowsInChunkConstraintRowsIncluded _ hconstMem)
  have hle : z col ≤ 1 := bitRow_le_one hq (hcanon col) hone hbit
  have hz : z col = 0 ∨ z col = 1 := by omega
  by_cases hsmall : i < 1 <;>
    rcases hz with hz | hz <;>
    simp_all [col, expectedRowsBit, constantRow, RowHolds, lcEval,
      goldilocksP]

private theorem rowsInChunkValue_eq
    (hq : EuclidPrime goldilocksP) {z : Nat → Nat}
    (hcanon : ∀ i, z i < goldilocksP) (hone : z 0 = 1)
    (hsat : Satisfies FPrimeFullHistoryCounter.rows z) :
    rowsInChunkValue z = FPrimeFullHistoryCounter.rowsInChunk := by
  have hbits : ∀ i, i < 64 →
      z (addColumnMap (U64Add.rhsBitCol i)) = expectedRowsBit i :=
    fun i hi => rowsInChunk_bit hq hcanon hone hsat i hi
  calc
    rowsInChunkValue z = bitVectorValue expectedRowsBit := by
      apply bitVectorValue_congr
      exact hbits
    _ = FPrimeFullHistoryCounter.rowsInChunk := by decide

private theorem pulledCanonical
    {z : Nat → Nat} (hcanon : ∀ i, z i < goldilocksP)
    (f : Nat → Nat) :
    ∀ i, pullAssignment z f i < goldilocksP :=
  fun i => hcanon (f i)

private theorem pulledOne {z : Nat → Nat} (hone : z 0 = 1)
    (columns : List Nat) (hzero : columns.getD 0 0 = 0) :
    pullAssignment z (columnMap columns) 0 = 1 := by
  change z (columns.getD 0 0) = 1
  rw [hzero, hone]

/--
Every canonical-residue assignment satisfying the exact production-used F'
counter artifact binds its field counters to canonical source words and obeys
the two intended integer, no-wrap transition equations.
-/
theorem local_sound (hq : EuclidPrime goldilocksP)
    {z : Nat → Nat}
    (hcanon : ∀ i, z i < goldilocksP)
    (hone : z 0 = 1)
    (hsat : Satisfies FPrimeFullHistoryCounter.rows z) :
    z FPrimeFullHistoryCounter.chunkInputVarCol = chunkInputValue z ∧
    z FPrimeFullHistoryCounter.stepInputVarCol = stepInputValue z ∧
    z chunkOutputVarCol = chunkOutputValue z ∧
    z stepOutputVarCol = stepOutputValue z ∧
    chunkOutputValue z = chunkInputValue z + 1 ∧
    stepOutputValue z = stepInputValue z + FPrimeFullHistoryCounter.rowsInChunk ∧
    chunkInputValue z < goldilocksP ∧
    stepInputValue z < goldilocksP ∧
    chunkOutputValue z < goldilocksP ∧
    stepOutputValue z < goldilocksP := by
  have hChunkInSat := satisfies_pull_of_rowsIncluded
    canonicalChunkInputMap chunkInputCanonicalRowsIncluded hsat
  have hStepInSat := satisfies_pull_of_rowsIncluded
    canonicalStepInputMap stepInputCanonicalRowsIncluded hsat
  have hChunkOutSat := satisfies_pull_of_rowsIncluded
    canonicalChunkOutputMap chunkOutputCanonicalRowsIncluded hsat
  have hStepOutSat := satisfies_pull_of_rowsIncluded
    canonicalStepOutputMap stepOutputCanonicalRowsIncluded hsat
  have hIncrementSat := satisfies_pull_of_rowsIncluded
    incrementColumnMap incrementRowsIncluded hsat
  have hAddSat := satisfies_pull_of_rowsIncluded
    addColumnMap addRowsIncluded hsat

  have hChunkInCanonical := canonicalU64_sound hq
    (pulledCanonical hcanon canonicalChunkInputMap)
    (pulledOne hone FPrimeFullHistoryCounter.chunkInputCanonicalMap (by decide)) hChunkInSat
  have hStepInCanonical := canonicalU64_sound hq
    (pulledCanonical hcanon canonicalStepInputMap)
    (pulledOne hone FPrimeFullHistoryCounter.stepInputCanonicalMap (by decide)) hStepInSat
  have hChunkOutCanonical := canonicalU64_sound hq
    (pulledCanonical hcanon canonicalChunkOutputMap)
    (pulledOne hone FPrimeFullHistoryCounter.chunkOutputCanonicalMap (by decide)) hChunkOutSat
  have hStepOutCanonical := canonicalU64_sound hq
    (pulledCanonical hcanon canonicalStepOutputMap)
    (pulledOne hone FPrimeFullHistoryCounter.stepOutputCanonicalMap (by decide)) hStepOutSat
  have hIncrement := u64Increment_sound hq
    (pulledCanonical hcanon incrementColumnMap)
    (pulledOne hone FPrimeFullHistoryCounter.incrementMap (by decide)) hIncrementSat
  have hAdd := u64Add_sound hq
    (pulledCanonical hcanon addColumnMap)
    (pulledOne hone FPrimeFullHistoryCounter.addMap (by decide)) hAddSat
  have hRows := rowsInChunkValue_eq hq hcanon hone hsat

  have hChunkInBits :
      chunkInputValue z = bitsValue
        (pullAssignment z canonicalChunkInputMap) := by
    unfold chunkInputValue bitsValue
    simp only [pullAssignment, bitVectorValue]
    apply foldl_bits_congr
    intro i hi
    rw [columnsAgree_sound chunkInputColumnsAgree i (List.mem_range.mp hi)]
  have hChunkOutBits :
      chunkOutputValue z = bitsValue
        (pullAssignment z canonicalChunkOutputMap) := by
    unfold chunkOutputValue bitsValue
    simp only [pullAssignment, bitVectorValue]
    apply foldl_bits_congr
    intro i hi
    rw [columnsAgree_sound chunkOutputColumnsAgree i (List.mem_range.mp hi)]
  have hStepInBits :
      stepInputValue z = bitsValue
        (pullAssignment z canonicalStepInputMap) := by
    unfold stepInputValue bitsValue
    simp only [pullAssignment, bitVectorValue]
    apply foldl_bits_congr
    intro i hi
    rw [columnsAgree_sound stepInputColumnsAgree i (List.mem_range.mp hi)]
  have hStepOutBits :
      stepOutputValue z = bitsValue
        (pullAssignment z canonicalStepOutputMap) := by
    unfold stepOutputValue bitsValue
    simp only [pullAssignment, bitVectorValue]
    apply foldl_bits_congr
    intro i hi
    rw [columnsAgree_sound stepOutputColumnsAgree i (List.mem_range.mp hi)]

  have hChunkIncrement : chunkOutputValue z = chunkInputValue z + 1 := by
    simpa [chunkInputValue, chunkOutputValue, bitVectorValue,
      incrementInputValue, incrementOutputValue,
      pullAssignment] using hIncrement
  have hStepAdd : stepOutputValue z = stepInputValue z + rowsInChunkValue z := by
    simpa [stepInputValue, stepOutputValue, rowsInChunkValue, bitVectorValue,
      addLhsValue, addRhsValue, addOutputValue,
      pullAssignment] using hAdd

  refine ⟨?_, ?_, ?_, ?_, hChunkIncrement, ?_, ?_, ?_, ?_, ?_⟩
  · simpa [canonicalChunkInputMap, columnMap, FPrimeFullHistoryCounter.chunkInputVarCol,
      CanonicalU64.varCol, hChunkInBits] using hChunkInCanonical.1
  · simpa [canonicalStepInputMap, columnMap, FPrimeFullHistoryCounter.stepInputVarCol,
      CanonicalU64.varCol, hStepInBits] using hStepInCanonical.1
  · simpa [chunkOutputVarCol, hChunkOutBits] using hChunkOutCanonical.1
  · simpa [stepOutputVarCol, hStepOutBits] using hStepOutCanonical.1
  · simpa [hRows] using hStepAdd
  · simpa [hChunkInBits] using hChunkInCanonical.2
  · simpa [hStepInBits] using hStepInCanonical.2
  · simpa [hChunkOutBits] using hChunkOutCanonical.2
  · simpa [hStepOutBits] using hStepOutCanonical.2

namespace Compiler

open CanonicalU64Complete

/-! The compiler model below is deliberately source-first.  It runs the
ordinary ripple-carry algorithm on two input words and then exposes the four
canonical-u64 calls, the increment call, and the add-one call made by Rust.
No witness field states row satisfaction or a decoded counter conclusion. -/

/-- Carry entering bit `index` when incrementing a word by one. -/
def carryIn (source : CanonicalU64Complete.Source) : Nat → Bool
  | 0 => true
  | index + 1 => source.bit index && carryIn source index

def outputBit (source : CanonicalU64Complete.Source) (index : Nat) : Bool :=
  Bool.xor (source.bit index) (carryIn source index)

def carryOut (source : CanonicalU64Complete.Source) (index : Nat) : Bool :=
  source.bit index && carryIn source index

def outputSource (source : CanonicalU64Complete.Source) :
    CanonicalU64Complete.Source where
  bit := outputBit source

@[simp] theorem carryOut_eq_nextCarry
    (source : CanonicalU64Complete.Source) (index : Nat) :
    carryOut source index = carryIn source (index + 1) := by
  simp [carryOut, carryIn]

def boolValue (value : Bool) : Nat := value.toNat

theorem boolValue_lt_modulus (value : Bool) : boolValue value < goldilocksP := by
  cases value <;> simp [boolValue, goldilocksP]

/-- Local execution of `enforce_u64_increment`.  Row-irrelevant columns
default to one so finite `getD ... 0` relabelings have the same fallback. -/
def incrementInterpret (source : CanonicalU64Complete.Source) : Nat → Nat :=
  fun column =>
    if column = 0 then 1
    else if 1 ≤ column ∧ column < 65 then source.bit (column - 1) |>.toNat
    else if 65 ≤ column ∧ column < 129 then
      outputBit source (column - 65) |>.toNat
    else if 129 ≤ column ∧ column < 192 then
      carryOut source (column - 129) |>.toNat
    else 1

@[simp] theorem incrementInterpret_one (source : CanonicalU64Complete.Source) :
    incrementInterpret source 0 = 1 := by
  simp [incrementInterpret]

@[simp] theorem incrementInterpret_input
    (source : CanonicalU64Complete.Source) {index : Nat} (bounded : index < 64) :
    incrementInterpret source (U64Increment.inputBitCol index) =
      CanonicalU64Complete.bitValue source index := by
  simp [incrementInterpret, U64Increment.inputBitCol,
    CanonicalU64Complete.bitValue]
  omega

@[simp] theorem incrementInterpret_output
    (source : CanonicalU64Complete.Source) {index : Nat} (bounded : index < 64) :
    incrementInterpret source (U64Increment.outputBitCol index) =
      boolValue (outputBit source index) := by
  have notZero : index + 65 ≠ 0 := by omega
  have notInput : ¬(1 ≤ index + 65 ∧ index + 65 < 65) := by omega
  have notInputLt : ¬ index + 65 < 65 := by omega
  have isOutput : 65 ≤ index + 65 ∧ index + 65 < 129 := by omega
  simp [incrementInterpret, U64Increment.outputBitCol, notZero, notInput,
    notInputLt, isOutput, boolValue]

@[simp] theorem incrementInterpret_carry
    (source : CanonicalU64Complete.Source) {index : Nat} (bounded : index < 63) :
    incrementInterpret source (U64Increment.carryCol index) =
      boolValue (carryOut source index) := by
  have notZero : index + 129 ≠ 0 := by omega
  have notInput : ¬(1 ≤ index + 129 ∧ index + 129 < 65) := by omega
  have notOutput : ¬(65 ≤ index + 129 ∧ index + 129 < 129) := by omega
  have notInputLt : ¬ index + 129 < 65 := by omega
  have notOutputLt : ¬ index + 129 < 129 := by omega
  have isCarry : 129 ≤ index + 129 ∧ index + 129 < 192 := by omega
  simp [incrementInterpret, U64Increment.carryCol, notZero, notInput,
    notOutput, notInputLt, notOutputLt, isCarry, boolValue]

theorem incrementInterpret_canonical (source : CanonicalU64Complete.Source) :
    ∀ column, incrementInterpret source column < goldilocksP := by
  intro column
  unfold incrementInterpret
  split
  · simp [goldilocksP]
  split
  · exact CanonicalU64Complete.bitValue_lt_modulus source _
  split
  · exact boolValue_lt_modulus _
  split
  · exact boolValue_lt_modulus _
  · simp [goldilocksP]

private theorem bitRow_bool_complete (value : Bool) (column : Nat)
    {assignment : Nat → Nat}
    (one : assignment 0 = 1)
    (atColumn : assignment column = boolValue value) :
    RowHolds assignment (bitRow column) := by
  cases value <;> simp [RowHolds, bitRow, lcEval, one, atColumn,
    boolValue, goldilocksP]

private theorem incrementEquation_complete
    (source : CanonicalU64Complete.Source) (index : Nat) (bounded : index < 63) :
    RowHolds (incrementInterpret source) (incrementEquationRow index) := by
  by_cases zero : index = 0
  · subst index
    cases input : source.bit 0 <;>
      simp [incrementEquationRow, incrementCarryRow, RowHolds, lcEval,
        incrementInterpret_input, incrementInterpret_output,
        incrementInterpret_carry, outputBit, carryOut, carryIn,
        CanonicalU64Complete.bitValue, boolValue, input, goldilocksP]
  · have previousBound : index - 1 < 63 := by omega
    have inputBound : index < 64 := by omega
    have previousCarry : carryOut source (index - 1) = carryIn source index := by
      rw [carryOut_eq_nextCarry]
      congr
      omega
    simp only [incrementEquationRow, zero, ↓reduceIte, incrementCarryRow,
      RowHolds, lcEval, List.foldl]
    rw [incrementInterpret_input source inputBound,
      incrementInterpret_carry source previousBound,
      incrementInterpret_output source inputBound,
      incrementInterpret_carry source bounded]
    rw [previousCarry]
    cases input : source.bit index <;>
      cases carry : carryIn source index <;>
      simp [incrementEquationRow, incrementCarryRow, RowHolds, lcEval,
        outputBit, carryOut, CanonicalU64Complete.bitValue, boolValue,
        input, carry, zero,
        goldilocksP]

private theorem incrementFinal_complete
    (source : CanonicalU64Complete.Source)
    (noWrap : carryOut source 63 = false) :
    RowHolds (incrementInterpret source) incrementFinalRow := by
  have priorCarry : carryOut source 62 = carryIn source 63 := by
    simpa using carryOut_eq_nextCarry source 62
  simp only [incrementFinalRow, RowHolds, lcEval, List.foldl]
  rw [incrementInterpret_input source (by omega),
    incrementInterpret_carry source (by omega),
    incrementInterpret_output source (by omega)]
  rw [priorCarry]
  cases input : source.bit 63 <;>
    cases carry : carryIn source 63 <;>
    simp [incrementFinalRow, RowHolds, lcEval, incrementInterpret_input,
      incrementInterpret_output, incrementInterpret_carry, outputBit,
      carryOut, CanonicalU64Complete.bitValue, boolValue, input, carry,
      goldilocksP] at noWrap ⊢

/-- Independent compiler completeness for the exact 255-row increment
artifact. -/
theorem increment_complete
    (source : CanonicalU64Complete.Source)
    (noWrap : carryOut source 63 = false) :
    Satisfies U64Increment.rows (incrementInterpret source) := by
  rw [artifactRows_eq]
  intro row member
  simp only [expectedIncrementRows, List.mem_append] at member
  rcases member with ((inInput | inOutput) | inBody) | inFinal
  · rcases List.mem_map.mp inInput with ⟨index, indexMember, rfl⟩
    apply bitRow_bool_complete (source.bit index)
    · exact incrementInterpret_one source
    · simpa [CanonicalU64Complete.bitValue, boolValue] using
        incrementInterpret_input source (List.mem_range.mp indexMember)
  · rcases List.mem_map.mp inOutput with ⟨index, indexMember, rfl⟩
    apply bitRow_bool_complete (outputBit source index)
    · exact incrementInterpret_one source
    · exact incrementInterpret_output source (List.mem_range.mp indexMember)
  · rcases List.mem_flatMap.mp inBody with ⟨index, indexMember, inPair⟩
    simp only [List.mem_cons, List.not_mem_nil, or_false] at inPair
    rcases inPair with rfl | rfl
    · apply bitRow_bool_complete (carryOut source index)
      · exact incrementInterpret_one source
      · exact incrementInterpret_carry source (List.mem_range.mp indexMember)
    · exact incrementEquation_complete source index
        (List.mem_range.mp indexMember)
  · have equal : row = incrementFinalRow := by simpa using inFinal
    subst row
    exact incrementFinal_complete source noWrap

/-- Local execution of the production add gadget specialized to the batch
size one. -/
def addOneInterpret (source : CanonicalU64Complete.Source) : Nat → Nat :=
  fun column =>
    if column = 0 then 1
    else if 1 ≤ column ∧ column < 65 then source.bit (column - 1) |>.toNat
    else if 65 ≤ column ∧ column < 129 then
      if column = 65 then 1 else 0
    else if 129 ≤ column ∧ column < 193 then
      outputBit source (column - 129) |>.toNat
    else if 193 ≤ column ∧ column < 256 then
      carryOut source (column - 193) |>.toNat
    else 1

@[simp] theorem addOneInterpret_one (source : CanonicalU64Complete.Source) :
    addOneInterpret source 0 = 1 := by simp [addOneInterpret]

@[simp] theorem addOneInterpret_lhs (source : CanonicalU64Complete.Source)
    {index : Nat} (bounded : index < 64) :
    addOneInterpret source (U64Add.lhsBitCol index) =
      CanonicalU64Complete.bitValue source index := by
  simp [addOneInterpret, U64Add.lhsBitCol, CanonicalU64Complete.bitValue]
  omega

@[simp] theorem addOneInterpret_rhs (source : CanonicalU64Complete.Source)
    {index : Nat} (bounded : index < 64) :
    addOneInterpret source (U64Add.rhsBitCol index) =
      if index = 0 then 1 else 0 := by
  have notZero : index + 65 ≠ 0 := by omega
  have notLhs : ¬(1 ≤ index + 65 ∧ index + 65 < 65) := by omega
  have notLhsLt : ¬ index + 65 < 65 := by omega
  have isRhs : 65 ≤ index + 65 ∧ index + 65 < 129 := by omega
  simp [addOneInterpret, U64Add.rhsBitCol, notZero, notLhs, notLhsLt,
    isRhs]

@[simp] theorem addOneInterpret_output (source : CanonicalU64Complete.Source)
    {index : Nat} (bounded : index < 64) :
    addOneInterpret source (U64Add.outputBitCol index) =
      boolValue (outputBit source index) := by
  have notZero : index + 129 ≠ 0 := by omega
  have notLhs : ¬(1 ≤ index + 129 ∧ index + 129 < 65) := by omega
  have notRhs : ¬(65 ≤ index + 129 ∧ index + 129 < 129) := by omega
  have notLhsLt : ¬ index + 129 < 65 := by omega
  have notRhsLt : ¬ index + 129 < 129 := by omega
  have isOutput : 129 ≤ index + 129 ∧ index + 129 < 193 := by omega
  simp [addOneInterpret, U64Add.outputBitCol, notZero, notLhs, notRhs,
    notLhsLt, notRhsLt, isOutput, boolValue]

@[simp] theorem addOneInterpret_carry (source : CanonicalU64Complete.Source)
    {index : Nat} (bounded : index < 63) :
    addOneInterpret source (U64Add.carryCol index) =
      boolValue (carryOut source index) := by
  have notZero : index + 193 ≠ 0 := by omega
  have notLhs : ¬(1 ≤ index + 193 ∧ index + 193 < 65) := by omega
  have notRhs : ¬(65 ≤ index + 193 ∧ index + 193 < 129) := by omega
  have notOutput : ¬(129 ≤ index + 193 ∧ index + 193 < 193) := by omega
  have notLhsLt : ¬ index + 193 < 65 := by omega
  have notRhsLt : ¬ index + 193 < 129 := by omega
  have notOutputLt : ¬ index + 193 < 193 := by omega
  have isCarry : 193 ≤ index + 193 ∧ index + 193 < 256 := by omega
  simp [addOneInterpret, U64Add.carryCol, notZero, notLhs, notRhs,
    notOutput, notLhsLt, notRhsLt, notOutputLt, isCarry, boolValue]

theorem addOneInterpret_canonical (source : CanonicalU64Complete.Source) :
    ∀ column, addOneInterpret source column < goldilocksP := by
  intro column
  unfold addOneInterpret
  split
  · simp [goldilocksP]
  split
  · exact CanonicalU64Complete.bitValue_lt_modulus source _
  split
  · split <;> simp [goldilocksP]
  split
  · exact boolValue_lt_modulus _
  split
  · exact boolValue_lt_modulus _
  · simp [goldilocksP]

private theorem addEquation_complete
    (source : CanonicalU64Complete.Source) (index : Nat) (bounded : index < 63) :
    RowHolds (addOneInterpret source) (u64AddEquationRow index) := by
  by_cases zero : index = 0
  · subst index
    cases input : source.bit 0 <;>
      simp [u64AddEquationRow, u64AddFirstRow, RowHolds, lcEval,
        addOneInterpret_lhs, addOneInterpret_rhs, addOneInterpret_output,
        addOneInterpret_carry, outputBit, carryOut, carryIn,
        CanonicalU64Complete.bitValue, boolValue, input, goldilocksP]
  · have previousBound : index - 1 < 63 := by omega
    have inputBound : index < 64 := by omega
    have previousCarry : carryOut source (index - 1) = carryIn source index := by
      rw [carryOut_eq_nextCarry]
      congr
      omega
    simp only [u64AddEquationRow, zero, ↓reduceIte, u64AddCarryRow,
      RowHolds, lcEval, List.foldl]
    rw [addOneInterpret_lhs source inputBound,
      addOneInterpret_rhs source inputBound,
      addOneInterpret_carry source previousBound,
      addOneInterpret_output source inputBound,
      addOneInterpret_carry source bounded]
    rw [previousCarry]
    cases input : source.bit index <;>
      cases carry : carryIn source index <;>
      simp [u64AddEquationRow, u64AddCarryRow, RowHolds, lcEval,
        outputBit, carryOut,
        CanonicalU64Complete.bitValue, boolValue, input, carry, zero,
        goldilocksP]

private theorem addFinal_complete
    (source : CanonicalU64Complete.Source)
    (noWrap : carryOut source 63 = false) :
    RowHolds (addOneInterpret source) u64AddFinalRow := by
  have priorCarry : carryOut source 62 = carryIn source 63 := by
    simpa using carryOut_eq_nextCarry source 62
  simp only [u64AddFinalRow, RowHolds, lcEval, List.foldl]
  rw [addOneInterpret_lhs source (by omega),
    addOneInterpret_rhs source (by omega),
    addOneInterpret_carry source (by omega),
    addOneInterpret_output source (by omega)]
  rw [priorCarry]
  cases input : source.bit 63 <;>
    cases carry : carryIn source 63 <;>
    simp [u64AddFinalRow, RowHolds, lcEval, addOneInterpret_lhs,
      addOneInterpret_rhs, addOneInterpret_output, addOneInterpret_carry,
      outputBit, carryOut, CanonicalU64Complete.bitValue, boolValue,
      input, carry, goldilocksP] at noWrap ⊢

/-- Independent compiler completeness for the exact 319-row add artifact,
specialized to the production batch size one. -/
theorem addOne_complete
    (source : CanonicalU64Complete.Source)
    (noWrap : carryOut source 63 = false) :
    Satisfies U64Add.rows (addOneInterpret source) := by
  rw [u64AddArtifactRows_eq]
  intro row member
  simp only [expectedU64AddRows, List.mem_append] at member
  rcases member with (((inLhs | inRhs) | inOutput) | inBody) | inFinal
  · rcases List.mem_map.mp inLhs with ⟨index, indexMember, rfl⟩
    apply bitRow_bool_complete (source.bit index)
    · exact addOneInterpret_one source
    · simpa [CanonicalU64Complete.bitValue, boolValue] using
        addOneInterpret_lhs source (List.mem_range.mp indexMember)
  · rcases List.mem_map.mp inRhs with ⟨index, indexMember, rfl⟩
    by_cases zero : index = 0
    · apply bitRow_bool_complete true
      · exact addOneInterpret_one source
      · simp [addOneInterpret_rhs source (List.mem_range.mp indexMember),
          zero, boolValue]
    · apply bitRow_bool_complete false
      · exact addOneInterpret_one source
      · simp [addOneInterpret_rhs source (List.mem_range.mp indexMember),
          zero, boolValue]
  · rcases List.mem_map.mp inOutput with ⟨index, indexMember, rfl⟩
    apply bitRow_bool_complete (outputBit source index)
    · exact addOneInterpret_one source
    · exact addOneInterpret_output source (List.mem_range.mp indexMember)
  · rcases List.mem_flatMap.mp inBody with ⟨index, indexMember, inPair⟩
    simp only [List.mem_cons, List.not_mem_nil, or_false] at inPair
    rcases inPair with rfl | rfl
    · apply bitRow_bool_complete (carryOut source index)
      · exact addOneInterpret_one source
      · exact addOneInterpret_carry source (List.mem_range.mp indexMember)
    · exact addEquation_complete source index (List.mem_range.mp indexMember)
  · have equal : row = u64AddFinalRow := by simpa using inFinal
    subst row
    exact addFinal_complete source noWrap

private theorem range32_shape : List.range 32 =
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
     30, 31] := by decide

private theorem range64_shape : List.range 64 =
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
     30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
     44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
     58, 59, 60, 61, 62, 63] := by decide

private theorem incrementInput_word
    (source : CanonicalU64Complete.Source) :
    incrementInputValue (incrementInterpret source) =
      CanonicalU64Complete.wordValue source := by
  simp [incrementInputValue, CanonicalU64Complete.wordValue,
    CanonicalU64Complete.lowValue, CanonicalU64Complete.highValue,
    range32_shape, range64_shape]
  omega

private theorem incrementOutput_word
    (source : CanonicalU64Complete.Source) :
    incrementOutputValue (incrementInterpret source) =
      CanonicalU64Complete.wordValue (outputSource source) := by
  simp [incrementOutputValue, CanonicalU64Complete.wordValue,
    CanonicalU64Complete.lowValue, CanonicalU64Complete.highValue,
    CanonicalU64Complete.bitValue, outputSource, boolValue,
    range32_shape, range64_shape]
  omega

private theorem addLhs_word (source : CanonicalU64Complete.Source) :
    addLhsValue (addOneInterpret source) =
      CanonicalU64Complete.wordValue source := by
  simp [addLhsValue, CanonicalU64Complete.wordValue,
    CanonicalU64Complete.lowValue, CanonicalU64Complete.highValue,
    range32_shape, range64_shape]
  omega

private theorem addRhs_one (source : CanonicalU64Complete.Source) :
    addRhsValue (addOneInterpret source) = 1 := by
  simp [addRhsValue, range64_shape]

private theorem addOutput_word (source : CanonicalU64Complete.Source) :
    addOutputValue (addOneInterpret source) =
      CanonicalU64Complete.wordValue (outputSource source) := by
  simp [addOutputValue, CanonicalU64Complete.wordValue,
    CanonicalU64Complete.lowValue, CanonicalU64Complete.highValue,
    CanonicalU64Complete.bitValue, outputSource, boolValue,
    range32_shape, range64_shape]
  omega

theorem incrementOutput_word_eq (prime : EuclidPrime goldilocksP)
    (source : CanonicalU64Complete.Source)
    (noWrap : carryOut source 63 = false) :
    CanonicalU64Complete.wordValue (outputSource source) =
      CanonicalU64Complete.wordValue source + 1 := by
  have sound := u64Increment_sound prime
    (incrementInterpret_canonical source)
    (incrementInterpret_one source)
    (increment_complete source noWrap)
  rw [incrementInput_word, incrementOutput_word] at sound
  exact sound

theorem addOneOutput_word_eq (prime : EuclidPrime goldilocksP)
    (source : CanonicalU64Complete.Source)
    (noWrap : carryOut source 63 = false) :
    CanonicalU64Complete.wordValue (outputSource source) =
      CanonicalU64Complete.wordValue source + 1 := by
  have sound := u64Add_sound prime
    (addOneInterpret_canonical source)
    (addOneInterpret_one source)
    (addOne_complete source noWrap)
  rw [addLhs_word, addRhs_one, addOutput_word] at sound
  exact sound

/-- Valid source inputs for the production full-history counter compiler.
The only preconditions are canonical/no-wrap source conditions; output words
and all intermediate columns are computed by the interpreters above. -/
structure Source where
  chunkInput : CanonicalU64Complete.Source
  stepInput : CanonicalU64Complete.Source
  chunkInputCanonical :
    CanonicalU64Complete.wordValue chunkInput < goldilocksP
  stepInputCanonical :
    CanonicalU64Complete.wordValue stepInput < goldilocksP
  chunkOutputCanonical :
    CanonicalU64Complete.wordValue chunkInput + 1 < goldilocksP
  stepOutputCanonical :
    CanonicalU64Complete.wordValue stepInput + 1 < goldilocksP
  chunkNoWrap : carryOut chunkInput 63 = false
  stepNoWrap : carryOut stepInput 63 = false

/-- One actual compiler run.  Each equation says that the indicated Rust
component interpreter produced the corresponding shared local columns. -/
structure ExecutionWitness
    (field : CanonicalU64Complete.FieldInverse)
    (assignment : Nat → Nat) where
  source : Source
  chunkInputExecuted :
    CanonicalU64Complete.interpret field source.chunkInput =
      pullAssignment assignment canonicalChunkInputMap
  stepInputExecuted :
    CanonicalU64Complete.interpret field source.stepInput =
      pullAssignment assignment canonicalStepInputMap
  chunkOutputExecuted :
    CanonicalU64Complete.interpret field (outputSource source.chunkInput) =
      pullAssignment assignment canonicalChunkOutputMap
  stepOutputExecuted :
    CanonicalU64Complete.interpret field (outputSource source.stepInput) =
      pullAssignment assignment canonicalStepOutputMap
  incrementExecuted :
    incrementInterpret source.chunkInput =
      pullAssignment assignment incrementColumnMap
  addOneExecuted :
    addOneInterpret source.stepInput =
      pullAssignment assignment addColumnMap

def outputEquationRows : List Row :=
  [⟨[(chunkOutputVarCol, 1),
      (FPrimeFullHistoryCounter.chunkInputVarCol, goldilocksP - 1),
      (0, goldilocksP - 1)], [(0, 1)], []⟩,
   ⟨[(stepOutputVarCol, 1),
      (FPrimeFullHistoryCounter.stepInputVarCol, goldilocksP - 1),
      (0, goldilocksP - 1)], [(0, 1)], []⟩]

/-- The two input decompositions are emitted before their source-field link,
so their five-row suffix has the generated order 65,66,67,68,64. -/
def inputCanonicalTail (rows : List Row) : List Row :=
  rows.drop 65 ++ (rows.drop 64).take 1

def rowsInChunkCompilerRows : List Row :=
  (List.range 64).map (fun index =>
    bitRow (addColumnMap (U64Add.rhsBitCol index))) ++
  (List.range 64).map (fun index =>
    constantRow (addColumnMap (U64Add.rhsBitCol index)) (expectedRowsBit index))

def compilerPieces : List (List Row) :=
  [(CanonicalU64.rows.map (renameRow canonicalChunkInputMap)).take 64,
   (CanonicalU64.rows.map (renameRow canonicalStepInputMap)).take 64,
   inputCanonicalTail (CanonicalU64.rows.map (renameRow canonicalChunkInputMap)),
   inputCanonicalTail (CanonicalU64.rows.map (renameRow canonicalStepInputMap)),
   outputEquationRows,
   CanonicalU64.rows.map (renameRow canonicalChunkOutputMap),
   (U64Increment.rows.map (renameRow incrementColumnMap)).drop 128,
   rowsInChunkCompilerRows,
   CanonicalU64.rows.map (renameRow canonicalStepOutputMap),
   (U64Add.rows.map (renameRow addColumnMap)).drop 192]

def compilerRows : List Row := compilerPieces.flatten

def splitRows : List Nat → List Row → List (List Row)
  | [], rows => [rows]
  | count :: counts, rows => rows.take count :: splitRows counts (rows.drop count)

theorem splitRows_flatten (counts : List Nat) (rows : List Row) :
    (splitRows counts rows).flatten = rows := by
  induction counts generalizing rows with
  | nil => simp [splitRows]
  | cons count counts inductionHypothesis =>
      simp only [splitRows, List.flatten_cons, inductionHypothesis]
      exact List.take_append_drop count rows

def artifactPieces : List (List Row) :=
  splitRows [64, 64, 5, 5, 2, 69, 127, 128, 69]
    FPrimeFullHistoryCounter.rows

private theorem piece0 : FPrimeFullHistoryCounter.rows.take 64 =
    compilerPieces.getD 0 [] := by native_decide
private theorem piece1 : (FPrimeFullHistoryCounter.rows.drop 64).take 64 =
    compilerPieces.getD 1 [] := by native_decide
private theorem piece2 : (FPrimeFullHistoryCounter.rows.drop 128).take 5 =
    compilerPieces.getD 2 [] := by native_decide
private theorem piece3 : (FPrimeFullHistoryCounter.rows.drop 133).take 5 =
    compilerPieces.getD 3 [] := by native_decide
private theorem piece4 : (FPrimeFullHistoryCounter.rows.drop 138).take 2 =
    compilerPieces.getD 4 [] := by native_decide
private theorem piece5 : (FPrimeFullHistoryCounter.rows.drop 140).take 69 =
    compilerPieces.getD 5 [] := by native_decide
private theorem piece6 : (FPrimeFullHistoryCounter.rows.drop 209).take 127 =
    compilerPieces.getD 6 [] := by native_decide
private theorem piece7 : (FPrimeFullHistoryCounter.rows.drop 336).take 128 =
    compilerPieces.getD 7 [] := by native_decide
private theorem piece8 : (FPrimeFullHistoryCounter.rows.drop 464).take 69 =
    compilerPieces.getD 8 [] := by native_decide
private theorem piece9 : FPrimeFullHistoryCounter.rows.drop 533 =
    compilerPieces.getD 9 [] := by native_decide

private theorem artifactPieces_eq : artifactPieces = compilerPieces := by
  simp only [artifactPieces, splitRows, compilerPieces]
  simp only [List.drop_drop]
  simp only [List.cons.injEq]
  exact ⟨piece0, piece1, piece2, piece3, piece4, piece5, piece6, piece7,
    piece8, piece9, True.intro⟩

/-- The exact 660 generated rows are precisely the compact compiler schedule.
The drift gate is split into ten bounded certificates so ordinary `lake build`
does not need to materialize one enormous equality proof. -/
theorem exactRows_eq : FPrimeFullHistoryCounter.rows = compilerRows := by
  calc
    FPrimeFullHistoryCounter.rows = artifactPieces.flatten :=
      (splitRows_flatten _ _).symm
    _ = compilerPieces.flatten := congrArg List.flatten artifactPieces_eq
    _ = compilerRows := rfl

private theorem mappedSatisfies_of_executed
    {sourceRows : List Row} {map : Nat → Nat}
    {sourceAssignment assignment : Nat → Nat}
    (sourceSatisfies : Satisfies sourceRows sourceAssignment)
    (executed : sourceAssignment = pullAssignment assignment map) :
    Satisfies (sourceRows.map (renameRow map)) assignment := by
  intro mapped mappedMember
  rcases List.mem_map.mp mappedMember with ⟨sourceRow, sourceMember, rfl⟩
  apply (rowHolds_pull_iff assignment map sourceRow).mp
  rw [← executed]
  exact sourceSatisfies sourceRow sourceMember

private theorem satisfies_take {rows : List Row} {assignment : Nat → Nat}
    (count : Nat) (satisfies : Satisfies rows assignment) :
    Satisfies (rows.take count) assignment := by
  intro row member
  exact satisfies row (List.mem_of_mem_take member)

private theorem satisfies_drop {rows : List Row} {assignment : Nat → Nat}
    (count : Nat) (satisfies : Satisfies rows assignment) :
    Satisfies (rows.drop count) assignment := by
  intro row member
  exact satisfies row (List.mem_of_mem_drop member)

private theorem satisfies_append {left right : List Row}
    {assignment : Nat → Nat}
    (leftSatisfies : Satisfies left assignment)
    (rightSatisfies : Satisfies right assignment) :
    Satisfies (left ++ right) assignment := by
  intro row member
  rcases List.mem_append.mp member with inLeft | inRight
  · exact leftSatisfies row inLeft
  · exact rightSatisfies row inRight

private theorem inputTail_complete {rows : List Row}
    {assignment : Nat → Nat} (satisfies : Satisfies rows assignment) :
    Satisfies (inputCanonicalTail rows) assignment := by
  apply satisfies_append
  · exact satisfies_drop 65 satisfies
  · exact satisfies_take 1 (satisfies_drop 64 satisfies)

private theorem sourceValues (prime : EuclidPrime goldilocksP)
    {field : CanonicalU64Complete.FieldInverse} {assignment : Nat → Nat}
    (witness : ExecutionWitness field assignment) :
    assignment 0 = 1 ∧
    assignment FPrimeFullHistoryCounter.chunkInputVarCol =
      CanonicalU64Complete.wordValue witness.source.chunkInput ∧
    assignment FPrimeFullHistoryCounter.stepInputVarCol =
      CanonicalU64Complete.wordValue witness.source.stepInput ∧
    assignment chunkOutputVarCol =
      CanonicalU64Complete.wordValue witness.source.chunkInput + 1 ∧
    assignment stepOutputVarCol =
      CanonicalU64Complete.wordValue witness.source.stepInput + 1 := by
  have one := congrFun witness.chunkInputExecuted 0
  have chunkIn := congrFun witness.chunkInputExecuted CanonicalU64.varCol
  have stepIn := congrFun witness.stepInputExecuted CanonicalU64.varCol
  have chunkOut := congrFun witness.chunkOutputExecuted CanonicalU64.varCol
  have stepOut := congrFun witness.stepOutputExecuted CanonicalU64.varCol
  have chunkWord := incrementOutput_word_eq prime witness.source.chunkInput
    witness.source.chunkNoWrap
  have stepWord := addOneOutput_word_eq prime witness.source.stepInput
    witness.source.stepNoWrap
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · simpa [CanonicalU64Complete.interpret, pullAssignment,
      canonicalChunkInputMap, columnMap] using one.symm
  · simpa [CanonicalU64Complete.interpret, pullAssignment,
      canonicalChunkInputMap, canonicalChunkInputMap, columnMap,
      CanonicalU64.varCol] using chunkIn.symm
  · simpa [CanonicalU64Complete.interpret, pullAssignment,
      canonicalStepInputMap, columnMap, CanonicalU64.varCol] using stepIn.symm
  · simpa [CanonicalU64Complete.interpret, pullAssignment,
      canonicalChunkOutputMap, columnMap, CanonicalU64.varCol, chunkWord] using
      chunkOut.symm
  · simpa [CanonicalU64Complete.interpret, pullAssignment,
      canonicalStepOutputMap, columnMap, CanonicalU64.varCol, stepWord] using
      stepOut.symm

private theorem outputEquations_complete (prime : EuclidPrime goldilocksP)
    {field : CanonicalU64Complete.FieldInverse} {assignment : Nat → Nat}
    (witness : ExecutionWitness field assignment) :
    Satisfies outputEquationRows assignment := by
  rcases sourceValues prime witness with ⟨one, chunkIn, stepIn, chunkOut, stepOut⟩
  have subtractIncrement (value : Nat) :
      (value + 1 + (goldilocksP - 1) * value + (goldilocksP - 1)) %
          goldilocksP = 0 := by
    have raw :
        value + 1 + (goldilocksP - 1) * value + (goldilocksP - 1) =
          goldilocksP * (value + 1) := by
      simp [goldilocksP]
      omega
    rw [raw]
    exact Nat.mul_mod_right _ _
  intro row member
  simp only [outputEquationRows, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · simpa [RowHolds, lcEval, one, chunkIn, chunkOut, goldilocksP] using
      subtractIncrement (CanonicalU64Complete.wordValue witness.source.chunkInput)
  · simpa [RowHolds, lcEval, one, stepIn, stepOut, goldilocksP] using
      subtractIncrement (CanonicalU64Complete.wordValue witness.source.stepInput)

private theorem rowsInChunk_complete
    {field : CanonicalU64Complete.FieldInverse} {assignment : Nat → Nat}
    (witness : ExecutionWitness field assignment) :
    Satisfies rowsInChunkConstraintRows assignment := by
  have oneExecuted := congrFun witness.addOneExecuted 0
  have one : assignment 0 = 1 := by
    simpa [addOneInterpret, pullAssignment, addColumnMap, columnMap] using
      oneExecuted.symm
  intro row member
  rcases List.mem_flatMap.mp member with ⟨index, indexMember, inPair⟩
  have bounded := List.mem_range.mp indexMember
  have executed := congrFun witness.addOneExecuted (U64Add.rhsBitCol index)
  have value :
      assignment (addColumnMap (U64Add.rhsBitCol index)) = expectedRowsBit index := by
    simpa [pullAssignment, expectedRowsBit, columnMap,
      addOneInterpret_rhs witness.source.stepInput bounded] using executed.symm
  simp only [List.mem_cons, List.not_mem_nil, or_false] at inPair
  rcases inPair with rfl | rfl
  · by_cases zero : index = 0
    · apply bitRow_bool_complete true
      · exact one
      · simpa [expectedRowsBit, zero, boolValue] using value
    · apply bitRow_bool_complete false
      · exact one
      · simpa [expectedRowsBit, zero, boolValue] using value
  · by_cases zero : index = 0
    · subst index
      have valueOne :
          assignment (addColumnMap (U64Add.rhsBitCol 0)) = 1 := by
        simpa [expectedRowsBit] using value
      simp [constantRow, expectedRowsBit, RowHolds, lcEval, valueOne,
        one, goldilocksP]
    · have valueZero :
          assignment (addColumnMap (U64Add.rhsBitCol index)) = 0 := by
        simpa [zero, expectedRowsBit] using value
      simp [constantRow, expectedRowsBit, zero, RowHolds, lcEval, valueZero,
        one, goldilocksP]

private theorem rowsInChunkCompiler_complete
    {field : CanonicalU64Complete.FieldInverse} {assignment : Nat → Nat}
    (witness : ExecutionWitness field assignment) :
    Satisfies rowsInChunkCompilerRows assignment := by
  have interleaved := rowsInChunk_complete witness
  intro row member
  rcases List.mem_append.mp member with inBits | inConstants
  · rcases List.mem_map.mp inBits with ⟨index, indexMember, rfl⟩
    apply interleaved
    apply List.mem_flatMap.mpr
    exact ⟨index, indexMember, by simp⟩
  · rcases List.mem_map.mp inConstants with ⟨index, indexMember, rfl⟩
    apply interleaved
    apply List.mem_flatMap.mpr
    exact ⟨index, indexMember, by simp⟩

/-- Honest compiler completeness of all exact 660 local counter rows. -/
theorem complete (prime : EuclidPrime goldilocksP)
    {field : CanonicalU64Complete.FieldInverse} {assignment : Nat → Nat}
    (witness : ExecutionWitness field assignment) :
    Satisfies FPrimeFullHistoryCounter.rows assignment := by
  have chunkIn := mappedSatisfies_of_executed
    (CanonicalU64Complete.complete field witness.source.chunkInput
      witness.source.chunkInputCanonical)
    witness.chunkInputExecuted
  have stepIn := mappedSatisfies_of_executed
    (CanonicalU64Complete.complete field witness.source.stepInput
      witness.source.stepInputCanonical)
    witness.stepInputExecuted
  have chunkOutputCanonical :
      CanonicalU64Complete.wordValue (outputSource witness.source.chunkInput) <
        goldilocksP := by
    rw [incrementOutput_word_eq prime witness.source.chunkInput
      witness.source.chunkNoWrap]
    exact witness.source.chunkOutputCanonical
  have stepOutputCanonical :
      CanonicalU64Complete.wordValue (outputSource witness.source.stepInput) <
        goldilocksP := by
    rw [addOneOutput_word_eq prime witness.source.stepInput
      witness.source.stepNoWrap]
    exact witness.source.stepOutputCanonical
  have chunkOut := mappedSatisfies_of_executed
    (CanonicalU64Complete.complete field (outputSource witness.source.chunkInput)
      chunkOutputCanonical)
    witness.chunkOutputExecuted
  have stepOut := mappedSatisfies_of_executed
    (CanonicalU64Complete.complete field (outputSource witness.source.stepInput)
      stepOutputCanonical)
    witness.stepOutputExecuted
  have increment := mappedSatisfies_of_executed
    (increment_complete witness.source.chunkInput witness.source.chunkNoWrap)
    witness.incrementExecuted
  have addOne := mappedSatisfies_of_executed
    (addOne_complete witness.source.stepInput witness.source.stepNoWrap)
    witness.addOneExecuted
  have pieces : ∀ piece ∈ compilerPieces, Satisfies piece assignment := by
    intro piece member
    simp only [compilerPieces, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    · exact satisfies_take 64 chunkIn
    · exact satisfies_take 64 stepIn
    · exact inputTail_complete chunkIn
    · exact inputTail_complete stepIn
    · exact outputEquations_complete prime witness
    · exact chunkOut
    · exact satisfies_drop 128 increment
    · exact rowsInChunkCompiler_complete witness
    · exact stepOut
    · exact satisfies_drop 192 addOne
  have all : Satisfies compilerRows assignment :=
    (satisfies_flatten_iff compilerPieces assignment).mpr pieces
  rw [exactRows_eq]
  exact all

end Compiler

end FPrimeFullHistoryCounterLocalSound

end Nightstream.Implementation.R1CS
