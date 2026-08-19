import Nightstream.Implementation.R1CS.Correspondence.CanonicalU64.CanonicalU64Sound
import Nightstream.Implementation.R1CS.Correspondence.U64.U64IncrementSound
import Nightstream.Implementation.R1CS.Correspondence.U64.U64AddSound
import Nightstream.Implementation.R1CS.Artifacts.FPrime

/-!
Contract: artifact-level soundness of the production-used recursive F' counter
block. Property `CIR-FPR-COUNTER`.

The 660 exported rows bind source-image words to the incoming field counters,
enforce canonical decompositions of both outputs, pin the batch size to seven,
and enforce no-wrap increment/addition. The proof below is quantified over
every canonical-residue assignment satisfying those exact rows. Its small
row-inclusion certificates project the large block onto the previously proved
canonical-u64, increment, and addition row programs.
-/

set_option maxRecDepth 32768
set_option maxHeartbeats 4000000

namespace Nightstream.Implementation.R1CS

namespace FPrimeCounterSound

def columnMap (columns : List Nat) : Nat → Nat :=
  fun i => columns.getD i 0

def canonicalChunkInputMap : Nat → Nat := columnMap FPrimeCounter.chunkInputCanonicalMap
def canonicalStepInputMap : Nat → Nat := columnMap FPrimeCounter.stepInputCanonicalMap
def canonicalChunkOutputMap : Nat → Nat := columnMap FPrimeCounter.chunkOutputCanonicalMap
def canonicalStepOutputMap : Nat → Nat := columnMap FPrimeCounter.stepOutputCanonicalMap
def incrementColumnMap : Nat → Nat := columnMap FPrimeCounter.incrementMap
def addColumnMap : Nat → Nat := columnMap FPrimeCounter.addMap

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
      FPrimeCounter.rows = true := by
  decide

private theorem stepInputCanonicalRowsIncluded :
    rowsIncluded
      (CanonicalU64.rows.map (renameRow canonicalStepInputMap))
      FPrimeCounter.rows = true := by
  decide

private theorem chunkOutputCanonicalRowsIncluded :
    rowsIncluded
      (CanonicalU64.rows.map (renameRow canonicalChunkOutputMap))
      FPrimeCounter.rows = true := by
  decide

private theorem stepOutputCanonicalRowsIncluded :
    rowsIncluded
      (CanonicalU64.rows.map (renameRow canonicalStepOutputMap))
      FPrimeCounter.rows = true := by
  decide

private theorem incrementRowsIncluded :
    rowsIncluded
      (U64Increment.rows.map (renameRow incrementColumnMap))
      FPrimeCounter.rows = true := by
  decide

private theorem addRowsIncluded :
    rowsIncluded
      (U64Add.rows.map (renameRow addColumnMap))
      FPrimeCounter.rows = true := by
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

def expectedRowsBit (i : Nat) : Nat := if i < 3 then 1 else 0

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
    rowsIncluded rowsInChunkConstraintRows FPrimeCounter.rows = true := by
  decide

private theorem rowsInChunk_bit
    (hq : EuclidPrime goldilocksP) {z : Nat → Nat}
    (hcanon : ∀ i, z i < goldilocksP) (hone : z 0 = 1)
    (hsat : Satisfies FPrimeCounter.rows z) (i : Nat) (hi : i < 64) :
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
  by_cases hsmall : i < 3 <;>
    rcases hz with hz | hz <;>
    simp_all [col, expectedRowsBit, constantRow, RowHolds, lcEval,
      goldilocksP]

private theorem rowsInChunkValue_eq
    (hq : EuclidPrime goldilocksP) {z : Nat → Nat}
    (hcanon : ∀ i, z i < goldilocksP) (hone : z 0 = 1)
    (hsat : Satisfies FPrimeCounter.rows z) :
    rowsInChunkValue z = FPrimeCounter.rowsInChunk := by
  have hbits : ∀ i, i < 64 →
      z (addColumnMap (U64Add.rhsBitCol i)) = expectedRowsBit i :=
    fun i hi => rowsInChunk_bit hq hcanon hone hsat i hi
  calc
    rowsInChunkValue z = bitVectorValue expectedRowsBit := by
      apply bitVectorValue_congr
      exact hbits
    _ = FPrimeCounter.rowsInChunk := by decide

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
theorem fPrimeCounter_sound (hq : EuclidPrime goldilocksP)
    {z : Nat → Nat}
    (hcanon : ∀ i, z i < goldilocksP)
    (hone : z 0 = 1)
    (hsat : Satisfies FPrimeCounter.rows z) :
    z FPrimeCounter.chunkInputVarCol = chunkInputValue z ∧
    z FPrimeCounter.stepInputVarCol = stepInputValue z ∧
    z chunkOutputVarCol = chunkOutputValue z ∧
    z stepOutputVarCol = stepOutputValue z ∧
    chunkOutputValue z = chunkInputValue z + 1 ∧
    stepOutputValue z = stepInputValue z + FPrimeCounter.rowsInChunk ∧
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
    (pulledOne hone FPrimeCounter.chunkInputCanonicalMap (by decide)) hChunkInSat
  have hStepInCanonical := canonicalU64_sound hq
    (pulledCanonical hcanon canonicalStepInputMap)
    (pulledOne hone FPrimeCounter.stepInputCanonicalMap (by decide)) hStepInSat
  have hChunkOutCanonical := canonicalU64_sound hq
    (pulledCanonical hcanon canonicalChunkOutputMap)
    (pulledOne hone FPrimeCounter.chunkOutputCanonicalMap (by decide)) hChunkOutSat
  have hStepOutCanonical := canonicalU64_sound hq
    (pulledCanonical hcanon canonicalStepOutputMap)
    (pulledOne hone FPrimeCounter.stepOutputCanonicalMap (by decide)) hStepOutSat
  have hIncrement := u64Increment_sound hq
    (pulledCanonical hcanon incrementColumnMap)
    (pulledOne hone FPrimeCounter.incrementMap (by decide)) hIncrementSat
  have hAdd := u64Add_sound hq
    (pulledCanonical hcanon addColumnMap)
    (pulledOne hone FPrimeCounter.addMap (by decide)) hAddSat
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
  · simpa [canonicalChunkInputMap, columnMap, FPrimeCounter.chunkInputVarCol,
      CanonicalU64.varCol, hChunkInBits] using hChunkInCanonical.1
  · simpa [canonicalStepInputMap, columnMap, FPrimeCounter.stepInputVarCol,
      CanonicalU64.varCol, hStepInBits] using hStepInCanonical.1
  · simpa [chunkOutputVarCol, hChunkOutBits] using hChunkOutCanonical.1
  · simpa [stepOutputVarCol, hStepOutBits] using hStepOutCanonical.1
  · simpa [hRows] using hStepAdd
  · simpa [hChunkInBits] using hChunkInCanonical.2
  · simpa [hStepInBits] using hStepInCanonical.2
  · simpa [hChunkOutBits] using hChunkOutCanonical.2
  · simpa [hStepOutBits] using hStepOutCanonical.2

end FPrimeCounterSound

end Nightstream.Implementation.R1CS
