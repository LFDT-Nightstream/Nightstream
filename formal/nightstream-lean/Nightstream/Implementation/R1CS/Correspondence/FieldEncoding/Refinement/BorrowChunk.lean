import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.Refinement.DerivedBorrow

/-!
Exact two-step composition of the shifted-radix-3 canonicality transition.

The outer `b = 2` norm owns digit membership in `{-1, 0, 1}`. This module owns
only the deterministic comparison transition, its two-step composition, and
the degree/count bounds needed by the fixed 13-port F′ CCS image.
-/

namespace Nightstream.Implementation.R1CS.CenteredTernaryBorrowChunk

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler
open Nightstream.Implementation.R1CS.ShiftedTernarySound
open Nightstream.Implementation.R1CS.CenteredTernaryField
open Nightstream.Implementation.R1CS.CenteredTernaryNormDischarged
open Nightstream.Implementation.R1CS.CenteredTernaryDerivedNegative
open Nightstream.Implementation.R1CS.CenteredTernaryDerivedBorrow

abbrev Polynomial :=
  Nightstream.Implementation.R1CS.CenteredTernaryDerivedBorrow.Polynomial

def polyNeg (value : Polynomial) : Polynomial :=
  .mul (.constant (goldilocksP - 1)) value

def polySub (left right : Polynomial) : Polynomial :=
  .add left (polyNeg right)

def polyOneMinus (value : Polynomial) : Polynomial :=
  polySub (.constant 1) value

/-- Quadratic negative-digit indicator for an arbitrary polynomial digit. -/
def negativePolynomial (digit : Polynomial) : Polynomial :=
  .mul
    (.mul digit (.add digit (.constant (goldilocksP - 1))))
    (.constant inverseTwo)

/-- Indicator of the centered digit `+1`. -/
def positivePolynomial (digit : Polynomial) : Polynomial :=
  .add digit (negativePolynomial digit)

/-- Indicator of the centered digit `0`. -/
def zeroPolynomial (digit : Polynomial) : Polynomial :=
  polySub (.constant 1)
    (.add digit (.mul (.constant 2) (negativePolynomial digit)))

/-- Solved next-borrow polynomial for one fixed bound trit. -/
def stepPolynomial (bound : Nat) (digit borrow : Polynomial) : Polynomial :=
  match bound with
  | 0 =>
      polySub (.constant 1)
        (.mul (negativePolynomial digit) (polyOneMinus borrow))
  | 1 =>
      .add (positivePolynomial digit)
        (.mul (zeroPolynomial digit) borrow)
  | _ =>
      .mul (positivePolynomial digit) borrow

/-- Compose a fixed sequence of `(bound trit, digit column)` transitions. -/
def composePolynomial :
    List (Nat × Nat) → Polynomial → Polynomial
  | [], borrow => borrow
  | (bound, digitColumn) :: tail, borrow =>
      composePolynomial tail
        (stepPolynomial bound (.variable digitColumn) borrow)

def chunkCount : Nat := 21
def chunkWidth : Nat := 2
def chunkBorrowCount : Nat := chunkCount - 1
def chunkBorrowColumnBase : Nat := 99

def chunkStart (chunk : Nat) : Nat :=
  chunk * chunkWidth

def chunkLength (chunk : Nat) : Nat :=
  min chunkWidth (digitCount - chunkStart chunk)

def chunkEntries (chunk : Nat) : List (Nat × Nat) :=
  (List.range (chunkLength chunk)).map fun offset =>
    let index := chunkStart chunk + offset
    (boundDigits.getD index 0,
      ShiftedTernary.digitCols.getD index 0)

def chunkInput (chunk : Nat) : Polynomial :=
  if chunk = 0 then .constant 0
  else .variable (chunkBorrowColumnBase + chunk - 1)

def chunkOutput (chunk : Nat) : Polynomial :=
  if chunk + 1 = chunkCount then .constant 0
  else .variable (chunkBorrowColumnBase + chunk)

def chunkEquation (chunk : Nat) :
    Nightstream.Implementation.R1CS.CenteredTernaryDerivedBorrow.Equation where
  left := chunkOutput chunk
  right := composePolynomial (chunkEntries chunk) (chunkInput chunk)

def chunkEquations :
    List Nightstream.Implementation.R1CS.CenteredTernaryDerivedBorrow.Equation :=
  (List.range chunkCount).map chunkEquation

theorem chunkEquations_length :
    chunkEquations.length = 21 := by
  decide

theorem chunkBorrowCount_eq :
    chunkBorrowCount = 20 := by
  decide

theorem chunkLengths :
    (List.range chunkCount).map chunkLength =
      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1] := by
  decide

theorem chunkEntries_cover_41 :
    (List.range chunkCount).foldl
      (fun count chunk => count + (chunkEntries chunk).length) 0 = 41 := by
  decide

def chunkIndices (chunk : Nat) : List Nat :=
  (List.range (chunkLength chunk)).map fun offset =>
    chunkStart chunk + offset

def scheduledDigitIndices : List Nat :=
  (List.range chunkCount).flatMap chunkIndices

theorem scheduledDigitIndices_eq_range :
    scheduledDigitIndices = List.range digitCount := by
  decide

def entriesPrefix (count : Nat) : List (Nat × Nat) :=
  (List.range count).map fun index =>
    (boundDigits.getD index 0,
      ShiftedTernary.digitCols.getD index 0)

def allEntries : List (Nat × Nat) :=
  entriesPrefix digitCount

theorem entriesPrefix_succ (count : Nat) :
    entriesPrefix (count + 1) =
      entriesPrefix count ++
        [(boundDigits.getD count 0,
          ShiftedTernary.digitCols.getD count 0)] := by
  simp [entriesPrefix, List.range_succ]

def scheduledEntries : List (Nat × Nat) :=
  (List.range chunkCount).flatMap chunkEntries

theorem scheduledEntries_eq_allEntries :
    scheduledEntries = allEntries := by
  decide

def prefixEntries (chunks : Nat) : List (Nat × Nat) :=
  (List.range chunks).flatMap chunkEntries

theorem prefixEntries_succ (chunks : Nat) :
    prefixEntries (chunks + 1) =
      prefixEntries chunks ++ chunkEntries chunks := by
  simp [prefixEntries, List.range_succ]

theorem scheduledEntries_last :
    scheduledEntries =
      prefixEntries (chunkCount - 1) ++
        chunkEntries (chunkCount - 1) := by
  decide

def chunkBoundValue (chunk : Nat) : Nat :=
  let start := chunkStart chunk
  boundDigits.getD start 0 +
    3 * boundDigits.getD (start + 1) 0

def normalizedChunkBound (chunk : Nat) : Nat :=
  min (chunkBoundValue chunk) (8 - chunkBoundValue chunk)

theorem normalizedChunkBounds :
    (List.range chunkCount).map normalizedChunkBound =
      [3, 0, 3, 3, 3, 0, 1, 3, 1, 2, 4, 3, 2, 1, 3, 0, 0, 0, 3, 4, 1] := by
  decide

theorem normalizedChunkBound_lt_five
    {chunk : Nat} (chunkLt : chunk < chunkCount) :
    normalizedChunkBound chunk < 5 := by
  have member :
      normalizedChunkBound chunk ∈
        (List.range chunkCount).map normalizedChunkBound :=
    List.mem_map.mpr ⟨chunk, List.mem_range.mpr chunkLt, rfl⟩
  rw [normalizedChunkBounds] at member
  simp at member
  omega

/-- Every two-step equation has degree at most five. -/
theorem chunkEquations_degree_le_five :
    ∀ equation ∈ chunkEquations, equation.degree ≤ 5 := by
  decide

def maximumChunkDegree : Nat :=
  chunkEquations.foldl
    (fun maximum equation => max maximum equation.degree) 0

theorem maximumChunkDegree_eq_five :
    maximumChunkDegree = 5 := by
  decide

/-- A branch selector and one fixed-bound-class selector keep the uniform
13-port polynomial below degree eight. -/
def uniformSelectorGatedDegree
    (equation :
      Nightstream.Implementation.R1CS.CenteredTernaryDerivedBorrow.Equation) :
    Nat :=
  2 + equation.degree

theorem uniformSelectorGatedDegrees_le_seven :
    ∀ equation ∈ chunkEquations,
      uniformSelectorGatedDegree equation ≤ 7 := by
  decide

def openingCoordinateCount : Nat :=
  digitCount + chunkBorrowCount

def rankTwoMapRows : Nat := 108
def rankTwoOutputCoordinates : Nat := 108 * digitCount
def rankOneMapRows : Nat := 54
def rankTwoOutputFields : Nat := 108
def rankOneOutputCoordinates : Nat := 54 * digitCount

/-- Shared input encoding is paid once; each distinct rank-two output is paid
once per commitment. -/
def rowsForRankTwoCommitments
    (sourceFields commitments : Nat) : Nat :=
  chunkCount * sourceFields + rankTwoMapRows * commitments

def coordinatesForRankTwoCommitments
    (sourceFields commitments : Nat) : Nat :=
  openingCoordinateCount * sourceFields +
    rankTwoOutputCoordinates * commitments

/-- Complete rank-two → rank-one digest chains. Source openings are shared
across chains; each chain canonically reopens its 108 rank-two outputs and
retains 54 rank-one output words. -/
def rowsForRankTwoDigestChains
    (sourceFields chains : Nat) : Nat :=
  chunkCount * (sourceFields + rankTwoOutputFields * chains) +
    (rankTwoMapRows + rankOneMapRows) * chains

def coordinatesForRankTwoDigestChains
    (sourceFields chains : Nat) : Nat :=
  openingCoordinateCount *
      (sourceFields + rankTwoOutputFields * chains) +
    rankOneOutputCoordinates * chains

theorem openingCoordinateCount_eq :
    openingCoordinateCount = 61 := by
  decide

theorem activeProfile_oneCommitment_rows :
    rowsForRankTwoCommitments 23033 1 = 483801 := by
  decide

theorem activeProfile_oneCommitment_coordinates :
    coordinatesForRankTwoCommitments 23033 1 = 1409441 := by
  decide

theorem activeProfile_oneDigestChain_rows :
    rowsForRankTwoDigestChains 23033 1 = 486123 := by
  decide

theorem activeProfile_oneDigestChain_coordinates :
    coordinatesForRankTwoDigestChains 23033 1 = 1413815 := by
  decide

theorem firstChunkInput :
    chunkInput 0 = .constant 0 := by
  rfl

theorem lastChunkOutput :
    chunkOutput (chunkCount - 1) = .constant 0 := by
  decide

theorem adjacentChunkBorrowAlias
    {chunk : Nat} (chunkLt : chunk + 1 < chunkCount) :
    chunkOutput chunk = chunkInput (chunk + 1) := by
  unfold chunkOutput chunkInput
  rw [if_neg (by omega), if_neg (by omega)]
  congr 1

/-- Ordinary comparator transition on a trit and Boolean incoming borrow. -/
def scalarStep (bound trit borrow : Nat) : Nat :=
  if trit + borrow > bound then 1 else 0

theorem scalarStep_le_one
    (bound trit borrow : Nat) :
    scalarStep bound trit borrow ≤ 1 := by
  unfold scalarStep
  split <;> omega

def scalarTwoValues
    (boundZero boundOne tritZero tritOne borrow : Nat) : Nat :=
  scalarStep boundOne tritOne
    (scalarStep boundZero tritZero borrow)

/-- Complementing both radix-3 digits and both endpoint borrows changes a
base-9 bound `H` into `8-H`. This reduces all rows to five fixed classes. -/
theorem scalarTwoValues_complement
    {boundZero boundOne tritZero tritOne borrow : Nat}
    (boundZeroLt : boundZero < 3)
    (boundOneLt : boundOne < 3)
    (tritZeroLt : tritZero < 3)
    (tritOneLt : tritOne < 3)
    (borrowLe : borrow ≤ 1) :
    1 - scalarTwoValues boundZero boundOne tritZero tritOne borrow =
      scalarTwoValues
        (2 - boundZero) (2 - boundOne)
        (2 - tritZero) (2 - tritOne) (1 - borrow) := by
  have boundZeroCases :
      boundZero = 0 ∨ boundZero = 1 ∨ boundZero = 2 := by omega
  have boundOneCases :
      boundOne = 0 ∨ boundOne = 1 ∨ boundOne = 2 := by omega
  have tritZeroCases :
      tritZero = 0 ∨ tritZero = 1 ∨ tritZero = 2 := by omega
  have tritOneCases :
      tritOne = 0 ∨ tritOne = 1 ∨ tritOne = 2 := by omega
  have borrowCases : borrow = 0 ∨ borrow = 1 := by omega
  rcases boundZeroCases with rfl | rfl | rfl <;>
    rcases boundOneCases with rfl | rfl | rfl <;>
    rcases tritZeroCases with rfl | rfl | rfl <;>
    rcases tritOneCases with rfl | rfl | rfl <;>
    rcases borrowCases with rfl | rfl <;>
    decide

/-- On centered digits and a Boolean incoming borrow, the polynomial is the
ordinary comparator transition. -/
theorem eval_stepPolynomial_eq_scalar
    (assignment : Nat → Nat)
    {bound : Nat} (boundLt : bound < 3)
    (digit borrow : Polynomial)
    (centered : CenteredResidue (digit.eval assignment))
    (borrowLe : borrow.eval assignment ≤ 1) :
    (stepPolynomial bound digit borrow).eval assignment =
      scalarStep bound (tritValue (digit.eval assignment))
        (borrow.eval assignment) := by
  have boundCases : bound = 0 ∨ bound = 1 ∨ bound = 2 := by
    omega
  have borrowCases :
      borrow.eval assignment = 0 ∨ borrow.eval assignment = 1 := by
    omega
  rcases centered with digitEq | digitEq | digitEq <;>
    rcases boundCases with boundEq | boundEq | boundEq <;>
    rcases borrowCases with borrowEq | borrowEq <;>
    subst bound
  all_goals
    simp [stepPolynomial, negativePolynomial, positivePolynomial,
      zeroPolynomial, polySub, polyNeg, polyOneMinus,
      Polynomial.eval, scalarStep, tritValue, digitEq, borrowEq,
      inverseTwo, goldilocksP]

/-- Scalar execution of an arbitrary fixed chunk. -/
def scalarCompose (assignment : Nat → Nat) :
    List (Nat × Nat) → Nat → Nat
  | [], borrow => borrow
  | (bound, digitColumn) :: tail, borrow =>
      scalarCompose assignment tail
        (scalarStep bound
          (tritValue (assignment digitColumn % goldilocksP)) borrow)

theorem scalarCompose_append
    (assignment : Nat → Nat)
    (left right : List (Nat × Nat)) (borrow : Nat) :
    scalarCompose assignment (left ++ right) borrow =
      scalarCompose assignment right
        (scalarCompose assignment left borrow) := by
  induction left generalizing borrow with
  | nil => rfl
  | cons entry tail inductionHypothesis =>
      rcases entry with ⟨bound, digitColumn⟩
      simp only [List.cons_append, scalarCompose]
      exact inductionHypothesis _

theorem scalarCompose_le_one
    (assignment : Nat → Nat)
    (entries : List (Nat × Nat)) {borrow : Nat}
    (borrowLe : borrow ≤ 1) :
    scalarCompose assignment entries borrow ≤ 1 := by
  induction entries generalizing borrow with
  | nil => exact borrowLe
  | cons entry tail inductionHypothesis =>
      rcases entry with ⟨bound, digitColumn⟩
      simp only [scalarCompose]
      exact inductionHypothesis (scalarStep_le_one _ _ _)

theorem eval_composePolynomial_eq_scalar
    (assignment : Nat → Nat)
    (entries : List (Nat × Nat))
    (borrow : Polynomial)
    (bounds : ∀ entry ∈ entries, entry.1 < 3)
    (centered : ∀ entry ∈ entries,
      CenteredResidue (assignment entry.2 % goldilocksP))
    (borrowLe : borrow.eval assignment ≤ 1) :
    (composePolynomial entries borrow).eval assignment =
      scalarCompose assignment entries (borrow.eval assignment) := by
  induction entries generalizing borrow with
  | nil => rfl
  | cons entry tail inductionHypothesis =>
      rcases entry with ⟨bound, digitColumn⟩
      have boundLt : bound < 3 :=
        bounds (bound, digitColumn) (by simp)
      have digitCentered :
          CenteredResidue (assignment digitColumn % goldilocksP) :=
        centered (bound, digitColumn) (by simp)
      have stepEq :=
        eval_stepPolynomial_eq_scalar assignment boundLt
          (.variable digitColumn) borrow digitCentered borrowLe
      simp only [Polynomial.eval] at stepEq
      have nextLe :
          (stepPolynomial bound (.variable digitColumn) borrow).eval
              assignment ≤ 1 := by
        rw [stepEq]
        exact scalarStep_le_one _ _ _
      rw [composePolynomial,
        inductionHypothesis
          (stepPolynomial bound (.variable digitColumn) borrow)
          (fun next nextMember =>
            bounds next (by simp [nextMember]))
          (fun next nextMember =>
            centered next (by simp [nextMember]))
          nextLe,
        scalarCompose, stepEq]

/-- The concrete Goldilocks chunks use only radix-3 bound digits. -/
theorem chunkEntry_bound_lt_three
    {chunk : Nat} (chunkLt : chunk < chunkCount) :
    ∀ entry ∈ chunkEntries chunk, entry.1 < 3 := by
  intro entry entryMember
  rcases List.mem_map.mp entryMember with
    ⟨offset, offsetMember, entryEq⟩
  have offsetLt : offset < chunkLength chunk :=
    List.mem_range.mp offsetMember
  have indexLt : chunkStart chunk + offset < digitCount := by
    unfold chunkLength chunkStart chunkCount chunkWidth digitCount at *
    omega
  rw [← entryEq]
  exact Nightstream.Implementation.R1CS.ShiftedTernarySound.boundDigit_lt_three
    indexLt

def ChunkDigitsCentered (assignment : Nat → Nat) (chunk : Nat) : Prop :=
  ∀ entry ∈ chunkEntries chunk,
    CenteredResidue (assignment entry.2 % goldilocksP)

def chunkInputValue (assignment : Nat → Nat) (chunk : Nat) : Nat :=
  (chunkInput chunk).eval assignment

def chunkOutputValue (assignment : Nat → Nat) (chunk : Nat) : Nat :=
  (chunkOutput chunk).eval assignment

def scalarChunkValue (assignment : Nat → Nat) (chunk : Nat) : Nat :=
  scalarCompose assignment (chunkEntries chunk)
    (chunkInputValue assignment chunk)

/-- Each emitted chunk equation is exactly its scalar transition chain. -/
theorem chunkEquation_holds_iff_scalar
    (assignment : Nat → Nat)
    {chunk : Nat} (chunkLt : chunk < chunkCount)
    (centered : ChunkDigitsCentered assignment chunk)
    (inputLe : chunkInputValue assignment chunk ≤ 1) :
    (chunkEquation chunk).Holds assignment ↔
      chunkOutputValue assignment chunk =
        scalarChunkValue assignment chunk := by
  unfold chunkEquation
  simp only [
    Nightstream.Implementation.R1CS.CenteredTernaryDerivedBorrow.Equation.Holds,
    chunkOutputValue, scalarChunkValue, chunkInputValue]
  rw [eval_composePolynomial_eq_scalar assignment
    (chunkEntries chunk) (chunkInput chunk)
    (chunkEntry_bound_lt_three chunkLt) centered inputLe]

def ChunkScheduleHolds (assignment : Nat → Nat) : Prop :=
  ∀ chunk, chunk < chunkCount →
    (chunkEquation chunk).Holds assignment

theorem chunkDigitsCentered_of_norm
    {assignment : Nat → Nat}
    (norm : DigitNormBoundTwo assignment)
    {chunk : Nat} (chunkLt : chunk < chunkCount) :
    ChunkDigitsCentered assignment chunk := by
  intro entry entryMember
  rcases List.mem_map.mp entryMember with
    ⟨offset, offsetMember, entryEq⟩
  have offsetLt : offset < chunkLength chunk :=
    List.mem_range.mp offsetMember
  let index := chunkStart chunk + offset
  have indexLt : index < digitCount := by
    unfold index chunkLength chunkStart chunkCount chunkWidth digitCount at *
    omega
  have bounded := norm index indexLt
  have centered :
      CenteredResidue
        (assignment
          (ShiftedTernary.digitCols.getD index 0)) :=
    normBoundTwo_iff_centeredResidue.mp bounded
  rw [← entryEq]
  change CenteredResidue
    (assignment (ShiftedTernary.digitCols.getD index 0) % goldilocksP)
  rw [Nat.mod_eq_of_lt bounded.1]
  exact centered

theorem chunkInput_eq_prefix
    {assignment : Nat → Nat}
    (norm : DigitNormBoundTwo assignment)
    (holds : ChunkScheduleHolds assignment) :
    ∀ chunk, chunk < chunkCount →
      chunkInputValue assignment chunk =
        scalarCompose assignment (prefixEntries chunk) 0 := by
  intro chunk chunkLt
  induction chunk with
  | zero =>
      simp [chunkInputValue, chunkInput, prefixEntries,
        scalarCompose, Polynomial.eval]
  | succ previous inductionHypothesis =>
      have previousLt : previous < chunkCount := by omega
      have previousHasNext : previous + 1 < chunkCount := by omega
      have inputEq := inductionHypothesis previousLt
      have inputLe : chunkInputValue assignment previous ≤ 1 := by
        rw [inputEq]
        exact scalarCompose_le_one assignment
          (prefixEntries previous) (by omega)
      have transition :=
        (chunkEquation_holds_iff_scalar assignment previousLt
          (chunkDigitsCentered_of_norm norm previousLt) inputLe).mp
          (holds previous previousLt)
      have aliasEval := congrArg
        (Nightstream.Implementation.R1CS.CenteredTernaryDerivedBorrow.Polynomial.eval
          assignment)
        (adjacentChunkBorrowAlias previousHasNext)
      calc
        chunkInputValue assignment (Nat.succ previous) =
            chunkOutputValue assignment previous := by
              simpa [chunkInputValue, chunkOutputValue] using aliasEval.symm
        _ = scalarCompose assignment (chunkEntries previous)
              (chunkInputValue assignment previous) := transition
        _ = scalarCompose assignment (chunkEntries previous)
              (scalarCompose assignment (prefixEntries previous) 0) := by
                rw [inputEq]
        _ = scalarCompose assignment
              (prefixEntries previous ++ chunkEntries previous) 0 := by
                rw [scalarCompose_append]
        _ = scalarCompose assignment
              (prefixEntries (Nat.succ previous)) 0 := by
                rw [show Nat.succ previous = previous + 1 by omega,
                  prefixEntries_succ]

theorem chunkSchedule_final_scalar_zero
    {assignment : Nat → Nat}
    (norm : DigitNormBoundTwo assignment)
    (holds : ChunkScheduleHolds assignment) :
    scalarCompose assignment scheduledEntries 0 = 0 := by
  let last := chunkCount - 1
  have lastLt : last < chunkCount := by
    unfold last chunkCount
    omega
  have inputEq :=
    chunkInput_eq_prefix norm holds last lastLt
  have inputLe : chunkInputValue assignment last ≤ 1 := by
    rw [inputEq]
    exact scalarCompose_le_one assignment
      (prefixEntries last) (by omega)
  have transition :=
    (chunkEquation_holds_iff_scalar assignment lastLt
      (chunkDigitsCentered_of_norm norm lastLt) inputLe).mp
      (holds last lastLt)
  have outputZero : chunkOutputValue assignment last = 0 := by
    unfold chunkOutputValue
    rw [show chunkOutput last = .constant 0 by
      simpa [last] using lastChunkOutput]
    simp [Polynomial.eval]
  calc
    scalarCompose assignment scheduledEntries 0 =
        scalarCompose assignment
          (prefixEntries last ++ chunkEntries last) 0 := by
            rw [scheduledEntries_last]
    _ = scalarCompose assignment (chunkEntries last)
          (scalarCompose assignment (prefixEntries last) 0) :=
            scalarCompose_append assignment _ _ _
    _ = scalarCompose assignment (chunkEntries last)
          (chunkInputValue assignment last) := by rw [inputEq]
    _ = chunkOutputValue assignment last := transition.symm
    _ = 0 := outputZero

def assignmentTritMod (assignment : Nat → Nat) (index : Nat) : Nat :=
  tritValue
    (assignment (ShiftedTernary.digitCols.getD index 0) % goldilocksP)

theorem scalarCompose_entriesPrefix
    (assignment : Nat → Nat) (count : Nat) :
    scalarCompose assignment (entriesPrefix count) 0 =
      expectedBorrow (assignmentTritMod assignment) boundDigit 0 count := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [entriesPrefix_succ, scalarCompose_append,
        inductionHypothesis]
      simp [scalarCompose, expectedBorrow, assignmentTritMod,
        boundDigit, scalarStep]

theorem chunkSchedule_expectedBorrow_zero
    {assignment : Nat → Nat}
    (norm : DigitNormBoundTwo assignment)
    (holds : ChunkScheduleHolds assignment) :
    expectedBorrow
      (assignmentTritMod assignment) boundDigit 0 digitCount = 0 := by
  rw [← scalarCompose_entriesPrefix assignment digitCount,
    ← show allEntries = entriesPrefix digitCount by rfl,
    ← scheduledEntries_eq_allEntries]
  exact chunkSchedule_final_scalar_zero norm holds

theorem assignmentTritMod_lt_three
    {assignment : Nat → Nat}
    (norm : DigitNormBoundTwo assignment) :
    ∀ index, index < digitCount →
      assignmentTritMod assignment index < 3 := by
  intro index indexLt
  have bounded := norm index indexLt
  have centered :=
    normBoundTwo_iff_centeredResidue.mp bounded
  unfold assignmentTritMod
  rw [Nat.mod_eq_of_lt bounded.1]
  exact Digit.tritValue_lt_three
    (digit_of_centeredResidue centered)

/-- The complete 21-row schedule plus the outer `b = 2` norm forces the
ordinary 41-trit integer below the Goldilocks modulus. -/
theorem chunkSchedule_encoded_lt_modulus
    {assignment : Nat → Nat}
    (norm : DigitNormBoundTwo assignment)
    (holds : ChunkScheduleHolds assignment) :
    lowValue (assignmentTritMod assignment) digitCount < goldilocksP := by
  have bounded :=
    lowValue_le_of_expectedBorrow_zero
      (digits := assignmentTritMod assignment)
      (bounds := boundDigit)
      (count := digitCount)
      (assignmentTritMod_lt_three norm)
      (fun _ indexLt => boundDigit_lt_three indexLt)
      (chunkSchedule_expectedBorrow_zero norm holds)
  rw [boundLowValue] at bounded
  exact Nat.lt_of_le_of_lt bounded (by decide)

/-- Two scalar transitions. -/
def scalarTwo
    (bounds trits : Fin 2 → Nat) (borrow : Nat) : Nat :=
  scalarStep (bounds 1) (trits 1)
    (scalarStep (bounds 0) (trits 0) borrow)

/-- One chunk transition is definitionally the same two-step function. -/
def chunkTwo
    (bounds trits : Fin 2 → Nat) (borrow : Nat) : Nat :=
  scalarTwo bounds trits borrow

theorem chunkTwo_eq_scalarTwo
    (bounds trits : Fin 2 → Nat) (borrow : Nat) :
    chunkTwo bounds trits borrow =
      scalarStep (bounds 1) (trits 1)
        (scalarStep (bounds 0) (trits 0) borrow) := by
  rfl

/-- Eliminating the one internal scalar borrow preserves the exact relation. -/
theorem chunkTwo_iff_scalarWitness
    (bounds trits : Fin 2 → Nat) (input output : Nat) :
    output = chunkTwo bounds trits input ↔
      ∃ middle,
        middle = scalarStep (bounds 0) (trits 0) input ∧
        output = scalarStep (bounds 1) (trits 1) middle := by
  constructor
  · intro outputEq
    refine ⟨scalarStep (bounds 0) (trits 0) input, rfl, ?_⟩
    simpa [chunkTwo, scalarTwo] using outputEq
  · rintro ⟨middle, middleEq, outputEq⟩
    subst middle
    simpa [chunkTwo, scalarTwo] using outputEq

theorem chunkTwo_le_one
    (bounds trits : Fin 2 → Nat) (borrow : Nat) :
    chunkTwo bounds trits borrow ≤ 1 := by
  exact scalarStep_le_one _ _ _

/-- Concrete boundary regression: `p - 1` is accepted. -/
theorem goldilocks_bound_accepts :
    Nightstream.Implementation.R1CS.ShiftedTernarySound.expectedBorrow
      (fun index => boundDigits.getD index 0)
      (fun index => boundDigits.getD index 0)
      0 digitCount = 0 := by
  native_decide

/-- Concrete adversarial regression: the 41-trit encoding of `p` is rejected. -/
theorem goldilocks_modulus_rejects :
    Nightstream.Implementation.R1CS.ShiftedTernarySound.expectedBorrow
      (fun index => (base3Digits goldilocksP digitCount).getD index 0)
      (fun index => boundDigits.getD index 0)
      0 digitCount = 1 := by
  native_decide

end Nightstream.Implementation.R1CS.CenteredTernaryBorrowChunk
