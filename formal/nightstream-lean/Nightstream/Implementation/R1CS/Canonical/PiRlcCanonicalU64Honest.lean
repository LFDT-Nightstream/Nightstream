import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalU64

/-!
Contract: honest witness construction for the Lean-owned canonical-u64
occurrences used by the `Pi_RLC` sampler.

The source bits are reconstructed internally from the exact value of the
caller-owned lane expression.  They are not supplied by a prover and no
decoded digest is accepted as authority.  This module first proves the
arithmetic reconstruction independently of row placement; batch witness
threading follows below.

Assurance tier: model-level canonical encoding.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalU64Honest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.CanonicalU64Recipe
open Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeHonest

/-- The authoritative little-endian source attached to one evaluated field
lane.  Only its first 64 positions are consumed by the recipe. -/
def sourceOf (value : Nat) : Source where
  bit := fun index => value.testBit index

@[simp] theorem sourceBit_sourceOf (value index : Nat) :
    sourceBit (sourceOf value) index =
      value / 2 ^ index % 2 := by
  exact Nat.toNat_testBit value index

/-- Little-endian value of the first `count` bits. -/
private def bitPrefix (value count : Nat) : Nat :=
  (List.range count).foldl
    (fun total index =>
      total + 2 ^ index * (value.testBit index).toNat) 0

private theorem bitPrefix_eq_mod (value : Nat) :
    ∀ count, bitPrefix value count = value % 2 ^ count
  | 0 => by
      change 0 = value % 1
      exact (Nat.mod_one value).symm
  | count + 1 => by
      unfold bitPrefix
      rw [List.range_succ, List.foldl_append]
      simp only [List.foldl_cons, List.foldl_nil]
      change
        bitPrefix value count +
            2 ^ count * (value.testBit count).toNat =
          value % 2 ^ (count + 1)
      rw [bitPrefix_eq_mod value count, Nat.toNat_testBit,
        Nat.mod_pow_succ]

theorem sourceLow_sourceOf (value : Nat) :
    sourceLow (sourceOf value) = value % 2 ^ 32 := by
  change bitPrefix value 32 = value % 2 ^ 32
  exact bitPrefix_eq_mod value 32

theorem sourceHigh_sourceOf (value : Nat) :
    sourceHigh (sourceOf value) =
      (value / 2 ^ 32) % 2 ^ 32 := by
  change
    (List.range 32).foldl
        (fun total index =>
          total + 2 ^ index *
            (value.testBit (32 + index)).toNat) 0 =
      (value / 2 ^ 32) % 2 ^ 32
  have shifted :
      (List.range 32).foldl
          (fun total index =>
            total + 2 ^ index *
              (value.testBit (32 + index)).toNat) 0 =
        bitPrefix (value / 2 ^ 32) 32 := by
    unfold bitPrefix
    congr 1
    funext total index
    rw [Nat.testBit_div_two_pow]
    simp only [Nat.add_comm]
  rw [shifted, bitPrefix_eq_mod]

/-- Every canonical Goldilocks residue is reconstructed exactly from the
internally derived 64-bit source. -/
theorem sourceWord_sourceOf
    (value : Nat) (canonical : value < goldilocksP) :
    sourceWord (sourceOf value) = value := by
  rw [sourceWord, sourceLow_sourceOf, sourceHigh_sourceOf]
  have valueLt64 : value < 2 ^ 64 :=
    Nat.lt_trans canonical (by decide)
  have highLt : value / 2 ^ 32 < 2 ^ 32 := by
    apply Nat.div_lt_of_lt_mul
    simpa only [show 2 ^ 32 * 2 ^ 32 = 2 ^ 64 by decide] using valueLt64
  rw [Nat.mod_eq_of_lt highLt]
  exact Nat.mod_add_div value (2 ^ 32)

/-! ## One scalar's sixteen occurrences -/

/-- The only placement premise for the batch: every caller-owned symbolic
lane expression lies before the canonical-u64 allocation. -/
structure InputsBelow
    (duplexBase u64Base count : Nat)
    (initialBuilder : SymbolicDuplex.Builder) : Prop where
  input :
    ∀ (coordinate : Fin count)
      (position : Fin PiRlcCanonicalU64.lanesPerScalar)
      column coefficient,
      (column, coefficient) ∈
          (PiRlcCanonicalU64.laneLayout duplexBase u64Base initialBuilder
            coordinate position).input →
        column < u64Base

theorem inputBelowBase
    (duplexBase u64Base count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (below : InputsBelow duplexBase u64Base count initialBuilder)
    (coordinate : Fin count)
    (position : Fin PiRlcCanonicalU64.lanesPerScalar) :
    InputBelowBase
      (PiRlcCanonicalU64.laneLayout duplexBase u64Base initialBuilder
        coordinate position) := by
  intro column coefficient member
  have sourceLt := below.input coordinate position column coefficient member
  simp only [PiRlcCanonicalU64.laneLayout,
    PiRlcCanonicalU64.occurrenceIndex]
  omega

/-- Authoritative source for one occurrence, derived from the exact value of
its symbolic lane expression under the incoming assignment. -/
def laneSource
    (duplexBase u64Base : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (position : Fin PiRlcCanonicalU64.lanesPerScalar)
    (initial : Nat → Nat) : Source :=
  sourceOf
    (lcEval initial
      (PiRlcCanonicalU64.laneLayout duplexBase u64Base initialBuilder
        coordinate position).input)

theorem sourceWord_laneSource
    (duplexBase u64Base : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (position : Fin PiRlcCanonicalU64.lanesPerScalar)
    (initial : Nat → Nat) :
    sourceWord
        (laneSource duplexBase u64Base initialBuilder coordinate position
          initial) =
      lcEval initial
        (PiRlcCanonicalU64.laneLayout duplexBase u64Base initialBuilder
          coordinate position).input := by
  apply sourceWord_sourceOf
  exact SymbolicDuplexSemantics.lcEval_lt _ _

/-- Every row reference of one occurrence lies before the end of its exact
66-column allocation. -/
theorem occurrenceRows_mentions_lt
    (layout : Layout) (positive : 0 < layout.base)
    (below : InputBelowBase layout)
    (row : Row) (rowMember : row ∈ rows layout)
    (column : Nat)
    (mentioned :
      Mentions row.a column ∨ Mentions row.b column ∨
        Mentions row.c column) :
    column < layout.base + auxiliaryCount := by
  rcases mentioned with inA | inB | inC
  · rcases (by simpa [Mentions] using inA :
        ∃ coefficient, (column, coefficient) ∈ row.a) with
      ⟨coefficient, member⟩
    have classified :=
      rows_conservation layout row rowMember column coefficient
        (Or.inl member)
    rcases classified with rfl | ⟨coefficient, source⟩ | allocated
    · simp only [auxiliaryCount]
      omega
    · have sourceLt := below column coefficient source
      simp only [auxiliaryCount]
      omega
    · exact (allocation_in_window layout column allocated).2
  · rcases (by simpa [Mentions] using inB :
        ∃ coefficient, (column, coefficient) ∈ row.b) with
      ⟨coefficient, member⟩
    have classified :=
      rows_conservation layout row rowMember column coefficient
        (Or.inr (Or.inl member))
    rcases classified with rfl | ⟨coefficient, source⟩ | allocated
    · simp only [auxiliaryCount]
      omega
    · have sourceLt := below column coefficient source
      simp only [auxiliaryCount]
      omega
    · exact (allocation_in_window layout column allocated).2
  · rcases (by simpa [Mentions] using inC :
        ∃ coefficient, (column, coefficient) ∈ row.c) with
      ⟨coefficient, member⟩
    have classified :=
      rows_conservation layout row rowMember column coefficient
        (Or.inr (Or.inr member))
    rcases classified with rfl | ⟨coefficient, source⟩ | allocated
    · simp only [auxiliaryCount]
      omega
    · have sourceLt := below column coefficient source
      simp only [auxiliaryCount]
      omega
    · exact (allocation_in_window layout column allocated).2

def scalarPrefixBoundary
    (u64Base : Nat) {count : Nat} (coordinate : Fin count)
    (processed : Nat) : Nat :=
  u64Base +
    (coordinate.val * PiRlcCanonicalU64.lanesPerScalar + processed) *
      auxiliaryCount

/-- Sequentially apply the first `processed` lane witnesses of one scalar. -/
def scalarPrefixWitness
    (field : FieldInverse)
    (duplexBase u64Base : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat) : Nat → Nat → Nat
  | 0 => initial
  | processed + 1 =>
      if bounded : processed < PiRlcCanonicalU64.lanesPerScalar then
        let position : Fin PiRlcCanonicalU64.lanesPerScalar :=
          ⟨processed, bounded⟩
        witness field
          (laneSource duplexBase u64Base initialBuilder coordinate position
            initial)
          (PiRlcCanonicalU64.laneLayout duplexBase u64Base initialBuilder
            coordinate position)
          (scalarPrefixWitness field duplexBase u64Base initialBuilder
            coordinate initial processed)
      else
        scalarPrefixWitness field duplexBase u64Base initialBuilder
          coordinate initial processed

theorem scalarPrefixWitness_succ
    (field : FieldInverse)
    (duplexBase u64Base : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat) {processed : Nat}
    (bounded : processed < PiRlcCanonicalU64.lanesPerScalar) :
    scalarPrefixWitness field duplexBase u64Base initialBuilder coordinate
        initial (processed + 1) =
      witness field
        (laneSource duplexBase u64Base initialBuilder coordinate
          ⟨processed, bounded⟩ initial)
        (PiRlcCanonicalU64.laneLayout duplexBase u64Base initialBuilder
          coordinate ⟨processed, bounded⟩)
        (scalarPrefixWitness field duplexBase u64Base initialBuilder
          coordinate initial processed) := by
  simp [scalarPrefixWitness, bounded]

theorem scalarPrefixWitness_before_u64Base
    (field : FieldInverse)
    (duplexBase u64Base : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat) (processed : Nat)
    {column : Nat} (before : column < u64Base) :
    scalarPrefixWitness field duplexBase u64Base initialBuilder coordinate
        initial processed column =
      initial column := by
  induction processed with
  | zero => rfl
  | succ processed hypothesis =>
      simp only [scalarPrefixWitness]
      split
      · rw [witness_before]
        · exact hypothesis
        · simp only [PiRlcCanonicalU64.laneLayout,
            PiRlcCanonicalU64.occurrenceIndex]
          omega
      · exact hypothesis

theorem scalarPrefixWitness_canonical
    (field : FieldInverse)
    (duplexBase u64Base : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat)
    (initialCanonical : ∀ column, initial column < goldilocksP) :
    ∀ processed column,
      scalarPrefixWitness field duplexBase u64Base initialBuilder coordinate
        initial processed column < goldilocksP := by
  intro processed
  induction processed with
  | zero => exact initialCanonical
  | succ processed hypothesis =>
      simp only [scalarPrefixWitness]
      split
      · apply witness_canonical
        exact hypothesis
      · exact hypothesis

theorem scalarPrefixWitness_stable
    (field : FieldInverse)
    (duplexBase u64Base : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat)
    {start finish : Nat} (ordered : start ≤ finish)
    (finishBounded : finish ≤ PiRlcCanonicalU64.lanesPerScalar)
    {column : Nat}
    (before : column < scalarPrefixBoundary u64Base coordinate start) :
    scalarPrefixWitness field duplexBase u64Base initialBuilder coordinate
        initial finish column =
      scalarPrefixWitness field duplexBase u64Base initialBuilder coordinate
        initial start column := by
  induction finish generalizing start with
  | zero =>
      have startZero : start = 0 := by omega
      subst start
      rfl
  | succ finish hypothesis =>
      by_cases atEnd : start = finish + 1
      · subst start
        rfl
      · have startLe : start ≤ finish := by omega
        have finishLt : finish < PiRlcCanonicalU64.lanesPerScalar := by
          omega
        rw [scalarPrefixWitness_succ field duplexBase u64Base initialBuilder
          coordinate initial finishLt]
        rw [witness_before]
        · exact hypothesis startLe (by omega) before
        · simp only [PiRlcCanonicalU64.laneLayout,
            PiRlcCanonicalU64.occurrenceIndex, scalarPrefixBoundary,
            PiRlcCanonicalU64.lanesPerScalar, auxiliaryCount] at before ⊢
          omega

/-- The next lane occurrence is honestly satisfiable over the assignment
produced by all preceding lane occurrences. -/
theorem stage_complete
    (field : FieldInverse)
    (duplexBase u64Base count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (coordinate : Fin count)
    (initial : Nat → Nat)
    (positive : 0 < u64Base) (constantWire : initial 0 = 1)
    (below : InputsBelow duplexBase u64Base count initialBuilder)
    {processed : Nat}
    (processedLt : processed < PiRlcCanonicalU64.lanesPerScalar) :
    Satisfies
      (rows
        (PiRlcCanonicalU64.laneLayout duplexBase u64Base initialBuilder
          coordinate ⟨processed, processedLt⟩))
      (scalarPrefixWitness field duplexBase u64Base initialBuilder coordinate
        initial (processed + 1)) := by
  let position : Fin PiRlcCanonicalU64.lanesPerScalar :=
    ⟨processed, processedLt⟩
  let layout :=
    PiRlcCanonicalU64.laneLayout duplexBase u64Base initialBuilder
      coordinate position
  let source :=
    laneSource duplexBase u64Base initialBuilder coordinate position initial
  let prior :=
    scalarPrefixWitness field duplexBase u64Base initialBuilder coordinate
      initial processed
  have layoutPositive : 0 < layout.base := by
    simp only [layout, PiRlcCanonicalU64.laneLayout,
      PiRlcCanonicalU64.occurrenceIndex]
    omega
  have priorConstant : prior 0 = 1 := by
    change
      scalarPrefixWitness field duplexBase u64Base initialBuilder coordinate
        initial processed 0 = 1
    rw [scalarPrefixWitness_before_u64Base field duplexBase u64Base
      initialBuilder coordinate initial processed positive]
    exact constantWire
  have layoutBelow : InputBelowBase layout := by
    exact inputBelowBase duplexBase u64Base count initialBuilder below
      coordinate position
  have inputPreserved :
      lcEval prior layout.input = lcEval initial layout.input := by
    apply KMulHonest.lcEval_congr
    intro column mentioned
    rcases (by simpa [Mentions] using mentioned :
        ∃ coefficient, (column, coefficient) ∈ layout.input) with
      ⟨coefficient, member⟩
    exact scalarPrefixWitness_before_u64Base field duplexBase u64Base
      initialBuilder coordinate initial processed
      (below.input coordinate position column coefficient member)
  have inputMatches : lcEval prior layout.input = sourceWord source := by
    rw [inputPreserved]
    exact
      (sourceWord_laneSource duplexBase u64Base initialBuilder coordinate
        position initial).symm
  have sourceCanonical : sourceWord source < goldilocksP := by
    rw [sourceWord_laneSource duplexBase u64Base initialBuilder coordinate
      position initial]
    exact SymbolicDuplexSemantics.lcEval_lt _ _
  rw [scalarPrefixWitness_succ field duplexBase u64Base initialBuilder
    coordinate initial processedLt]
  exact complete field source layout prior layoutPositive priorConstant
    layoutBelow inputMatches sourceCanonical

private theorem rowHolds_congr
    (left right : Nat → Nat) (row : Row)
    (agree :
      ∀ column,
        Mentions row.a column ∨ Mentions row.b column ∨
          Mentions row.c column →
        left column = right column) :
    RowHolds left row ↔ RowHolds right row := by
  unfold RowHolds
  rw [KMulHonest.lcEval_congr
      left right row.a (fun column member => agree column (Or.inl member)),
    KMulHonest.lcEval_congr
      left right row.b
      (fun column member => agree column (Or.inr (Or.inl member))),
    KMulHonest.lcEval_congr
      left right row.c
      (fun column member => agree column (Or.inr (Or.inr member)))]

/-- Final assignment after all sixteen lane decompositions of one scalar. -/
def scalarWitness
    (field : FieldInverse)
    (duplexBase u64Base : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat) : Nat → Nat :=
  scalarPrefixWitness field duplexBase u64Base initialBuilder coordinate
    initial PiRlcCanonicalU64.lanesPerScalar

theorem scalarWitness_before_u64Base
    (field : FieldInverse)
    (duplexBase u64Base : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat)
    {column : Nat} (before : column < u64Base) :
    scalarWitness field duplexBase u64Base initialBuilder coordinate initial
        column =
      initial column := by
  exact scalarPrefixWitness_before_u64Base field duplexBase u64Base
    initialBuilder coordinate initial PiRlcCanonicalU64.lanesPerScalar before

theorem scalarWitness_canonical
    (field : FieldInverse)
    (duplexBase u64Base : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat)
    (initialCanonical : ∀ column, initial column < goldilocksP) :
    ∀ column,
      scalarWitness field duplexBase u64Base initialBuilder coordinate initial
        column < goldilocksP := by
  exact scalarPrefixWitness_canonical field duplexBase u64Base
    initialBuilder coordinate initial initialCanonical
    PiRlcCanonicalU64.lanesPerScalar

/-- Honest completeness of all 1,104 canonical-u64 rows for one scalar. -/
theorem scalarRows_complete
    (field : FieldInverse)
    (duplexBase u64Base count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (coordinate : Fin count)
    (initial : Nat → Nat)
    (positive : 0 < u64Base) (constantWire : initial 0 = 1)
    (below : InputsBelow duplexBase u64Base count initialBuilder) :
    Satisfies
      (PiRlcCanonicalU64.scalarRows duplexBase u64Base initialBuilder
        coordinate)
      (scalarWitness field duplexBase u64Base initialBuilder coordinate
        initial) := by
  intro row member
  rcases List.mem_flatMap.mp member with
    ⟨position, _, rowMember⟩
  have stage :=
    stage_complete field duplexBase u64Base count initialBuilder coordinate
      initial positive constantWire below position.isLt
  have stageHolds := stage row (by
    simpa using rowMember)
  apply
    (rowHolds_congr
      (scalarPrefixWitness field duplexBase u64Base initialBuilder coordinate
        initial (position.val + 1))
      (scalarWitness field duplexBase u64Base initialBuilder coordinate
        initial)
      row ?_).mp
  · exact stageHolds
  · intro column mentioned
    symm
    apply scalarPrefixWitness_stable field duplexBase u64Base initialBuilder
      coordinate initial
      (start := position.val + 1)
      (finish := PiRlcCanonicalU64.lanesPerScalar)
    · omega
    · exact Nat.le_refl _
    · have layoutPositive :
          0 <
            (PiRlcCanonicalU64.laneLayout duplexBase u64Base initialBuilder
              coordinate position).base := by
          simp only [PiRlcCanonicalU64.laneLayout,
            PiRlcCanonicalU64.occurrenceIndex]
          omega
      have localBound :=
        occurrenceRows_mentions_lt
          (PiRlcCanonicalU64.laneLayout duplexBase u64Base initialBuilder
            coordinate position)
          layoutPositive
          (inputBelowBase duplexBase u64Base count initialBuilder below
            coordinate position)
          row rowMember column mentioned
      simp only [PiRlcCanonicalU64.laneLayout,
        PiRlcCanonicalU64.occurrenceIndex, scalarPrefixBoundary,
        PiRlcCanonicalU64.lanesPerScalar, auxiliaryCount] at localBound ⊢
      omega

/-- Every source bit produced by the scalar witness is the corresponding bit
of the authoritative evaluated lane. -/
theorem scalarWitness_bit
    (field : FieldInverse)
    (duplexBase u64Base : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat)
    (position : Fin PiRlcCanonicalU64.lanesPerScalar)
    {index : Nat} (indexLt : index < 64) :
    scalarWitness field duplexBase u64Base initialBuilder coordinate initial
        (bitColumn
          (PiRlcCanonicalU64.laneLayout duplexBase u64Base initialBuilder
            coordinate position)
          index) =
      sourceBit
        (laneSource duplexBase u64Base initialBuilder coordinate position
          initial)
        index := by
  unfold scalarWitness
  rw [scalarPrefixWitness_stable field duplexBase u64Base initialBuilder
    coordinate initial
    (start := position.val + 1)
    (finish := PiRlcCanonicalU64.lanesPerScalar)
    (by omega) (Nat.le_refl _)
    (by
      simp only [bitColumn, PiRlcCanonicalU64.laneLayout,
        PiRlcCanonicalU64.occurrenceIndex, scalarPrefixBoundary,
        PiRlcCanonicalU64.lanesPerScalar, auxiliaryCount]
      omega)]
  rw [scalarPrefixWitness_succ field duplexBase u64Base initialBuilder
    coordinate initial position.isLt]
  exact witness_bit field
    (laneSource duplexBase u64Base initialBuilder coordinate position initial)
    (PiRlcCanonicalU64.laneLayout duplexBase u64Base initialBuilder
      coordinate position)
    (scalarPrefixWitness field duplexBase u64Base initialBuilder coordinate
      initial position.val)
    indexLt

/-! ## Coordinate-major batch -/

def batchPrefixBoundary
    (u64Base processed : Nat) : Nat :=
  u64Base +
    processed * PiRlcCanonicalU64.lanesPerScalar * auxiliaryCount

theorem scalarWitness_before_scalarBase
    (field : FieldInverse)
    (duplexBase u64Base : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat)
    {column : Nat}
    (before : column < batchPrefixBoundary u64Base coordinate.val) :
    scalarWitness field duplexBase u64Base initialBuilder coordinate initial
        column =
      initial column := by
  unfold scalarWitness
  have stable :=
    scalarPrefixWitness_stable field duplexBase u64Base initialBuilder
      coordinate initial
      (start := 0) (finish := PiRlcCanonicalU64.lanesPerScalar)
      (Nat.zero_le _) (Nat.le_refl _) (column := column)
  apply stable
  simpa only [scalarPrefixBoundary, batchPrefixBoundary,
    PiRlcCanonicalU64.lanesPerScalar, auxiliaryCount,
    Nat.mul_zero, Nat.add_zero, Nat.zero_mul] using before

/-- Sequentially apply the complete sixteen-lane witness for each of the
first `processed` coordinates. -/
def batchPrefixWitness
    (field : FieldInverse)
    (duplexBase u64Base count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat) : Nat → Nat → Nat
  | 0 => initial
  | processed + 1 =>
      if bounded : processed < count then
        scalarWitness field duplexBase u64Base initialBuilder
          ⟨processed, bounded⟩
          (batchPrefixWitness field duplexBase u64Base count initialBuilder
            initial processed)
      else
        batchPrefixWitness field duplexBase u64Base count initialBuilder
          initial processed

theorem batchPrefixWitness_succ
    (field : FieldInverse)
    (duplexBase u64Base count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    {processed : Nat} (bounded : processed < count) :
    batchPrefixWitness field duplexBase u64Base count initialBuilder initial
        (processed + 1) =
      scalarWitness field duplexBase u64Base initialBuilder
        ⟨processed, bounded⟩
        (batchPrefixWitness field duplexBase u64Base count initialBuilder initial
          processed) := by
  simp [batchPrefixWitness, bounded]

theorem batchPrefixWitness_before_u64Base
    (field : FieldInverse)
    (duplexBase u64Base count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat) (processed : Nat)
    {column : Nat} (before : column < u64Base) :
    batchPrefixWitness field duplexBase u64Base count initialBuilder initial
        processed
        column =
      initial column := by
  induction processed with
  | zero => rfl
  | succ processed hypothesis =>
      simp only [batchPrefixWitness]
      split
      · rw [scalarWitness_before_u64Base]
        exact hypothesis
        exact before
      · exact hypothesis

theorem batchPrefixWitness_canonical
    (field : FieldInverse)
    (duplexBase u64Base count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    (initialCanonical : ∀ column, initial column < goldilocksP) :
    ∀ processed column,
      batchPrefixWitness field duplexBase u64Base count initialBuilder initial
        processed
        column < goldilocksP := by
  intro processed
  induction processed with
  | zero => exact initialCanonical
  | succ processed hypothesis =>
      simp only [batchPrefixWitness]
      split
      · apply scalarWitness_canonical
        exact hypothesis
      · exact hypothesis

theorem batchPrefixWitness_stable
    (field : FieldInverse)
    (duplexBase u64Base count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    {start finish : Nat} (ordered : start ≤ finish)
    (finishBounded : finish ≤ count)
    {column : Nat}
    (before : column < batchPrefixBoundary u64Base start) :
    batchPrefixWitness field duplexBase u64Base count initialBuilder initial
        finish column =
      batchPrefixWitness field duplexBase u64Base count initialBuilder initial
        start
        column := by
  induction finish generalizing start with
  | zero =>
      have startZero : start = 0 := by omega
      subst start
      rfl
  | succ finish hypothesis =>
      by_cases atEnd : start = finish + 1
      · subst start
        rfl
      · have startLe : start ≤ finish := by omega
        have finishLt : finish < count := by omega
        rw [batchPrefixWitness_succ field duplexBase u64Base count
          initialBuilder initial finishLt]
        rw [scalarWitness_before_scalarBase]
        · exact hypothesis startLe (by omega) before
        · simp only [batchPrefixBoundary,
            PiRlcCanonicalU64.lanesPerScalar, auxiliaryCount] at before ⊢
          omega

/-- Final coordinate-major canonical-u64 witness. -/
def batchWitness
    (field : FieldInverse)
    (duplexBase u64Base count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat) : Nat → Nat :=
  batchPrefixWitness field duplexBase u64Base count initialBuilder initial
    count

theorem batchWitness_before_u64Base
    (field : FieldInverse)
    (duplexBase u64Base count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    {column : Nat} (before : column < u64Base) :
    batchWitness field duplexBase u64Base count initialBuilder initial column =
      initial column := by
  exact batchPrefixWitness_before_u64Base field duplexBase u64Base count
    initialBuilder initial count before

theorem batchWitness_canonical
    (field : FieldInverse)
    (duplexBase u64Base count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    (initialCanonical : ∀ column, initial column < goldilocksP) :
    ∀ column,
      batchWitness field duplexBase u64Base count initialBuilder initial
        column <
        goldilocksP := by
  exact batchPrefixWitness_canonical field duplexBase u64Base count
    initialBuilder initial initialCanonical count

theorem scalarRows_mentions_lt
    (duplexBase u64Base count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (below : InputsBelow duplexBase u64Base count initialBuilder)
    (coordinate : Fin count)
    (positive : 0 < u64Base)
    (row : Row)
    (rowMember :
      row ∈ PiRlcCanonicalU64.scalarRows duplexBase u64Base initialBuilder
        coordinate)
    (column : Nat)
    (mentioned :
      Mentions row.a column ∨ Mentions row.b column ∨
        Mentions row.c column) :
    column < batchPrefixBoundary u64Base (coordinate.val + 1) := by
  rcases List.mem_flatMap.mp rowMember with
    ⟨position, _, localMember⟩
  have layoutPositive :
      0 <
        (PiRlcCanonicalU64.laneLayout duplexBase u64Base initialBuilder
          coordinate position).base := by
    simp only [PiRlcCanonicalU64.laneLayout,
      PiRlcCanonicalU64.occurrenceIndex]
    omega
  have localBound :=
    occurrenceRows_mentions_lt
      (PiRlcCanonicalU64.laneLayout duplexBase u64Base initialBuilder
        coordinate position)
      layoutPositive
      (inputBelowBase duplexBase u64Base count initialBuilder below
        coordinate position)
      row localMember column mentioned
  simp only [PiRlcCanonicalU64.laneLayout,
    PiRlcCanonicalU64.occurrenceIndex, batchPrefixBoundary,
    PiRlcCanonicalU64.lanesPerScalar, auxiliaryCount] at localBound ⊢
  have positionLt := position.isLt
  omega

theorem batchStage_complete
    (field : FieldInverse)
    (duplexBase u64Base count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    (positive : 0 < u64Base) (constantWire : initial 0 = 1)
    (below : InputsBelow duplexBase u64Base count initialBuilder)
    {processed : Nat} (processedLt : processed < count) :
    Satisfies
      (PiRlcCanonicalU64.scalarRows duplexBase u64Base initialBuilder
        ⟨processed, processedLt⟩)
      (batchPrefixWitness field duplexBase u64Base count initialBuilder initial
        (processed + 1)) := by
  let prior :=
    batchPrefixWitness field duplexBase u64Base count initialBuilder initial
      processed
  have priorConstant : prior 0 = 1 := by
    change
      batchPrefixWitness field duplexBase u64Base count initialBuilder initial
        processed 0 = 1
    rw [batchPrefixWitness_before_u64Base field duplexBase u64Base count
      initialBuilder initial processed positive]
    exact constantWire
  rw [batchPrefixWitness_succ field duplexBase u64Base count initialBuilder
    initial processedLt]
  exact scalarRows_complete field duplexBase u64Base count initialBuilder
    ⟨processed, processedLt⟩ prior positive priorConstant below

/-- Honest completeness of the exact coordinate-major canonical-u64 batch. -/
theorem rows_complete
    (field : FieldInverse)
    (duplexBase u64Base count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    (positive : 0 < u64Base) (constantWire : initial 0 = 1)
    (below : InputsBelow duplexBase u64Base count initialBuilder) :
    Satisfies
      (PiRlcCanonicalU64.rows duplexBase u64Base count initialBuilder)
      (batchWitness field duplexBase u64Base count initialBuilder initial) := by
  intro row member
  rcases List.mem_flatMap.mp member with
    ⟨coordinate, _, rowMember⟩
  have stage :=
    batchStage_complete field duplexBase u64Base count initialBuilder initial
      positive constantWire below coordinate.isLt
  have stageHolds := stage row (by simpa using rowMember)
  apply
    (rowHolds_congr
      (batchPrefixWitness field duplexBase u64Base count initialBuilder initial
        (coordinate.val + 1))
      (batchWitness field duplexBase u64Base count initialBuilder initial)
      row ?_).mp
  · exact stageHolds
  · intro column mentioned
    symm
    apply batchPrefixWitness_stable field duplexBase u64Base count
      initialBuilder initial
      (start := coordinate.val + 1) (finish := count)
    · omega
    · exact Nat.le_refl _
    · exact scalarRows_mentions_lt duplexBase u64Base count initialBuilder
        below coordinate positive row rowMember column mentioned

/-- The final batch value of every allocated bit is a Boolean derived from
the exact evaluated symbolic lane. -/
theorem batchWitness_bit
    (field : FieldInverse)
    (duplexBase u64Base count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    (below : InputsBelow duplexBase u64Base count initialBuilder)
    (coordinate : Fin count)
    (position : Fin PiRlcCanonicalU64.lanesPerScalar)
    {index : Nat} (indexLt : index < 64) :
    batchWitness field duplexBase u64Base count initialBuilder initial
        (bitColumn
          (PiRlcCanonicalU64.laneLayout duplexBase u64Base initialBuilder
            coordinate position)
          index) =
      sourceBit
        (laneSource duplexBase u64Base initialBuilder coordinate position
          initial)
        index := by
  unfold batchWitness
  rw [batchPrefixWitness_stable field duplexBase u64Base count initialBuilder
    initial
    (start := coordinate.val + 1) (finish := count)
    (by omega) (Nat.le_refl _)
    (by
      simp only [bitColumn, PiRlcCanonicalU64.laneLayout,
        PiRlcCanonicalU64.occurrenceIndex, batchPrefixBoundary,
        PiRlcCanonicalU64.lanesPerScalar, auxiliaryCount]
      omega)]
  rw [batchPrefixWitness_succ field duplexBase u64Base count initialBuilder
    initial coordinate.isLt]
  have bitEq :=
    scalarWitness_bit field duplexBase u64Base initialBuilder coordinate
      (batchPrefixWitness field duplexBase u64Base count initialBuilder initial
        coordinate.val)
      position indexLt
  rw [bitEq]
  congr 1
  unfold laneSource
  congr 1
  apply KMulHonest.lcEval_congr
  intro column mentioned
  rcases (by simpa [Mentions] using mentioned :
      ∃ coefficient,
        (column, coefficient) ∈
          (PiRlcCanonicalU64.laneLayout duplexBase u64Base initialBuilder
            coordinate position).input) with
    ⟨coefficient, member⟩
  exact batchPrefixWitness_before_u64Base field duplexBase u64Base count
    initialBuilder initial coordinate.val
    (below.input coordinate position column coefficient member)

end Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalU64Honest
