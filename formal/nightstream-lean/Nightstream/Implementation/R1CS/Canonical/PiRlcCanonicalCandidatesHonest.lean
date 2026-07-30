import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelectorComplete

/-!
Contract: honest witness threading for one scalar's 64 canonical `Pi_RLC`
candidate classifiers.

Candidate `i` reads candidate `i - 1`'s cumulative accepted-count column.
The witness is therefore constructed in physical order.  Source bits remain
caller-owned and must precede the candidate allocation; the accepted-prefix
bound and every prior value are derived internally by the recursion.

This module does not construct the upstream canonical-u64 source bits or the
downstream selector witness.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidatesHonest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidate
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidateHonest
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidateSound
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidates
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidatesSound
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

/-- Every authoritative source bit read by this scalar lies before the
candidate allocation. -/
structure SourcesBelow
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count) : Prop where
  source :
    ∀ candidate : Fin candidatesPerScalar,
      ∀ bit : Fin sourceBitCount,
        (candidateLayout duplexBase u64Base candidateBase initialBuilder
          coordinate candidate).sourceBit bit <
        candidateBase

theorem inputsBelowBase
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (sourcesBelow :
      SourcesBelow duplexBase u64Base candidateBase initialBuilder coordinate)
    (candidate : Fin candidatesPerScalar) :
    InputsBelowBase
      (candidateLayout duplexBase u64Base candidateBase initialBuilder
        coordinate candidate) := by
  constructor
  · intro bit
    have sourceLt := sourcesBelow.source candidate bit
    change
      CanonicalU64Recipe.bitColumn
          (PiRlcCanonicalU64.laneLayout duplexBase u64Base initialBuilder
            coordinate (lanePosition candidate))
          (sourceBitIndex candidate bit) <
        candidateBase at sourceLt
    change
      CanonicalU64Recipe.bitColumn
          (PiRlcCanonicalU64.laneLayout duplexBase u64Base initialBuilder
            coordinate (lanePosition candidate))
          (sourceBitIndex candidate bit) <
        occurrenceBase candidateBase coordinate candidate
    simp only [occurrenceBase, occurrenceIndex]
    omega
  · intro column coefficient member
    simp only [candidateLayout] at member ⊢
    by_cases zero : candidate.val = 0
    · simp [prior, zero] at member
    · simp only [prior, zero, if_false, List.mem_cons, List.not_mem_nil,
        or_false, Prod.mk.injEq] at member
      rcases member with ⟨rfl, rfl⟩
      have candidatePositive : 0 < candidate.val := Nat.pos_of_ne_zero zero
      simp only [occurrenceBase, occurrenceIndex, candidatesPerScalar,
        auxiliaryCount]
      omega

/-- Accepted count computed from the authoritative source bits, before any
candidate-owned column is written. -/
def honestAcceptedPrefix
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat) : Nat → Nat
  | 0 => 0
  | index + 1 =>
      honestAcceptedPrefix duplexBase u64Base candidateBase initialBuilder
          coordinate initial index +
        acceptValue initial
          (candidateLayout duplexBase u64Base candidateBase initialBuilder
            coordinate (candidateOfNat index))

theorem honestAcceptedPrefix_le
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat) :
    ∀ index,
      honestAcceptedPrefix duplexBase u64Base candidateBase initialBuilder
        coordinate initial index ≤ index := by
  intro index
  induction index with
  | zero => exact Nat.le_refl 0
  | succ index hypothesis =>
      simp only [honestAcceptedPrefix]
      have acceptedLe :=
        acceptValue_le_one initial
          (candidateLayout duplexBase u64Base candidateBase initialBuilder
            coordinate (candidateOfNat index))
      omega

/-- Sequentially apply the first `processed` candidate witnesses.  The branch
past 64 makes the definition total; every theorem below stays in the bounded
prefix. -/
def prefixWitness
    (field : FieldInverse)
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat) : Nat → Nat → Nat
  | 0 => initial
  | processed + 1 =>
      if bounded : processed < candidatesPerScalar then
        witness field
          (candidateLayout duplexBase u64Base candidateBase initialBuilder
            coordinate (candidateOfNat processed))
          (prefixWitness field duplexBase u64Base candidateBase initialBuilder
            coordinate initial processed)
      else
        prefixWitness field duplexBase u64Base candidateBase initialBuilder
          coordinate initial processed

theorem prefixWitness_succ
    (field : FieldInverse)
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat) {processed : Nat}
    (bounded : processed < candidatesPerScalar) :
    prefixWitness field duplexBase u64Base candidateBase initialBuilder
        coordinate initial (processed + 1) =
      witness field
        (candidateLayout duplexBase u64Base candidateBase initialBuilder
          coordinate (candidateOfNat processed))
        (prefixWitness field duplexBase u64Base candidateBase initialBuilder
          coordinate initial processed) := by
  simp [prefixWitness, bounded]

theorem prefixWitness_before_candidateBase
    (field : FieldInverse)
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat) (processed : Nat)
    {column : Nat} (before : column < candidateBase) :
    prefixWitness field duplexBase u64Base candidateBase initialBuilder
        coordinate initial processed column =
      initial column := by
  induction processed with
  | zero => rfl
  | succ processed hypothesis =>
      simp only [prefixWitness]
      split
      · rw [witness_before]
        · exact hypothesis
        · simp only [candidateLayout, occurrenceBase, occurrenceIndex]
          omega
      · exact hypothesis

theorem prefixWitness_sourceBit
    (field : FieldInverse)
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat) (processed : Nat)
    (sourcesBelow :
      SourcesBelow duplexBase u64Base candidateBase initialBuilder coordinate)
    (candidate : Fin candidatesPerScalar) (bit : Fin sourceBitCount) :
    prefixWitness field duplexBase u64Base candidateBase initialBuilder
        coordinate initial processed
        ((candidateLayout duplexBase u64Base candidateBase initialBuilder
          coordinate candidate).sourceBit bit) =
      initial
        ((candidateLayout duplexBase u64Base candidateBase initialBuilder
          coordinate candidate).sourceBit bit) := by
  exact prefixWitness_before_candidateBase field duplexBase u64Base
    candidateBase initialBuilder coordinate initial processed
    (sourcesBelow.source candidate bit)

theorem prefixWitness_sourceBits
    (field : FieldInverse)
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat) (processed : Nat)
    (sourcesBelow :
      SourcesBelow duplexBase u64Base candidateBase initialBuilder coordinate)
    (candidate : Fin candidatesPerScalar)
    (sourceBits :
      SourceBitsBoolean initial
        (candidateLayout duplexBase u64Base candidateBase initialBuilder
          coordinate candidate)) :
    SourceBitsBoolean
      (prefixWitness field duplexBase u64Base candidateBase initialBuilder
        coordinate initial processed)
      (candidateLayout duplexBase u64Base candidateBase initialBuilder
        coordinate candidate) := by
  intro bit
  rw [prefixWitness_sourceBit field duplexBase u64Base candidateBase
    initialBuilder coordinate initial processed sourcesBelow candidate bit]
  exact sourceBits bit

theorem prefixWitness_sourceValue
    (field : FieldInverse)
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat) (processed : Nat)
    (sourcesBelow :
      SourcesBelow duplexBase u64Base candidateBase initialBuilder coordinate)
    (candidate : Fin candidatesPerScalar) :
    sourceValue
        (prefixWitness field duplexBase u64Base candidateBase initialBuilder
          coordinate initial processed)
        (candidateLayout duplexBase u64Base candidateBase initialBuilder
          coordinate candidate) =
      sourceValue initial
        (candidateLayout duplexBase u64Base candidateBase initialBuilder
          coordinate candidate) := by
  unfold sourceValue chunkValue
  have go :
      ∀ (bits : List (Fin sourceBitCount)) (accumulator : Nat),
        bits.foldl
            (fun value bit =>
              value + 2 ^ bit.val *
                prefixWitness field duplexBase u64Base candidateBase
                  initialBuilder coordinate initial processed
                  ((candidateLayout duplexBase u64Base candidateBase
                    initialBuilder coordinate candidate).sourceBit bit))
            accumulator =
          bits.foldl
            (fun value bit =>
              value + 2 ^ bit.val *
                initial
                  ((candidateLayout duplexBase u64Base candidateBase
                    initialBuilder coordinate candidate).sourceBit bit))
            accumulator := by
    intro bits accumulator
    induction bits generalizing accumulator with
    | nil => rfl
    | cons head tail hypothesis =>
        simp only [List.foldl]
        rw [prefixWitness_sourceBit field duplexBase u64Base candidateBase
          initialBuilder coordinate initial processed sourcesBelow candidate
          head]
        exact hypothesis _
  exact go _ 0

theorem prefixWitness_acceptValue
    (field : FieldInverse)
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat) (processed : Nat)
    (sourcesBelow :
      SourcesBelow duplexBase u64Base candidateBase initialBuilder coordinate)
    (candidate : Fin candidatesPerScalar) :
    acceptValue
        (prefixWitness field duplexBase u64Base candidateBase initialBuilder
          coordinate initial processed)
        (candidateLayout duplexBase u64Base candidateBase initialBuilder
          coordinate candidate) =
      acceptValue initial
        (candidateLayout duplexBase u64Base candidateBase initialBuilder
          coordinate candidate) := by
  unfold acceptValue
  rw [prefixWitness_sourceValue field duplexBase u64Base candidateBase
    initialBuilder coordinate initial processed sourcesBelow candidate]

private theorem prior_candidateOfNat_succ
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    {index : Nat} (bounded : index + 1 < candidatesPerScalar) :
    (candidateLayout duplexBase u64Base candidateBase initialBuilder coordinate
        (candidateOfNat (index + 1))).prior =
      [(cumulativeColumn
        (candidateLayout duplexBase u64Base candidateBase initialBuilder
          coordinate (candidateOfNat index)), 1)] := by
  change
    prior candidateBase coordinate (candidateOfNat (index + 1)) =
      _
  rw [prior_successor duplexBase u64Base candidateBase initialBuilder
    coordinate (candidateOfNat (index + 1)) (by
      rw [candidateOfNat_val bounded]
      omega)]
  congr 3
  congr 1
  apply Fin.ext
  change
    (candidateOfNat (index + 1)).val - 1 =
      (candidateOfNat index).val
  have current := candidateOfNat_val bounded
  have previous := candidateOfNat_val (by omega :
    index < candidatesPerScalar)
  omega

theorem prefixWitness_prior_eval
    (field : FieldInverse)
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat)
    (sourcesBelow :
      SourcesBelow duplexBase u64Base candidateBase initialBuilder coordinate) :
    ∀ index, index < candidatesPerScalar →
      lcEval
          (prefixWitness field duplexBase u64Base candidateBase initialBuilder
            coordinate initial index)
          (candidateLayout duplexBase u64Base candidateBase initialBuilder
            coordinate (candidateOfNat index)).prior =
        honestAcceptedPrefix duplexBase u64Base candidateBase initialBuilder
          coordinate initial index := by
  intro index bounded
  induction index with
  | zero =>
      simp [candidateLayout, prior, candidateOfNat, lcEval,
        honestAcceptedPrefix]
  | succ index hypothesis =>
      have indexBounded : index < candidatesPerScalar := by omega
      have cumulativeAt :
          prefixWitness field duplexBase u64Base candidateBase initialBuilder
              coordinate initial (index + 1)
              (cumulativeColumn
                (candidateLayout duplexBase u64Base candidateBase
                  initialBuilder coordinate (candidateOfNat index))) =
            honestAcceptedPrefix duplexBase u64Base candidateBase
              initialBuilder coordinate initial (index + 1) := by
        rw [prefixWitness_succ field duplexBase u64Base candidateBase
          initialBuilder coordinate initial indexBounded]
        rw [witness_cumulative]
        unfold cumulative
        rw [hypothesis indexBounded]
        rw [prefixWitness_acceptValue field duplexBase u64Base candidateBase
          initialBuilder coordinate initial index sourcesBelow
          (candidateOfNat index)]
        rfl
      rw [prior_candidateOfNat_succ duplexBase u64Base candidateBase
        initialBuilder coordinate bounded]
      unfold lcEval
      simp only [List.foldl, Nat.one_mul, Nat.zero_add, cumulativeAt]
      rw [Nat.mod_eq_of_lt]
      exact Nat.lt_trans
        (Nat.lt_of_le_of_lt
          (honestAcceptedPrefix_le duplexBase u64Base candidateBase
            initialBuilder coordinate initial (index + 1))
          bounded)
        (by decide : candidatesPerScalar < goldilocksP)

theorem prefixWitness_constant
    (field : FieldInverse)
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat) (processed : Nat)
    (positive : 0 < candidateBase) (constantWire : initial 0 = 1) :
    prefixWitness field duplexBase u64Base candidateBase initialBuilder
        coordinate initial processed 0 =
      1 := by
  rw [prefixWitness_before_candidateBase field duplexBase u64Base
    candidateBase initialBuilder coordinate initial processed positive]
  exact constantWire

/-- The assignment immediately after candidate `index` satisfies that
candidate's complete 25-row program. -/
theorem stage_complete
    (field : FieldInverse)
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat)
    (positive : 0 < candidateBase) (constantWire : initial 0 = 1)
    (sourcesBelow :
      SourcesBelow duplexBase u64Base candidateBase initialBuilder coordinate)
    (sourceBits :
      ∀ candidate : Fin candidatesPerScalar,
        SourceBitsBoolean initial
          (candidateLayout duplexBase u64Base candidateBase initialBuilder
            coordinate candidate))
    (index : Nat) (bounded : index < candidatesPerScalar) :
    Satisfies
      (rows
        (candidateLayout duplexBase u64Base candidateBase initialBuilder
          coordinate (candidateOfNat index)))
      (prefixWitness field duplexBase u64Base candidateBase initialBuilder
        coordinate initial (index + 1)) := by
  rw [prefixWitness_succ field duplexBase u64Base candidateBase initialBuilder
    coordinate initial bounded]
  apply complete field
  · simp only [candidateLayout, occurrenceBase, occurrenceIndex]
    omega
  · exact prefixWitness_constant field duplexBase u64Base candidateBase
      initialBuilder coordinate initial index positive constantWire
  · exact inputsBelowBase duplexBase u64Base candidateBase initialBuilder
      coordinate sourcesBelow (candidateOfNat index)
  · exact prefixWitness_sourceBits field duplexBase u64Base candidateBase
      initialBuilder coordinate initial index sourcesBelow
      (candidateOfNat index) (sourceBits _)
  · rw [prefixWitness_prior_eval field duplexBase u64Base candidateBase
      initialBuilder coordinate initial sourcesBelow index bounded]
    have countBound :=
      honestAcceptedPrefix_le duplexBase u64Base candidateBase initialBuilder
        coordinate initial index
    change
      honestAcceptedPrefix duplexBase u64Base candidateBase initialBuilder
          coordinate initial index <
        64
    change index < 64 at bounded
    omega

theorem prefixWitness_canonical
    (field : FieldInverse)
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat)
    (initialCanonical : ∀ column, initial column < goldilocksP)
    (sourcesBelow :
      SourcesBelow duplexBase u64Base candidateBase initialBuilder coordinate)
    (sourceBits :
      ∀ candidate : Fin candidatesPerScalar,
        SourceBitsBoolean initial
          (candidateLayout duplexBase u64Base candidateBase initialBuilder
            coordinate candidate)) :
    ∀ processed, processed ≤ candidatesPerScalar →
      ∀ column,
        prefixWitness field duplexBase u64Base candidateBase initialBuilder
            coordinate initial processed column <
          goldilocksP := by
  intro processed bounded
  induction processed with
  | zero => exact initialCanonical
  | succ processed hypothesis =>
      have processedLt : processed < candidatesPerScalar := by omega
      rw [prefixWitness_succ field duplexBase u64Base candidateBase
        initialBuilder coordinate initial processedLt]
      apply witness_canonical
      · exact hypothesis (by omega)
      · exact prefixWitness_sourceBits field duplexBase u64Base candidateBase
          initialBuilder coordinate initial processed sourcesBelow
          (candidateOfNat processed) (sourceBits _)
      · rw [prefixWitness_prior_eval field duplexBase u64Base candidateBase
          initialBuilder coordinate initial sourcesBelow processed processedLt]
        have countBound :=
          honestAcceptedPrefix_le duplexBase u64Base candidateBase
            initialBuilder coordinate initial processed
        change
          honestAcceptedPrefix duplexBase u64Base candidateBase initialBuilder
              coordinate initial processed <
            64
        change processed < 64 at processedLt
        omega

def prefixBoundary
    (candidateBase : Nat) {count : Nat} (coordinate : Fin count)
    (processed : Nat) : Nat :=
  candidateBase +
    (coordinate.val * candidatesPerScalar + processed) * auxiliaryCount

theorem prefixWitness_stable
    (field : FieldInverse)
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat)
    {start finish : Nat} (ordered : start ≤ finish)
    (finishBounded : finish ≤ candidatesPerScalar)
    {column : Nat}
    (before : column < prefixBoundary candidateBase coordinate start) :
    prefixWitness field duplexBase u64Base candidateBase initialBuilder
        coordinate initial finish column =
      prefixWitness field duplexBase u64Base candidateBase initialBuilder
        coordinate initial start column := by
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
        have finishLt : finish < candidatesPerScalar := by omega
        rw [prefixWitness_succ field duplexBase u64Base candidateBase
          initialBuilder coordinate initial finishLt]
        rw [witness_before]
        · exact hypothesis startLe (by omega) before
        · change
            column <
              occurrenceBase candidateBase coordinate (candidateOfNat finish)
          unfold prefixBoundary at before
          unfold occurrenceBase occurrenceIndex
          have candidateVal := candidateOfNat_val finishLt
          rw [candidateVal]
          simp only [candidatesPerScalar, auxiliaryCount] at before ⊢
          omega

private def CombBelow (layout : Layout) (comb : LinComb) : Prop :=
  ∀ column, Mentions comb column → column < layout.base + auxiliaryCount

private def RowBelow (layout : Layout) (row : Row) : Prop :=
  CombBelow layout row.a ∧ CombBelow layout row.b ∧ CombBelow layout row.c

private theorem combBelow_nil (layout : Layout) :
    CombBelow layout [] := by
  intro column mentioned
  simp [Mentions] at mentioned

private theorem combBelow_single
    (layout : Layout) (column coefficient : Nat)
    (bounded : column < layout.base + auxiliaryCount) :
    CombBelow layout [(column, coefficient)] := by
  intro target mentioned
  have equal := (mentions_single column target coefficient).mp mentioned
  subst target
  exact bounded

private theorem combBelow_append
    (layout : Layout) (left right : LinComb)
    (leftBelow : CombBelow layout left)
    (rightBelow : CombBelow layout right) :
    CombBelow layout (left ++ right) := by
  intro column mentioned
  rw [mentions_append] at mentioned
  exact mentioned.elim (leftBelow column) (rightBelow column)

private theorem combBelow_pair
    (layout : Layout)
    (leftColumn leftCoefficient rightColumn rightCoefficient : Nat)
    (leftBound : leftColumn < layout.base + auxiliaryCount)
    (rightBound : rightColumn < layout.base + auxiliaryCount) :
    CombBelow layout
      [(leftColumn, leftCoefficient), (rightColumn, rightCoefficient)] := by
  exact
    combBelow_append layout
      [(leftColumn, leftCoefficient)] [(rightColumn, rightCoefficient)]
      (combBelow_single layout leftColumn leftCoefficient leftBound)
      (combBelow_single layout rightColumn rightCoefficient rightBound)

private theorem chunkTerms_below
    (layout : Layout) (below : InputsBelowBase layout) :
    CombBelow layout (chunkTerms layout) := by
  intro column mentioned
  unfold chunkTerms Mentions at mentioned
  rw [List.map_map] at mentioned
  change
    column ∈ (List.finRange sourceBitCount).map layout.sourceBit at mentioned
  rcases List.mem_map.mp mentioned with ⟨index, _, rfl⟩
  have sourceLt := below.source index
  omega

private theorem quotientTerms_below (layout : Layout) :
    CombBelow layout (quotientTerms layout) := by
  intro column mentioned
  unfold quotientTerms Mentions at mentioned
  rw [List.map_map] at mentioned
  change
    column ∈
      (List.range quotientBitCount).map
        (quotientBitColumn layout) at mentioned
  rcases List.mem_map.mp mentioned with ⟨offset, inRange, rfl⟩
  have offsetLt := List.mem_range.mp inRange
  simp only [quotientBitColumn, quotientBitCount, auxiliaryCount] at offsetLt ⊢
  omega

private theorem prior_below
    (layout : Layout) (below : InputsBelowBase layout) :
    CombBelow layout layout.prior := by
  intro column mentioned
  unfold Mentions at mentioned
  rcases List.mem_map.mp mentioned with
    ⟨⟨termColumn, coefficient⟩, termMember, rfl⟩
  have priorLt := below.prior termColumn coefficient termMember
  omega

private theorem differenceTerms_below
    (layout : Layout) (positive : 0 < layout.base)
    (below : InputsBelowBase layout) :
    CombBelow layout (differenceTerms layout) := by
  unfold differenceTerms
  apply combBelow_append
  · exact chunkTerms_below layout below
  · apply combBelow_single
    simp only [auxiliaryCount]
    omega

private theorem oneMinusAccept_below
    (layout : Layout) (positive : 0 < layout.base) :
    CombBelow layout (oneMinusAccept layout) := by
  unfold oneMinusAccept
  apply combBelow_pair
  · simp only [acceptColumn, auxiliaryCount]
    omega
  · simp only [auxiliaryCount]
    omega

private theorem priorAccept_below
    (layout : Layout) (below : InputsBelowBase layout) :
    CombBelow layout
      (layout.prior ++ [(acceptColumn layout, 1)]) := by
  apply combBelow_append
  · exact prior_below layout below
  · apply combBelow_single
    simp only [acceptColumn, auxiliaryCount]
    omega

private theorem bitRow_below
    (layout : Layout) (positive : 0 < layout.base)
    (column : Nat) (bounded : column < layout.base + auxiliaryCount) :
    RowBelow layout (bitRow column) := by
  unfold RowBelow bitRow
  refine ⟨combBelow_single layout column 1 bounded, ?_, combBelow_nil layout⟩
  apply combBelow_pair
  · exact bounded
  · simp only [auxiliaryCount]
    omega

private theorem acceptanceRows_below
    (layout : Layout) (positive : 0 < layout.base)
    (below : InputsBelowBase layout)
    (row : Row) (member : row ∈ acceptanceRows layout) :
    RowBelow layout row := by
  simp only [acceptanceRows, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl
  · apply bitRow_below layout positive
    simp only [acceptColumn, auxiliaryCount]
    omega
  · exact
      ⟨oneMinusAccept_below layout positive,
        differenceTerms_below layout positive below,
        combBelow_nil layout⟩
  · exact
      ⟨differenceTerms_below layout positive below,
        combBelow_single layout (inverseColumn layout) 1 (by
          simp only [inverseColumn, auxiliaryCount]
          omega),
        combBelow_single layout (acceptColumn layout) 1 (by
          simp only [acceptColumn, auxiliaryCount]
          omega)⟩
  · exact
      ⟨oneMinusAccept_below layout positive,
        combBelow_single layout (inverseColumn layout) 1 (by
          simp only [inverseColumn, auxiliaryCount]
          omega),
        combBelow_nil layout⟩

private theorem residueRangeRows_below
    (layout : Layout) (positive : 0 < layout.base)
    (row : Row) (member : row ∈ residueRangeRows layout) :
    RowBelow layout row := by
  simp only [residueRangeRows, List.mem_cons, List.not_mem_nil, or_false]
    at member
  rcases member with rfl | rfl | rfl | rfl
  all_goals
    unfold RowBelow
    refine ⟨combBelow_single layout _ 1 (by
      simp only [residueColumn, productColumn, auxiliaryCount]
      omega), ?_, ?_⟩
  · apply combBelow_pair
    · simp only [residueColumn, auxiliaryCount]
      omega
    · simp only [auxiliaryCount]
      omega
  · apply combBelow_single
    simp only [productColumn, auxiliaryCount]
    omega
  · apply combBelow_pair
    · simp only [residueColumn, auxiliaryCount]
      omega
    · simp only [auxiliaryCount]
      omega
  · apply combBelow_single
    simp only [productColumn, auxiliaryCount]
    omega
  · apply combBelow_pair
    · simp only [residueColumn, auxiliaryCount]
      omega
    · simp only [auxiliaryCount]
      omega
  · apply combBelow_single
    simp only [productColumn, auxiliaryCount]
    omega
  · apply combBelow_pair
    · simp only [residueColumn, auxiliaryCount]
      omega
    · simp only [auxiliaryCount]
      omega
  · exact combBelow_nil layout

private theorem quotientBitRows_below
    (layout : Layout) (positive : 0 < layout.base)
    (row : Row) (member : row ∈ quotientBitRows layout) :
    RowBelow layout row := by
  rcases List.mem_map.mp member with ⟨offset, inRange, rfl⟩
  apply bitRow_below layout positive
  have offsetLt := List.mem_range.mp inRange
  simp only [quotientBitColumn, quotientBitCount, auxiliaryCount] at offsetLt ⊢
  omega

private theorem quotientRecompositionRow_below
    (layout : Layout) (positive : 0 < layout.base) :
    RowBelow layout (quotientRecompositionRow layout) := by
  exact
    ⟨combBelow_single layout (quotientColumn layout) 1 (by
        simp only [quotientColumn, auxiliaryCount]
        omega),
      combBelow_single layout 0 1 (by
        simp only [auxiliaryCount]
        omega),
      quotientTerms_below layout⟩

private theorem decompositionRow_below
    (layout : Layout) (positive : 0 < layout.base)
    (below : InputsBelowBase layout) :
    RowBelow layout (decompositionRow layout) := by
  exact
    ⟨chunkTerms_below layout below,
      combBelow_single layout 0 1 (by
        simp only [auxiliaryCount]
        omega),
      combBelow_append layout
        [(quotientColumn layout, 5)] [(residueColumn layout, 1)]
        (combBelow_single layout (quotientColumn layout) 5 (by
          simp only [quotientColumn, auxiliaryCount]
          omega))
        (combBelow_single layout (residueColumn layout) 1 (by
          simp only [residueColumn, auxiliaryCount]
          omega))⟩

private theorem cumulativeRow_below
    (layout : Layout) (positive : 0 < layout.base)
    (below : InputsBelowBase layout) :
    RowBelow layout (cumulativeRow layout) := by
  exact
    ⟨combBelow_single layout (cumulativeColumn layout) 1 (by
        simp only [cumulativeColumn, auxiliaryCount]
        omega),
      combBelow_single layout 0 1 (by
        simp only [auxiliaryCount]
        omega),
      priorAccept_below layout below⟩

/-- Every operand of one candidate occurrence lies before the next occurrence
base.  This is the exact frame condition used to preserve earlier rows while
later witnesses are written. -/
theorem candidateRows_mentions_lt
    (layout : Layout) (positive : 0 < layout.base)
    (below : InputsBelowBase layout)
    (row : Row)
    (member : row ∈ PiRlcCanonicalCandidate.rows layout)
    (column : Nat)
    (mentioned :
      Mentions row.a column ∨ Mentions row.b column ∨
        Mentions row.c column) :
    column < layout.base + auxiliaryCount := by
  have rowBelow : RowBelow layout row := by
    unfold PiRlcCanonicalCandidate.rows at member
    rcases List.mem_append.mp member with inHead | inTail
    · rcases List.mem_append.mp inHead with inAcceptanceResidue |
        inQuotientBits
      · rcases List.mem_append.mp inAcceptanceResidue with inAcceptance |
          inResidue
        · exact acceptanceRows_below layout positive below row inAcceptance
        · exact residueRangeRows_below layout positive row inResidue
      · exact quotientBitRows_below layout positive row inQuotientBits
    · simp only [List.mem_cons, List.not_mem_nil, or_false] at inTail
      rcases inTail with rfl | rfl | rfl
      · exact quotientRecompositionRow_below layout positive
      · exact decompositionRow_below layout positive below
      · exact cumulativeRow_below layout positive below
  rcases rowBelow with ⟨aBelow, bBelow, cBelow⟩
  exact mentioned.elim (aBelow column)
    (fun right => right.elim (bBelow column) (cBelow column))

private theorem rowHolds_congr
    (left right : Nat → Nat) (row : Row)
    (agree :
      ∀ column,
        Mentions row.a column ∨ Mentions row.b column ∨
          Mentions row.c column →
        left column = right column) :
    RowHolds left row ↔ RowHolds right row := by
  unfold RowHolds
  rw [Nightstream.Implementation.R1CS.Canonical.KMulHonest.lcEval_congr
      left right row.a (fun column member => agree column (Or.inl member)),
    Nightstream.Implementation.R1CS.Canonical.KMulHonest.lcEval_congr
      left right row.b
      (fun column member => agree column (Or.inr (Or.inl member))),
    Nightstream.Implementation.R1CS.Canonical.KMulHonest.lcEval_congr
      left right row.c
      (fun column member => agree column (Or.inr (Or.inr member)))]

/-- Honest completeness of all 1,600 candidate rows for one scalar. -/
theorem scalarRows_complete
    (field : FieldInverse)
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat)
    (positive : 0 < candidateBase) (constantWire : initial 0 = 1)
    (sourcesBelow :
      SourcesBelow duplexBase u64Base candidateBase initialBuilder coordinate)
    (sourceBits :
      ∀ candidate : Fin candidatesPerScalar,
        SourceBitsBoolean initial
          (candidateLayout duplexBase u64Base candidateBase initialBuilder
            coordinate candidate)) :
    Satisfies
      (PiRlcCanonicalCandidates.scalarRows duplexBase u64Base candidateBase
        initialBuilder coordinate)
      (prefixWitness field duplexBase u64Base candidateBase initialBuilder
        coordinate initial candidatesPerScalar) := by
  intro row member
  rcases List.mem_flatMap.mp member with
    ⟨candidate, _, rowMember⟩
  have stage :=
    stage_complete field duplexBase u64Base candidateBase initialBuilder
      coordinate initial positive constantWire sourcesBelow sourceBits
      candidate.val candidate.isLt
  have rowMemberAt :
      row ∈
        rows
          (candidateLayout duplexBase u64Base candidateBase initialBuilder
            coordinate (candidateOfNat candidate.val)) := by
    simpa [candidateOfNat_eq candidate] using rowMember
  have stageHolds := stage row rowMemberAt
  apply (rowHolds_congr
    (prefixWitness field duplexBase u64Base candidateBase initialBuilder
      coordinate initial (candidate.val + 1))
    (prefixWitness field duplexBase u64Base candidateBase initialBuilder
      coordinate initial candidatesPerScalar)
    row ?_).mp stageHolds
  intro column mentioned
  symm
  apply prefixWitness_stable field duplexBase u64Base candidateBase
    initialBuilder coordinate initial
    (start := candidate.val + 1) (finish := candidatesPerScalar)
  · omega
  · exact Nat.le_refl _
  · have localBound :=
      candidateRows_mentions_lt
        (candidateLayout duplexBase u64Base candidateBase initialBuilder
          coordinate candidate)
        (by
          simp only [candidateLayout, occurrenceBase, occurrenceIndex]
          omega)
        (inputsBelowBase duplexBase u64Base candidateBase initialBuilder
          coordinate sourcesBelow candidate)
        row rowMember column mentioned
    simp only [candidateLayout, occurrenceBase, occurrenceIndex,
      prefixBoundary, auxiliaryCount, candidatesPerScalar] at localBound ⊢
    omega

end Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidatesHonest
