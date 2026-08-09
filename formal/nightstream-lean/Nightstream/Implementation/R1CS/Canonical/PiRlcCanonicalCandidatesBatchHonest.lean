import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidatesHonest
import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalScalarComplete
import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalU64Placement

/-!
Contract: honest witness construction for the family-major batch of canonical
`Pi_RLC` candidate classifiers.

The upstream source bits are the exact final canonical-u64 witness bits.  A
single physical separation inequality constructs candidate source ownership.
Candidate witnesses are then threaded across coordinates in the same order as
`PiRlcCanonicalCandidates.rows`: all 64 candidates of coordinate zero, then
all 64 candidates of coordinate one, and so on.

Assurance tier: model-level canonical encoding.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidatesBatchHonest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeHonest
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidate
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidateSound
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

/-- First column not owned by the complete canonical-u64 batch. -/
def u64End (u64Base count : Nat) : Nat :=
  u64Base +
    count * PiRlcCanonicalU64.lanesPerScalar *
      CanonicalU64Recipe.auxiliaryCount

/-- Exact u64/candidate separation constructs every candidate source bound. -/
theorem sourcesBelow_of_u64End
    (duplexBase u64Base candidateBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (separated : u64End u64Base count ≤ candidateBase) :
    ∀ coordinate : Fin count,
      PiRlcCanonicalCandidatesHonest.SourcesBelow
        duplexBase u64Base candidateBase initialBuilder coordinate := by
  intro coordinate
  constructor
  intro candidate bit
  have coordinateLt := coordinate.isLt
  have positionLt :=
    (PiRlcCanonicalCandidates.lanePosition candidate).isLt
  have indexLt :=
    PiRlcCanonicalCandidates.sourceBitIndex_lt candidate bit
  change
    u64Base +
          (coordinate.val * PiRlcCanonicalU64.lanesPerScalar +
              (PiRlcCanonicalCandidates.lanePosition candidate).val) *
            CanonicalU64Recipe.auxiliaryCount +
        PiRlcCanonicalCandidates.sourceBitIndex candidate bit <
      candidateBase
  change
    (PiRlcCanonicalCandidates.lanePosition candidate).val <
      PiRlcCanonicalU64.lanesPerScalar at positionLt
  unfold u64End at separated
  simp only [PiRlcCanonicalU64.lanesPerScalar,
    CanonicalU64Recipe.auxiliaryCount] at separated positionLt ⊢
  have occurrenceLt :
      coordinate.val * PiRlcCanonicalU64.lanesPerScalar +
          (PiRlcCanonicalCandidates.lanePosition candidate).val <
        count * PiRlcCanonicalU64.lanesPerScalar := by
    simp only [PiRlcCanonicalU64.lanesPerScalar]
    omega
  omega

/-- The final u64 witness makes every candidate source bit Boolean. -/
theorem sourceBitsBoolean_of_u64Witness
    (field : FieldInverse)
    (duplexBase u64Base candidateBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    (below :
      PiRlcCanonicalU64Honest.InputsBelow
        duplexBase u64Base count initialBuilder)
    (coordinate : Fin count)
    (candidate : Fin PiRlcCanonicalCandidates.candidatesPerScalar) :
    SourceBitsBoolean
      (PiRlcCanonicalU64Honest.batchWitness
        field duplexBase u64Base count initialBuilder initial)
      (PiRlcCanonicalCandidates.candidateLayout
        duplexBase u64Base candidateBase initialBuilder coordinate candidate) := by
  intro bit
  change
    PiRlcCanonicalU64Honest.batchWitness
        field duplexBase u64Base count initialBuilder initial
        (CanonicalU64Recipe.bitColumn
          (PiRlcCanonicalU64.laneLayout
            duplexBase u64Base initialBuilder coordinate
              (PiRlcCanonicalCandidates.lanePosition candidate))
          (PiRlcCanonicalCandidates.sourceBitIndex candidate bit)) ≤
      1
  rw [PiRlcCanonicalU64Honest.batchWitness_bit
    field duplexBase u64Base count initialBuilder initial below coordinate
    (PiRlcCanonicalCandidates.lanePosition candidate)
    (PiRlcCanonicalCandidates.sourceBitIndex_lt candidate bit)]
  exact sourceBit_le_one _ _

/-! ## Coordinate-major candidate witness threading -/

/-- Boundary after the first `processed` candidate coordinates. -/
def batchPrefixBoundary (candidateBase processed : Nat) : Nat :=
  candidateBase +
    processed * PiRlcCanonicalCandidates.candidatesPerScalar *
      auxiliaryCount

/-- One complete scalar witness preserves every preceding coordinate. -/
theorem scalarWitness_before_scalarBase
    (field : FieldInverse)
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat)
    {column : Nat}
    (before : column < batchPrefixBoundary candidateBase coordinate.val) :
    PiRlcCanonicalCandidatesHonest.prefixWitness
        field duplexBase u64Base candidateBase initialBuilder coordinate
        initial PiRlcCanonicalCandidates.candidatesPerScalar column =
      initial column := by
  have stable :=
    PiRlcCanonicalCandidatesHonest.prefixWitness_stable
      field duplexBase u64Base candidateBase initialBuilder coordinate initial
      (start := 0)
      (finish := PiRlcCanonicalCandidates.candidatesPerScalar)
      (Nat.zero_le _) (Nat.le_refl _) (column := column)
  apply stable
  simpa only [PiRlcCanonicalCandidatesHonest.prefixBoundary,
    batchPrefixBoundary, PiRlcCanonicalCandidates.candidatesPerScalar,
    auxiliaryCount, Nat.mul_zero, Nat.add_zero, Nat.zero_mul] using before

/-- Apply complete candidate witnesses to the first `processed` coordinates. -/
def batchPrefixWitness
    (field : FieldInverse)
    (duplexBase u64Base candidateBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat) : Nat → Nat → Nat
  | 0 => initial
  | processed + 1 =>
      if bounded : processed < count then
        PiRlcCanonicalCandidatesHonest.prefixWitness
          field duplexBase u64Base candidateBase initialBuilder
          ⟨processed, bounded⟩
          (batchPrefixWitness field duplexBase u64Base candidateBase count
            initialBuilder initial processed)
          PiRlcCanonicalCandidates.candidatesPerScalar
      else
        batchPrefixWitness field duplexBase u64Base candidateBase count
          initialBuilder initial processed

theorem batchPrefixWitness_succ
    (field : FieldInverse)
    (duplexBase u64Base candidateBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    {processed : Nat} (bounded : processed < count) :
    batchPrefixWitness field duplexBase u64Base candidateBase count
        initialBuilder initial (processed + 1) =
      PiRlcCanonicalCandidatesHonest.prefixWitness
        field duplexBase u64Base candidateBase initialBuilder
        ⟨processed, bounded⟩
        (batchPrefixWitness field duplexBase u64Base candidateBase count
          initialBuilder initial processed)
        PiRlcCanonicalCandidates.candidatesPerScalar := by
  simp [batchPrefixWitness, bounded]

theorem batchPrefixWitness_before_candidateBase
    (field : FieldInverse)
    (duplexBase u64Base candidateBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat) (processed : Nat)
    {column : Nat} (before : column < candidateBase) :
    batchPrefixWitness field duplexBase u64Base candidateBase count
        initialBuilder initial processed column =
      initial column := by
  induction processed with
  | zero => rfl
  | succ processed hypothesis =>
      simp only [batchPrefixWitness]
      split
      · rw [PiRlcCanonicalCandidatesHonest.prefixWitness_before_candidateBase]
        · exact hypothesis
        · exact before
      · exact hypothesis

private theorem sourceBits_preserved
    (field : FieldInverse)
    (duplexBase u64Base candidateBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    (sourcesBelow :
      ∀ coordinate : Fin count,
        PiRlcCanonicalCandidatesHonest.SourcesBelow
          duplexBase u64Base candidateBase initialBuilder coordinate)
    (sourceBits :
      ∀ (coordinate : Fin count)
        (candidate : Fin PiRlcCanonicalCandidates.candidatesPerScalar),
        SourceBitsBoolean initial
          (PiRlcCanonicalCandidates.candidateLayout duplexBase u64Base
            candidateBase initialBuilder coordinate candidate))
    (processed : Nat) (coordinate : Fin count)
    (candidate : Fin PiRlcCanonicalCandidates.candidatesPerScalar) :
    SourceBitsBoolean
      (batchPrefixWitness field duplexBase u64Base candidateBase count
        initialBuilder initial processed)
      (PiRlcCanonicalCandidates.candidateLayout duplexBase u64Base
        candidateBase initialBuilder coordinate candidate) := by
  intro bit
  rw [batchPrefixWitness_before_candidateBase
    field duplexBase u64Base candidateBase count initialBuilder initial
    processed
    ((sourcesBelow coordinate).source candidate bit)]
  exact sourceBits coordinate candidate bit

theorem batchPrefixWitness_canonical
    (field : FieldInverse)
    (duplexBase u64Base candidateBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    (initialCanonical : ∀ column, initial column < goldilocksP)
    (sourcesBelow :
      ∀ coordinate : Fin count,
        PiRlcCanonicalCandidatesHonest.SourcesBelow
          duplexBase u64Base candidateBase initialBuilder coordinate)
    (sourceBits :
      ∀ (coordinate : Fin count)
        (candidate : Fin PiRlcCanonicalCandidates.candidatesPerScalar),
        SourceBitsBoolean initial
          (PiRlcCanonicalCandidates.candidateLayout duplexBase u64Base
            candidateBase initialBuilder coordinate candidate)) :
    ∀ processed column,
      batchPrefixWitness field duplexBase u64Base candidateBase count
        initialBuilder initial processed column < goldilocksP := by
  intro processed
  induction processed with
  | zero => exact initialCanonical
  | succ processed hypothesis =>
      simp only [batchPrefixWitness]
      split
      next bounded =>
        apply PiRlcCanonicalCandidatesHonest.prefixWitness_canonical
          field duplexBase u64Base candidateBase initialBuilder
          ⟨processed, bounded⟩
          (batchPrefixWitness field duplexBase u64Base candidateBase count
            initialBuilder initial processed)
          hypothesis (sourcesBelow ⟨processed, bounded⟩)
        · intro candidate
          exact sourceBits_preserved field duplexBase u64Base candidateBase
            count initialBuilder initial sourcesBelow sourceBits processed
            ⟨processed, bounded⟩ candidate
        · exact Nat.le_refl _
      next =>
        exact hypothesis

theorem batchPrefixWitness_stable
    (field : FieldInverse)
    (duplexBase u64Base candidateBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    {start finish : Nat} (ordered : start ≤ finish)
    (finishBounded : finish ≤ count)
    {column : Nat}
    (before : column < batchPrefixBoundary candidateBase start) :
    batchPrefixWitness field duplexBase u64Base candidateBase count
        initialBuilder initial finish column =
      batchPrefixWitness field duplexBase u64Base candidateBase count
        initialBuilder initial start column := by
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
        rw [batchPrefixWitness_succ field duplexBase u64Base candidateBase
          count initialBuilder initial finishLt]
        rw [scalarWitness_before_scalarBase]
        · exact hypothesis startLe (by omega) before
        · simp only [batchPrefixBoundary,
            PiRlcCanonicalCandidates.candidatesPerScalar,
            auxiliaryCount] at before ⊢
          omega

/-- Final family-major candidate witness. -/
def batchWitness
    (field : FieldInverse)
    (duplexBase u64Base candidateBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat) : Nat → Nat :=
  batchPrefixWitness field duplexBase u64Base candidateBase count
    initialBuilder initial count

theorem batchWitness_before_candidateBase
    (field : FieldInverse)
    (duplexBase u64Base candidateBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    {column : Nat} (before : column < candidateBase) :
    batchWitness field duplexBase u64Base candidateBase count initialBuilder
        initial column =
      initial column :=
  batchPrefixWitness_before_candidateBase field duplexBase u64Base
    candidateBase count initialBuilder initial count before

theorem batchWitness_canonical
    (field : FieldInverse)
    (duplexBase u64Base candidateBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    (initialCanonical : ∀ column, initial column < goldilocksP)
    (sourcesBelow :
      ∀ coordinate : Fin count,
        PiRlcCanonicalCandidatesHonest.SourcesBelow
          duplexBase u64Base candidateBase initialBuilder coordinate)
    (sourceBits :
      ∀ (coordinate : Fin count)
        (candidate : Fin PiRlcCanonicalCandidates.candidatesPerScalar),
        SourceBitsBoolean initial
          (PiRlcCanonicalCandidates.candidateLayout duplexBase u64Base
            candidateBase initialBuilder coordinate candidate)) :
    ∀ column,
      batchWitness field duplexBase u64Base candidateBase count initialBuilder
        initial column < goldilocksP :=
  batchPrefixWitness_canonical field duplexBase u64Base candidateBase count
    initialBuilder initial initialCanonical sourcesBelow sourceBits count

theorem scalarRows_mentions_lt
    (duplexBase u64Base candidateBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (sourcesBelow :
      ∀ coordinate : Fin count,
        PiRlcCanonicalCandidatesHonest.SourcesBelow
          duplexBase u64Base candidateBase initialBuilder coordinate)
    (coordinate : Fin count)
    (positive : 0 < candidateBase)
    (row : Row)
    (rowMember :
      row ∈ PiRlcCanonicalCandidates.scalarRows
        duplexBase u64Base candidateBase initialBuilder coordinate)
    (column : Nat)
    (mentioned :
      Mentions row.a column ∨ Mentions row.b column ∨
        Mentions row.c column) :
    column < batchPrefixBoundary candidateBase (coordinate.val + 1) := by
  rcases List.mem_flatMap.mp rowMember with
    ⟨candidate, _, localMember⟩
  have localPositive :
      0 <
        (PiRlcCanonicalCandidates.candidateLayout duplexBase u64Base
          candidateBase initialBuilder coordinate candidate).base := by
    simp only [PiRlcCanonicalCandidates.candidateLayout,
      PiRlcCanonicalCandidates.occurrenceBase,
      PiRlcCanonicalCandidates.occurrenceIndex]
    omega
  have localBound :=
    PiRlcCanonicalCandidatesHonest.candidateRows_mentions_lt
      (PiRlcCanonicalCandidates.candidateLayout duplexBase u64Base
        candidateBase initialBuilder coordinate candidate)
      localPositive
      (PiRlcCanonicalCandidatesHonest.inputsBelowBase duplexBase u64Base
        candidateBase initialBuilder coordinate
        (sourcesBelow coordinate) candidate)
      row localMember column mentioned
  simp only [PiRlcCanonicalCandidates.candidateLayout,
    PiRlcCanonicalCandidates.occurrenceBase,
    PiRlcCanonicalCandidates.occurrenceIndex, batchPrefixBoundary,
    PiRlcCanonicalCandidates.candidatesPerScalar, auxiliaryCount]
    at localBound ⊢
  have candidateLt := candidate.isLt
  omega

private theorem rowHolds_congr
    (left right : Nat → Nat) (row : Row)
    (agree :
      ∀ column,
        Mentions row.a column ∨ Mentions row.b column ∨
          Mentions row.c column →
        left column = right column) :
    RowHolds left row ↔ RowHolds right row := by
  unfold RowHolds
  rw [KMulHonest.lcEval_congr left right row.a
      (fun column member => agree column (Or.inl member)),
    KMulHonest.lcEval_congr left right row.b
      (fun column member => agree column (Or.inr (Or.inl member))),
    KMulHonest.lcEval_congr left right row.c
      (fun column member => agree column (Or.inr (Or.inr member)))]

theorem batchStage_complete
    (field : FieldInverse)
    (duplexBase u64Base candidateBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    (positive : 0 < candidateBase) (constantWire : initial 0 = 1)
    (sourcesBelow :
      ∀ coordinate : Fin count,
        PiRlcCanonicalCandidatesHonest.SourcesBelow
          duplexBase u64Base candidateBase initialBuilder coordinate)
    (sourceBits :
      ∀ (coordinate : Fin count)
        (candidate : Fin PiRlcCanonicalCandidates.candidatesPerScalar),
        SourceBitsBoolean initial
          (PiRlcCanonicalCandidates.candidateLayout duplexBase u64Base
            candidateBase initialBuilder coordinate candidate))
    {processed : Nat} (processedLt : processed < count) :
    Satisfies
      (PiRlcCanonicalCandidates.scalarRows duplexBase u64Base candidateBase
        initialBuilder ⟨processed, processedLt⟩)
      (batchPrefixWitness field duplexBase u64Base candidateBase count
        initialBuilder initial (processed + 1)) := by
  let prior :=
    batchPrefixWitness field duplexBase u64Base candidateBase count
      initialBuilder initial processed
  have priorConstant : prior 0 = 1 := by
    change
      batchPrefixWitness field duplexBase u64Base candidateBase count
        initialBuilder initial processed 0 = 1
    rw [batchPrefixWitness_before_candidateBase
      field duplexBase u64Base candidateBase count initialBuilder initial
      processed positive]
    exact constantWire
  rw [batchPrefixWitness_succ field duplexBase u64Base candidateBase count
    initialBuilder initial processedLt]
  apply PiRlcCanonicalCandidatesHonest.scalarRows_complete
    field duplexBase u64Base candidateBase initialBuilder
    ⟨processed, processedLt⟩ prior positive priorConstant
    (sourcesBelow ⟨processed, processedLt⟩)
  intro candidate
  exact sourceBits_preserved field duplexBase u64Base candidateBase count
    initialBuilder initial sourcesBelow sourceBits processed
    ⟨processed, processedLt⟩ candidate

/-- Honest completeness of the exact family-major candidate batch. -/
theorem rows_complete
    (field : FieldInverse)
    (duplexBase u64Base candidateBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    (positive : 0 < candidateBase) (constantWire : initial 0 = 1)
    (sourcesBelow :
      ∀ coordinate : Fin count,
        PiRlcCanonicalCandidatesHonest.SourcesBelow
          duplexBase u64Base candidateBase initialBuilder coordinate)
    (sourceBits :
      ∀ (coordinate : Fin count)
        (candidate : Fin PiRlcCanonicalCandidates.candidatesPerScalar),
        SourceBitsBoolean initial
          (PiRlcCanonicalCandidates.candidateLayout duplexBase u64Base
            candidateBase initialBuilder coordinate candidate)) :
    Satisfies
      (PiRlcCanonicalCandidates.rows duplexBase u64Base candidateBase count
        initialBuilder)
      (batchWitness field duplexBase u64Base candidateBase count initialBuilder
        initial) := by
  intro row member
  rcases List.mem_flatMap.mp member with
    ⟨coordinate, _, rowMember⟩
  have stage :=
    batchStage_complete field duplexBase u64Base candidateBase count
      initialBuilder initial positive constantWire sourcesBelow sourceBits
      coordinate.isLt
  have stageHolds := stage row (by simpa using rowMember)
  apply
    (rowHolds_congr
      (batchPrefixWitness field duplexBase u64Base candidateBase count
        initialBuilder initial (coordinate.val + 1))
      (batchWitness field duplexBase u64Base candidateBase count initialBuilder
        initial)
      row ?_).mp
  · exact stageHolds
  · intro column mentioned
    symm
    apply batchPrefixWitness_stable field duplexBase u64Base candidateBase
      count initialBuilder initial
      (start := coordinate.val + 1) (finish := count)
    · omega
    · exact Nat.le_refl _
    · exact scalarRows_mentions_lt duplexBase u64Base candidateBase count
        initialBuilder sourcesBelow coordinate positive row rowMember column
        mentioned

/-! ## Candidate values consumed by the selector batch -/

/-- Assignment immediately before one coordinate's candidate block. -/
def coordinatePrior
    (field : FieldInverse)
    (duplexBase u64Base candidateBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat) (coordinate : Fin count) : Nat → Nat :=
  batchPrefixWitness field duplexBase u64Base candidateBase count
    initialBuilder initial coordinate.val

/-- Boolean sources for one coordinate, transported through all earlier
candidate coordinates. -/
def coordinateSourceBits
    (field : FieldInverse)
    (duplexBase u64Base candidateBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    (sourcesBelow :
      ∀ coordinate : Fin count,
        PiRlcCanonicalCandidatesHonest.SourcesBelow
          duplexBase u64Base candidateBase initialBuilder coordinate)
    (sourceBits :
      ∀ (coordinate : Fin count)
        (candidate : Fin PiRlcCanonicalCandidates.candidatesPerScalar),
        SourceBitsBoolean initial
          (PiRlcCanonicalCandidates.candidateLayout duplexBase u64Base
            candidateBase initialBuilder coordinate candidate))
    (coordinate : Fin count) :
    ∀ candidate : Fin PiRlcCanonicalCandidates.candidatesPerScalar,
      SourceBitsBoolean
        (coordinatePrior field duplexBase u64Base candidateBase count
          initialBuilder initial coordinate)
        (PiRlcCanonicalCandidates.candidateLayout duplexBase u64Base
          candidateBase initialBuilder coordinate candidate) :=
  fun candidate =>
    sourceBits_preserved field duplexBase u64Base candidateBase count
      initialBuilder initial sourcesBelow sourceBits coordinate.val coordinate
      candidate

/-- Exact verifier-owned candidate list for one coordinate. -/
def coordinateCandidates
    (field : FieldInverse)
    (duplexBase u64Base candidateBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    (sourcesBelow :
      ∀ coordinate : Fin count,
        PiRlcCanonicalCandidatesHonest.SourcesBelow
          duplexBase u64Base candidateBase initialBuilder coordinate)
    (sourceBits :
      ∀ (coordinate : Fin count)
        (candidate : Fin PiRlcCanonicalCandidates.candidatesPerScalar),
        SourceBitsBoolean initial
          (PiRlcCanonicalCandidates.candidateLayout duplexBase u64Base
            candidateBase initialBuilder coordinate candidate))
    (coordinate : Fin count) :
    List ProductionAlphabet.Chunk :=
  PiRlcCanonicalScalarComplete.honestCandidates duplexBase u64Base
    candidateBase initialBuilder coordinate
    (coordinatePrior field duplexBase u64Base candidateBase count
      initialBuilder initial coordinate)
    (coordinateSourceBits field duplexBase u64Base candidateBase count
      initialBuilder initial sourcesBelow sourceBits coordinate)

/-- Later candidate coordinates preserve every value read from this completed
coordinate. -/
theorem batchWitness_eq_after_coordinate
    (field : FieldInverse)
    (duplexBase u64Base candidateBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat) (coordinate : Fin count)
    {column : Nat}
    (before :
      column < batchPrefixBoundary candidateBase (coordinate.val + 1)) :
    batchWitness field duplexBase u64Base candidateBase count initialBuilder
        initial column =
      PiRlcCanonicalCandidatesHonest.prefixWitness
        field duplexBase u64Base candidateBase initialBuilder coordinate
        (coordinatePrior field duplexBase u64Base candidateBase count
          initialBuilder initial coordinate)
        PiRlcCanonicalCandidates.candidatesPerScalar column := by
  unfold batchWitness coordinatePrior
  rw [batchPrefixWitness_stable field duplexBase u64Base candidateBase count
    initialBuilder initial
    (start := coordinate.val + 1) (finish := count)
    (by omega) (Nat.le_refl _) before]
  exact congrFun
    (batchPrefixWitness_succ field duplexBase u64Base candidateBase count
      initialBuilder initial coordinate.isLt)
    column

/-- The final family-major candidate witness constructs the selector's exact
source-value contract for every coordinate. -/
theorem coordinateSourcesMatch
    (field : FieldInverse)
    (duplexBase u64Base candidateBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    (sourcesBelow :
      ∀ coordinate : Fin count,
        PiRlcCanonicalCandidatesHonest.SourcesBelow
          duplexBase u64Base candidateBase initialBuilder coordinate)
    (sourceBits :
      ∀ (coordinate : Fin count)
        (candidate : Fin PiRlcCanonicalCandidates.candidatesPerScalar),
        SourceBitsBoolean initial
          (PiRlcCanonicalCandidates.candidateLayout duplexBase u64Base
            candidateBase initialBuilder coordinate candidate))
    (coordinate : Fin count) :
    PiRlcCanonicalSelectorHonest.SourcesMatch
      duplexBase u64Base candidateBase initialBuilder coordinate
      (batchWitness field duplexBase u64Base candidateBase count initialBuilder
        initial)
      (coordinateCandidates field duplexBase u64Base candidateBase count
        initialBuilder initial sourcesBelow sourceBits coordinate) := by
  let prior :=
    coordinatePrior field duplexBase u64Base candidateBase count
      initialBuilder initial coordinate
  let bits :=
    coordinateSourceBits field duplexBase u64Base candidateBase count
      initialBuilder initial sourcesBelow sourceBits coordinate
  let candidates :=
    coordinateCandidates field duplexBase u64Base candidateBase count
      initialBuilder initial sourcesBelow sourceBits coordinate
  have stage :=
    PiRlcCanonicalScalarComplete.candidateSourcesMatch field duplexBase
      u64Base candidateBase initialBuilder coordinate prior
      (sourcesBelow coordinate) bits
  refine
    { lengthExact := stage.lengthExact
      accept := ?_
      symbol := ?_
      prefixExact := ?_
      finalCount := ?_ }
  · intro candidate
    rw [batchWitness_eq_after_coordinate field duplexBase u64Base
      candidateBase count initialBuilder initial coordinate
      (column :=
        PiRlcCanonicalSelector.acceptSource duplexBase u64Base candidateBase
          initialBuilder coordinate candidate) (by
        simp only [PiRlcCanonicalSelector.acceptSource,
          PiRlcCanonicalSelector.candidateSourceLayout,
          PiRlcCanonicalCandidate.acceptColumn,
          PiRlcCanonicalCandidates.candidateLayout,
          PiRlcCanonicalCandidates.occurrenceBase,
          PiRlcCanonicalCandidates.occurrenceIndex, batchPrefixBoundary,
          PiRlcCanonicalCandidates.candidatesPerScalar, auxiliaryCount]
        have candidateLt := candidate.isLt
        simp only [PiRlcCanonicalCandidates.candidatesPerScalar]
          at candidateLt
        omega)]
    exact stage.accept candidate
  · intro candidate
    rw [batchWitness_eq_after_coordinate field duplexBase u64Base
      candidateBase count initialBuilder initial coordinate
      (column :=
        PiRlcCanonicalSelector.symbolSource duplexBase u64Base candidateBase
          initialBuilder coordinate candidate) (by
        simp only [PiRlcCanonicalSelector.symbolSource,
          PiRlcCanonicalSelector.candidateSourceLayout,
          PiRlcCanonicalCandidate.residueColumn,
          PiRlcCanonicalCandidates.candidateLayout,
          PiRlcCanonicalCandidates.occurrenceBase,
          PiRlcCanonicalCandidates.occurrenceIndex, batchPrefixBoundary,
          PiRlcCanonicalCandidates.candidatesPerScalar, auxiliaryCount]
        have candidateLt := candidate.isLt
        simp only [PiRlcCanonicalCandidates.candidatesPerScalar]
          at candidateLt
        omega)]
    exact stage.symbol candidate
  · intro candidate
    rw [KMulHonest.lcEval_congr
      (batchWitness field duplexBase u64Base candidateBase count
        initialBuilder initial)
      (PiRlcCanonicalCandidatesHonest.prefixWitness field duplexBase u64Base
        candidateBase initialBuilder coordinate prior
        PiRlcCanonicalCandidates.candidatesPerScalar)
      (PiRlcCanonicalSelector.prefixSource duplexBase u64Base candidateBase
        initialBuilder coordinate candidate)]
    · exact stage.prefixExact candidate
    · intro column mentioned
      exact batchWitness_eq_after_coordinate field duplexBase u64Base
        candidateBase count initialBuilder initial coordinate
        (column := column)
        (by
          unfold Mentions at mentioned
          rcases List.mem_map.mp mentioned with
            ⟨⟨termColumn, coefficient⟩, termMember, rfl⟩
          have termLt :=
            (PiRlcCanonicalCandidatesHonest.inputsBelowBase duplexBase u64Base
              candidateBase initialBuilder coordinate
              (sourcesBelow coordinate) candidate).prior
              termColumn coefficient termMember
          simp only [PiRlcCanonicalCandidates.candidateLayout,
            PiRlcCanonicalCandidates.occurrenceBase,
            PiRlcCanonicalCandidates.occurrenceIndex, batchPrefixBoundary,
            PiRlcCanonicalCandidates.candidatesPerScalar, auxiliaryCount]
            at termLt ⊢
          have := candidate.isLt
          omega)
  · rw [batchWitness_eq_after_coordinate field duplexBase u64Base
      candidateBase count initialBuilder initial coordinate
      (column :=
        PiRlcCanonicalSelector.finalCountSource duplexBase u64Base
          candidateBase initialBuilder coordinate) (by
        simp only [PiRlcCanonicalSelector.finalCountSource,
          PiRlcCanonicalSelector.candidateSourceLayout,
          PiRlcCanonicalCandidate.cumulativeColumn,
          PiRlcCanonicalCandidates.candidateLayout,
          PiRlcCanonicalCandidates.occurrenceBase,
          PiRlcCanonicalCandidates.occurrenceIndex, batchPrefixBoundary,
          PiRlcCanonicalCandidates.candidatesPerScalar, auxiliaryCount]
        omega)]
    exact stage.finalCount

end Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidatesBatchHonest
