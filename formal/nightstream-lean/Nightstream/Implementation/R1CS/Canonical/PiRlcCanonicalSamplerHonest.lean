import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelectorBatchHonest

/-!
Contract: honest witness composition for the Lean-owned canonical `Pi_RLC`
sampler suffix.

The emitted order is unchanged:

1. canonical-u64 decomposition for every coordinate;
2. all candidate classifiers;
3. all first-accepted selectors.

The sole semantic side condition is `FirstAccepted.Enough` for each exact
candidate list.  Transcript rows are not silently assumed honest: the main
theorem below constructs and satisfies only this downstream suffix.  A
separate composition theorem may transport already-satisfied transcript rows
once their operand placement is explicit.

Assurance tier: model-level canonical encoding.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerHonest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidateHonest
open Nightstream.SuperNeo.Sampling
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

/-- Exact downstream sampler row list, preserving the existing family-major
emission order. -/
def suffixRows
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder) : List Row :=
  PiRlcCanonicalU64.rows duplexBase u64Base count initialBuilder ++
    PiRlcCanonicalCandidates.rows duplexBase u64Base candidateBase count initialBuilder ++
    PiRlcCanonicalSelector.rows duplexBase u64Base candidateBase selectorBase count
      initialBuilder

/-- Exact allocation list owned by the downstream suffix. -/
def suffixAllocation
    (u64Base candidateBase selectorBase count : Nat) : List Nat :=
  PiRlcCanonicalU64.allocation u64Base count ++
    PiRlcCanonicalCandidates.allocation candidateBase count ++
    PiRlcCanonicalSelector.allocation selectorBase count

/-- Exact cost of the emitted suffix.  Both components are later tied back to
the concrete row and allocation lists. -/
def suffixCost (count : Nat) :
    Nightstream.Implementation.Lowering.Typed.Cost where
  recurringRows :=
    count * PiRlcCanonicalU64.lanesPerScalar *
        CanonicalU64Recipe.cost.recurringRows +
      count * PiRlcCanonicalCandidates.candidatesPerScalar *
        PiRlcCanonicalCandidate.cost.recurringRows +
      (PiRlcCanonicalSelector.cost count).recurringRows
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns :=
    count * PiRlcCanonicalU64.lanesPerScalar *
        CanonicalU64Recipe.cost.auxiliaryColumns +
      count * PiRlcCanonicalCandidates.candidatesPerScalar *
        PiRlcCanonicalCandidate.cost.auxiliaryColumns +
      (PiRlcCanonicalSelector.cost count).auxiliaryColumns

/-- Row cost is derived from the three emitted family lists. -/
theorem suffixRows_length
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder) :
    (suffixRows duplexBase u64Base candidateBase selectorBase count
      initialBuilder).length =
      (suffixCost count).recurringRows := by
  simp only [suffixRows, List.length_append,
    PiRlcCanonicalU64.rows_length,
    PiRlcCanonicalCandidates.rows_length,
    PiRlcCanonicalSelector.rows_length, suffixCost,
    PiRlcCanonicalSelector.cost,
    PiRlcCanonicalU64.lanesPerScalar,
    CanonicalU64Recipe.cost,
    PiRlcCanonicalCandidate.cost,
    PiRlcCanonicalCandidates.candidatesPerScalar]

/-- Column cost is derived from the exact concatenated allocation. -/
theorem suffixAllocation_length
    (u64Base candidateBase selectorBase count : Nat) :
    (suffixAllocation u64Base candidateBase selectorBase count).length =
      (suffixCost count).auxiliaryColumns := by
  simp only [suffixAllocation, List.length_append,
    PiRlcCanonicalU64.allocation_length,
    PiRlcCanonicalCandidates.allocation_length,
    PiRlcCanonicalSelector.allocation_length, suffixCost,
    PiRlcCanonicalSelector.cost,
    PiRlcCanonicalU64.lanesPerScalar,
    CanonicalU64Recipe.cost,
    PiRlcCanonicalCandidate.cost,
    PiRlcCanonicalCandidates.candidatesPerScalar,
    PiRlcCanonicalSelector.scalarAuxiliaryCount,
    PiRlcCanonicalSelector.outputCount,
    PiRlcCanonicalSelector.positionAuxiliaryCount]

theorem fixedActive_suffixRows_length
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder) :
    (suffixRows duplexBase u64Base candidateBase selectorBase 15
      initialBuilder).length = 96090 := by
  rw [suffixRows_length]
  rfl

theorem fixedActive_suffixAllocation_length
    (u64Base candidateBase selectorBase : Nat) :
    (suffixAllocation u64Base candidateBase selectorBase 15).length =
      89325 := by
  rw [suffixAllocation_length]
  rfl

theorem fixedActive_suffixCost_recurringRows :
    (suffixCost 15).recurringRows = 96090 := by
  rfl

theorem fixedActive_suffixCost_auxiliaryColumns :
    (suffixCost 15).auxiliaryColumns = 89325 := by
  rfl

/-- Exact placement separation makes the three contiguous allocations
pairwise disjoint. -/
theorem suffixAllocation_nodup
    (u64Base candidateBase selectorBase count : Nat)
    (u64Separated :
      PiRlcCanonicalCandidatesBatchHonest.u64End u64Base count ≤
        candidateBase)
    (candidateSeparated :
      PiRlcCanonicalSelectorBatchHonest.candidateEnd candidateBase count ≤
        selectorBase) :
    (suffixAllocation u64Base candidateBase selectorBase count).Nodup := by
  unfold suffixAllocation
  rw [List.nodup_append, List.nodup_append]
  refine
    ⟨⟨PiRlcCanonicalU64.allocation_nodup u64Base count,
      PiRlcCanonicalCandidates.allocation_nodup candidateBase count,
      ?_⟩,
      PiRlcCanonicalSelector.allocation_nodup selectorBase count,
      ?_⟩
  · intro left leftMember right rightMember equal
    subst right
    have leftWindow :=
      (PiRlcCanonicalU64.allocation_mem_iff u64Base count left).mp leftMember
    have rightWindow :=
      (PiRlcCanonicalCandidates.allocation_mem_iff candidateBase count
        left).mp rightMember
    exact (Nat.not_lt_of_ge rightWindow.1)
      (Nat.lt_of_lt_of_le leftWindow.2 u64Separated)
  · intro left leftMember right rightMember equal
    subst right
    simp only [List.mem_append] at leftMember
    have rightWindow :=
      (PiRlcCanonicalSelector.allocation_mem_iff selectorBase count left).mp
        rightMember
    rcases leftMember with inU64 | inCandidates
    · have leftWindow :=
        (PiRlcCanonicalU64.allocation_mem_iff u64Base count left).mp inU64
      have u64ToSelector :
          PiRlcCanonicalCandidatesBatchHonest.u64End u64Base count ≤
            selectorBase := by
        exact Nat.le_trans u64Separated
          (Nat.le_trans
            (Nat.le_add_right candidateBase _)
            candidateSeparated)
      exact (Nat.not_lt_of_ge rightWindow.1)
        (Nat.lt_of_lt_of_le leftWindow.2 u64ToSelector)
    · have leftWindow :=
        (PiRlcCanonicalCandidates.allocation_mem_iff candidateBase count
          left).mp inCandidates
      exact (Nat.not_lt_of_ge rightWindow.1)
        (Nat.lt_of_lt_of_le leftWindow.2 candidateSeparated)

/-- Canonical-u64 witness over the authoritative symbolic digest lanes. -/
def u64Witness
    (field : FieldInverse)
    (duplexBase u64Base count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat) : Nat → Nat :=
  PiRlcCanonicalU64Honest.batchWitness field duplexBase u64Base count initialBuilder initial

/-- Candidate witness over the exact canonical-u64 bit assignment. -/
def candidateWitness
    (field : FieldInverse)
    (duplexBase u64Base candidateBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat) : Nat → Nat :=
  PiRlcCanonicalCandidatesBatchHonest.batchWitness field duplexBase u64Base candidateBase count
    initialBuilder (u64Witness field duplexBase u64Base count initialBuilder
      initial)

/-- The exact verifier-owned candidate list decoded from one coordinate's
canonical-u64 bits. -/
def honestCandidates
    (field : FieldInverse)
    (duplexBase u64Base candidateBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    (u64Below : PiRlcCanonicalU64Honest.InputsBelow duplexBase u64Base count initialBuilder)
    (candidateBelow :
      ∀ coordinate : Fin count,
        PiRlcCanonicalCandidatesHonest.SourcesBelow
          duplexBase u64Base candidateBase initialBuilder coordinate)
    (coordinate : Fin count) :
    List ProductionAlphabet.Chunk :=
  PiRlcCanonicalCandidatesBatchHonest.coordinateCandidates field duplexBase u64Base candidateBase
    count initialBuilder
    (u64Witness field duplexBase u64Base count initialBuilder initial)
    candidateBelow
    (fun target candidate =>
      PiRlcCanonicalCandidatesBatchHonest.sourceBitsBoolean_of_u64Witness field duplexBase u64Base
        candidateBase count initialBuilder initial u64Below target candidate)
    coordinate

/-- The completed candidate batch binds every selector source to the exact
decoded candidate list. -/
theorem candidateSourcesMatch
    (field : FieldInverse)
    (duplexBase u64Base candidateBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    (u64Below : PiRlcCanonicalU64Honest.InputsBelow duplexBase u64Base count initialBuilder)
    (candidateBelow :
      ∀ coordinate : Fin count,
        PiRlcCanonicalCandidatesHonest.SourcesBelow
          duplexBase u64Base candidateBase initialBuilder coordinate)
    (coordinate : Fin count) :
    PiRlcCanonicalSelectorHonest.SourcesMatch
      duplexBase u64Base candidateBase initialBuilder coordinate
      (candidateWitness field duplexBase u64Base candidateBase count
        initialBuilder initial)
      (honestCandidates field duplexBase u64Base candidateBase count
        initialBuilder initial u64Below candidateBelow coordinate) := by
  exact PiRlcCanonicalCandidatesBatchHonest.coordinateSourcesMatch field duplexBase u64Base
    candidateBase count initialBuilder
    (u64Witness field duplexBase u64Base count initialBuilder initial)
    candidateBelow
    (fun target candidate =>
      PiRlcCanonicalCandidatesBatchHonest.sourceBitsBoolean_of_u64Witness field duplexBase u64Base
        candidateBase count initialBuilder initial u64Below target candidate)
    coordinate

/-- Final selector witness over the internally bound candidate values. -/
def finalWitness
    (field : FieldInverse)
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    (u64Below : PiRlcCanonicalU64Honest.InputsBelow duplexBase u64Base count initialBuilder)
    (candidateBelow :
      ∀ coordinate : Fin count,
        PiRlcCanonicalCandidatesHonest.SourcesBelow
          duplexBase u64Base candidateBase initialBuilder coordinate)
    (enough :
      ∀ coordinate : Fin count,
        FirstAccepted.Enough ProductionAlphabet.verifier PiRlcCanonicalSelector.outputCount
          (honestCandidates field duplexBase u64Base candidateBase count
            initialBuilder initial u64Below candidateBelow coordinate)) :
    Nat → Nat :=
  PiRlcCanonicalSelectorBatchHonest.batchWitness duplexBase u64Base candidateBase selectorBase
    count initialBuilder
    (candidateWitness field duplexBase u64Base candidateBase count
      initialBuilder initial)
    (honestCandidates field duplexBase u64Base candidateBase count
      initialBuilder initial u64Below candidateBelow)
    (candidateSourcesMatch field duplexBase u64Base candidateBase count
      initialBuilder initial u64Below candidateBelow)
    enough

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

private theorem u64Rows_mentions_lt_end
    (duplexBase u64Base count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (below : PiRlcCanonicalU64Honest.InputsBelow duplexBase u64Base count initialBuilder)
    (positive : 0 < u64Base)
    (row : Row)
    (member : row ∈ PiRlcCanonicalU64.rows duplexBase u64Base count initialBuilder)
    (column : Nat)
    (mentioned :
      Mentions row.a column ∨ Mentions row.b column ∨
        Mentions row.c column) :
    column < PiRlcCanonicalCandidatesBatchHonest.u64End u64Base count := by
  rcases List.mem_flatMap.mp member with ⟨coordinate, _, localMember⟩
  have localBound :=
    PiRlcCanonicalU64Honest.scalarRows_mentions_lt duplexBase u64Base count initialBuilder
      below coordinate positive row localMember column mentioned
  have coordinateLt := coordinate.isLt
  simp only [PiRlcCanonicalU64Honest.batchPrefixBoundary, PiRlcCanonicalCandidatesBatchHonest.u64End,
    PiRlcCanonicalU64.lanesPerScalar,
    CanonicalU64Recipe.auxiliaryCount] at localBound ⊢
  omega

private theorem candidateRows_mentions_lt_end
    (duplexBase u64Base candidateBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (below :
      ∀ coordinate : Fin count,
        PiRlcCanonicalCandidatesHonest.SourcesBelow
          duplexBase u64Base candidateBase initialBuilder coordinate)
    (positive : 0 < candidateBase)
    (row : Row)
    (member :
      row ∈ PiRlcCanonicalCandidates.rows duplexBase u64Base candidateBase count
        initialBuilder)
    (column : Nat)
    (mentioned :
      Mentions row.a column ∨ Mentions row.b column ∨
        Mentions row.c column) :
    column < PiRlcCanonicalSelectorBatchHonest.candidateEnd candidateBase count := by
  rcases List.mem_flatMap.mp member with ⟨coordinate, _, localMember⟩
  have localBound :=
    PiRlcCanonicalCandidatesBatchHonest.scalarRows_mentions_lt duplexBase u64Base candidateBase
      count initialBuilder below coordinate positive row localMember column
      mentioned
  have coordinateLt := coordinate.isLt
  simp only [PiRlcCanonicalCandidatesBatchHonest.batchPrefixBoundary,
    PiRlcCanonicalSelectorBatchHonest.candidateEnd, PiRlcCanonicalCandidates.candidatesPerScalar,
    PiRlcCanonicalCandidate.auxiliaryCount] at localBound ⊢
  omega

private theorem finalWitness_before_selectorBase
    (field : FieldInverse)
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    (u64Below : PiRlcCanonicalU64Honest.InputsBelow duplexBase u64Base count initialBuilder)
    (candidateBelow :
      ∀ coordinate : Fin count,
        PiRlcCanonicalCandidatesHonest.SourcesBelow
          duplexBase u64Base candidateBase initialBuilder coordinate)
    (enough :
      ∀ coordinate : Fin count,
        FirstAccepted.Enough ProductionAlphabet.verifier PiRlcCanonicalSelector.outputCount
          (honestCandidates field duplexBase u64Base candidateBase count
            initialBuilder initial u64Below candidateBelow coordinate))
    {column : Nat} (before : column < selectorBase) :
    finalWitness field duplexBase u64Base candidateBase selectorBase count
        initialBuilder initial u64Below candidateBelow enough column =
      candidateWitness field duplexBase u64Base candidateBase count
        initialBuilder initial column := by
  unfold finalWitness PiRlcCanonicalSelectorBatchHonest.batchWitness
  exact PiRlcCanonicalSelectorBatchHonest.batchPrefixWitness_before_selectorBase duplexBase
    u64Base candidateBase selectorBase count initialBuilder
    (candidateWitness field duplexBase u64Base candidateBase count
      initialBuilder initial)
    (honestCandidates field duplexBase u64Base candidateBase count
      initialBuilder initial u64Below candidateBelow)
    (candidateSourcesMatch field duplexBase u64Base candidateBase count
      initialBuilder initial u64Below candidateBelow)
    enough count before

private theorem candidateWitness_before_candidateBase
    (field : FieldInverse)
    (duplexBase u64Base candidateBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    {column : Nat} (before : column < candidateBase) :
    candidateWitness field duplexBase u64Base candidateBase count
        initialBuilder initial column =
      u64Witness field duplexBase u64Base count initialBuilder initial
        column := by
  unfold candidateWitness PiRlcCanonicalCandidatesBatchHonest.batchWitness
  exact PiRlcCanonicalCandidatesBatchHonest.batchPrefixWitness_before_candidateBase field
    duplexBase u64Base candidateBase count initialBuilder
    (u64Witness field duplexBase u64Base count initialBuilder initial)
    count before

/-- The complete suffix witness preserves every caller-owned column before the
u64 allocation.  This is the composition boundary used to retain already
satisfied transcript rows. -/
theorem finalWitness_before_u64Base
    (field : FieldInverse)
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    (u64Below :
      PiRlcCanonicalU64Honest.InputsBelow
        duplexBase u64Base count initialBuilder)
    (candidateBelow :
      ∀ coordinate : Fin count,
        PiRlcCanonicalCandidatesHonest.SourcesBelow
          duplexBase u64Base candidateBase initialBuilder coordinate)
    (enough :
      ∀ coordinate : Fin count,
        FirstAccepted.Enough ProductionAlphabet.verifier
          PiRlcCanonicalSelector.outputCount
          (honestCandidates field duplexBase u64Base candidateBase count
            initialBuilder initial u64Below candidateBelow coordinate))
    (u64Separated :
      PiRlcCanonicalCandidatesBatchHonest.u64End u64Base count ≤
        candidateBase)
    (candidateSeparated :
      PiRlcCanonicalSelectorBatchHonest.candidateEnd candidateBase count ≤
        selectorBase)
    {column : Nat} (before : column < u64Base) :
    finalWitness field duplexBase u64Base candidateBase selectorBase count
        initialBuilder initial u64Below candidateBelow enough column =
      initial column := by
  have beforeCandidate : column < candidateBase := by
    exact Nat.lt_of_lt_of_le before
      (Nat.le_trans
        (Nat.le_add_right u64Base _)
        u64Separated)
  have beforeSelector : column < selectorBase := by
    exact Nat.lt_of_lt_of_le beforeCandidate
      (Nat.le_trans
        (Nat.le_add_right candidateBase _)
        candidateSeparated)
  rw [finalWitness_before_selectorBase field duplexBase u64Base candidateBase
    selectorBase count initialBuilder initial u64Below candidateBelow enough
    beforeSelector]
  rw [candidateWitness_before_candidateBase field duplexBase u64Base
    candidateBase count initialBuilder initial beforeCandidate]
  exact PiRlcCanonicalU64Honest.batchWitness_before_u64Base field
    duplexBase u64Base count initialBuilder initial before

/-- Honest completeness of the exact downstream sampler suffix. -/
theorem suffixRows_complete
    (field : FieldInverse)
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    (initialCanonical : ∀ column, initial column < goldilocksP)
    (constantWire : initial 0 = 1)
    (positive : 0 < u64Base)
    (u64Below : PiRlcCanonicalU64Honest.InputsBelow duplexBase u64Base count initialBuilder)
    (u64Separated : PiRlcCanonicalCandidatesBatchHonest.u64End u64Base count ≤ candidateBase)
    (candidateSeparated :
      PiRlcCanonicalSelectorBatchHonest.candidateEnd candidateBase count ≤ selectorBase)
    (enough :
      ∀ coordinate : Fin count,
        FirstAccepted.Enough ProductionAlphabet.verifier PiRlcCanonicalSelector.outputCount
          (honestCandidates field duplexBase u64Base candidateBase count
            initialBuilder initial u64Below
            (PiRlcCanonicalCandidatesBatchHonest.sourcesBelow_of_u64End duplexBase u64Base
              candidateBase count initialBuilder u64Separated)
            coordinate)) :
    Satisfies
      (suffixRows duplexBase u64Base candidateBase selectorBase count
        initialBuilder)
      (finalWitness field duplexBase u64Base candidateBase selectorBase count
        initialBuilder initial u64Below
        (PiRlcCanonicalCandidatesBatchHonest.sourcesBelow_of_u64End duplexBase u64Base
          candidateBase count initialBuilder u64Separated)
        enough) := by
  let candidateBelow :=
    PiRlcCanonicalCandidatesBatchHonest.sourcesBelow_of_u64End duplexBase u64Base candidateBase
      count initialBuilder u64Separated
  let sourceBits :=
    fun (coordinate : Fin count)
      (candidate : Fin PiRlcCanonicalCandidates.candidatesPerScalar) =>
      PiRlcCanonicalCandidatesBatchHonest.sourceBitsBoolean_of_u64Witness field duplexBase u64Base
        candidateBase count initialBuilder initial u64Below coordinate candidate
  let u64Assignment :=
    u64Witness field duplexBase u64Base count initialBuilder initial
  let candidateAssignment :=
    candidateWitness field duplexBase u64Base candidateBase count
      initialBuilder initial
  let candidates :=
    honestCandidates field duplexBase u64Base candidateBase count initialBuilder
      initial u64Below candidateBelow
  let sources :=
    candidateSourcesMatch field duplexBase u64Base candidateBase count
      initialBuilder initial u64Below candidateBelow
  let selectorAssignment :=
    finalWitness field duplexBase u64Base candidateBase selectorBase count
      initialBuilder initial u64Below candidateBelow enough
  have candidatePositive : 0 < candidateBase := by
    have baseToEnd : u64Base ≤ PiRlcCanonicalCandidatesBatchHonest.u64End u64Base count := by
      unfold PiRlcCanonicalCandidatesBatchHonest.u64End
      exact Nat.le_add_right _ _
    exact Nat.lt_of_lt_of_le positive
      (Nat.le_trans baseToEnd u64Separated)
  have selectorPositive : 0 < selectorBase := by
    have baseToEnd :
        candidateBase ≤ PiRlcCanonicalSelectorBatchHonest.candidateEnd candidateBase count := by
      unfold PiRlcCanonicalSelectorBatchHonest.candidateEnd
      exact Nat.le_add_right _ _
    exact Nat.lt_of_lt_of_le candidatePositive
      (Nat.le_trans baseToEnd candidateSeparated)
  have u64Constant : u64Assignment 0 = 1 := by
    change
      PiRlcCanonicalU64Honest.batchWitness field duplexBase u64Base count initialBuilder
        initial 0 = 1
    rw [PiRlcCanonicalU64Honest.batchWitness_before_u64Base field duplexBase u64Base count
      initialBuilder initial positive]
    exact constantWire
  have u64Canonical : ∀ column, u64Assignment column < goldilocksP :=
    PiRlcCanonicalU64Honest.batchWitness_canonical field duplexBase u64Base count
      initialBuilder initial initialCanonical
  have candidateConstant : candidateAssignment 0 = 1 := by
    change
      candidateWitness field duplexBase u64Base candidateBase count
        initialBuilder initial 0 = 1
    rw [candidateWitness_before_candidateBase field duplexBase u64Base
      candidateBase count initialBuilder initial candidatePositive]
    exact u64Constant
  have candidateCanonical :
      ∀ column, candidateAssignment column < goldilocksP := by
    exact PiRlcCanonicalCandidatesBatchHonest.batchWitness_canonical field duplexBase u64Base
      candidateBase count initialBuilder u64Assignment u64Canonical
      candidateBelow sourceBits
  have u64Satisfied :
      Satisfies (PiRlcCanonicalU64.rows duplexBase u64Base count initialBuilder)
        u64Assignment :=
    PiRlcCanonicalU64Honest.rows_complete field duplexBase u64Base count initialBuilder
      initial positive constantWire u64Below
  have candidateSatisfied :
      Satisfies
        (PiRlcCanonicalCandidates.rows duplexBase u64Base candidateBase count initialBuilder)
        candidateAssignment :=
    PiRlcCanonicalCandidatesBatchHonest.rows_complete field duplexBase u64Base candidateBase count
      initialBuilder u64Assignment candidatePositive u64Constant candidateBelow
      sourceBits
  have selectorSatisfied :
      Satisfies
        (PiRlcCanonicalSelector.rows duplexBase u64Base candidateBase selectorBase count
          initialBuilder)
        selectorAssignment := by
    apply PiRlcCanonicalSelectorBatchHonest.rows_complete duplexBase u64Base candidateBase
      selectorBase count initialBuilder candidateAssignment candidateCanonical
      candidateConstant selectorPositive candidates sources enough
    exact PiRlcCanonicalSelectorBatchHonest.sourcesBeforeSelector_of_candidateEnd duplexBase
      u64Base candidateBase selectorBase count initialBuilder
      candidateSeparated
  intro row member
  simp only [suffixRows, List.mem_append] at member
  rcases member with (inU64 | inCandidates) | inSelectors
  · have holds := u64Satisfied row inU64
    apply
      (rowHolds_congr u64Assignment selectorAssignment row ?_).mp holds
    intro column mentioned
    have columnLt :=
      u64Rows_mentions_lt_end duplexBase u64Base count initialBuilder u64Below
        positive row inU64 column mentioned
    have beforeCandidate : column < candidateBase :=
      Nat.lt_of_lt_of_le columnLt u64Separated
    have beforeSelector : column < selectorBase :=
      Nat.lt_of_lt_of_le
        (Nat.lt_of_lt_of_le beforeCandidate
          (Nat.le_add_right candidateBase _))
        candidateSeparated
    change
      u64Witness field duplexBase u64Base count initialBuilder initial column =
        finalWitness field duplexBase u64Base candidateBase selectorBase count
          initialBuilder initial u64Below candidateBelow enough column
    rw [finalWitness_before_selectorBase field duplexBase u64Base candidateBase
      selectorBase count initialBuilder initial u64Below candidateBelow enough
      beforeSelector]
    symm
    exact candidateWitness_before_candidateBase field duplexBase u64Base
      candidateBase count initialBuilder initial beforeCandidate
  · have holds := candidateSatisfied row inCandidates
    apply
      (rowHolds_congr candidateAssignment selectorAssignment row ?_).mp holds
    intro column mentioned
    have columnLt :=
      candidateRows_mentions_lt_end duplexBase u64Base candidateBase count
        initialBuilder candidateBelow candidatePositive row inCandidates column
        mentioned
    change
      candidateWitness field duplexBase u64Base candidateBase count
          initialBuilder initial column =
        finalWitness field duplexBase u64Base candidateBase selectorBase count
          initialBuilder initial u64Below candidateBelow enough column
    exact (finalWitness_before_selectorBase field duplexBase u64Base
      candidateBase selectorBase count initialBuilder initial u64Below
      candidateBelow enough
      (Nat.lt_of_lt_of_le columnLt candidateSeparated)).symm
  · exact selectorSatisfied row inSelectors

end Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerHonest
