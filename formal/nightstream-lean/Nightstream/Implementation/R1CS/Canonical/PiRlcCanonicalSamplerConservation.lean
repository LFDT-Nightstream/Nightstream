import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerOwnership

/-!
Contract: complete column conservation for the Lean-owned canonical `Pi_RLC`
sampler suffix.

Under the canonical contiguous placement, every column mentioned by every
emitted suffix row is either:

* in the caller-owned prefix strictly below `u64Base`; or
* in the suffix's exact declared allocation.

The prefix contains the constant wire and the symbolic duplex inputs.  The
theorem deliberately requires equalities between adjacent family boundaries:
mere non-overlap would leave gaps that no family owns.

Assurance tier: model-level canonical encoding.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerConservation

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerHonest

private theorem u64Rows_mentions_lt_end
    (duplexBase u64Base count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (below :
      PiRlcCanonicalU64Honest.InputsBelow duplexBase u64Base count
        initialBuilder)
    (positive : 0 < u64Base)
    (row : Row)
    (member :
      row ∈ PiRlcCanonicalU64.rows duplexBase u64Base count initialBuilder)
    (column : Nat)
    (mentioned :
      Mentions row.a column ∨ Mentions row.b column ∨
        Mentions row.c column) :
    column <
      PiRlcCanonicalCandidatesBatchHonest.u64End u64Base count := by
  rcases List.mem_flatMap.mp member with ⟨coordinate, _, localMember⟩
  have localBound :=
    PiRlcCanonicalU64Honest.scalarRows_mentions_lt duplexBase u64Base count
      initialBuilder below coordinate positive row localMember column
      mentioned
  have coordinateLt := coordinate.isLt
  simp only [PiRlcCanonicalU64Honest.batchPrefixBoundary,
    PiRlcCanonicalCandidatesBatchHonest.u64End,
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
      row ∈
        PiRlcCanonicalCandidates.rows duplexBase u64Base candidateBase count
          initialBuilder)
    (column : Nat)
    (mentioned :
      Mentions row.a column ∨ Mentions row.b column ∨
        Mentions row.c column) :
    column <
      PiRlcCanonicalSelectorBatchHonest.candidateEnd candidateBase count := by
  rcases List.mem_flatMap.mp member with ⟨coordinate, _, localMember⟩
  have localBound :=
    PiRlcCanonicalCandidatesBatchHonest.scalarRows_mentions_lt duplexBase
      u64Base candidateBase count initialBuilder below coordinate positive row
      localMember column mentioned
  have coordinateLt := coordinate.isLt
  simp only [PiRlcCanonicalCandidatesBatchHonest.batchPrefixBoundary,
    PiRlcCanonicalSelectorBatchHonest.candidateEnd,
    PiRlcCanonicalCandidates.candidatesPerScalar,
    PiRlcCanonicalCandidate.auxiliaryCount] at localBound ⊢
  omega

private theorem selectorRows_mentions_lt_end
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (sourcesBefore :
      ∀ coordinate : Fin count,
        PiRlcCanonicalSelectorBatchHonest.SourcesBeforeSelector
          duplexBase u64Base candidateBase selectorBase initialBuilder
          coordinate)
    (positive : 0 < selectorBase)
    (row : Row)
    (member :
      row ∈
        PiRlcCanonicalSelector.rows duplexBase u64Base candidateBase
          selectorBase count initialBuilder)
    (column : Nat)
    (mentioned :
      Mentions row.a column ∨ Mentions row.b column ∨
        Mentions row.c column) :
    column <
      selectorBase + count * PiRlcCanonicalSelector.scalarAuxiliaryCount := by
  rcases List.mem_flatMap.mp member with ⟨coordinate, _, localMember⟩
  have localPositive :
      0 < PiRlcCanonicalSelector.scalarBase selectorBase coordinate := by
    unfold PiRlcCanonicalSelector.scalarBase
    omega
  have localBound :=
    PiRlcCanonicalSelectorBatchHonest.scalarRows_mentions_lt duplexBase
      u64Base candidateBase selectorBase initialBuilder coordinate
      (sourcesBefore coordinate).toSourcesBelow localPositive row localMember
      column mentioned
  have coordinateLt := coordinate.isLt
  simp only [PiRlcCanonicalSelectorBatchHonest.scalarEnd,
    PiRlcCanonicalSelector.scalarBase] at localBound
  have coordinateBound : coordinate.val + 1 ≤ count := by omega
  have scaled :=
    Nat.mul_le_mul_right PiRlcCanonicalSelector.scalarAuxiliaryCount
      coordinateBound
  simp only [Nat.add_mul, Nat.one_mul] at scaled
  omega

private theorem u64Allocation_mem_suffix
    (u64Base candidateBase selectorBase count column : Nat)
    (member : column ∈ PiRlcCanonicalU64.allocation u64Base count) :
    column ∈ suffixAllocation u64Base candidateBase selectorBase count := by
  unfold suffixAllocation
  exact List.mem_append.mpr
    (Or.inl (List.mem_append.mpr (Or.inl member)))

private theorem candidateAllocation_mem_suffix
    (u64Base candidateBase selectorBase count column : Nat)
    (member :
      column ∈ PiRlcCanonicalCandidates.allocation candidateBase count) :
    column ∈ suffixAllocation u64Base candidateBase selectorBase count := by
  unfold suffixAllocation
  exact List.mem_append.mpr
    (Or.inl (List.mem_append.mpr (Or.inr member)))

private theorem selectorAllocation_mem_suffix
    (u64Base candidateBase selectorBase count column : Nat)
    (member :
      column ∈ PiRlcCanonicalSelector.allocation selectorBase count) :
    column ∈ suffixAllocation u64Base candidateBase selectorBase count := by
  unfold suffixAllocation
  exact List.mem_append.mpr (Or.inr member)

/-- Complete suffix column conservation under the exact contiguous
family-major placement. -/
theorem suffixRows_conservation
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (positive : 0 < u64Base)
    (u64Below :
      PiRlcCanonicalU64Honest.InputsBelow duplexBase u64Base count
        initialBuilder)
    (u64Contiguous :
      PiRlcCanonicalCandidatesBatchHonest.u64End u64Base count =
        candidateBase)
    (candidateContiguous :
      PiRlcCanonicalSelectorBatchHonest.candidateEnd candidateBase count =
        selectorBase)
    (row : Row)
    (member :
      row ∈
        suffixRows duplexBase u64Base candidateBase selectorBase count
          initialBuilder)
    (column : Nat)
    (mentioned :
      Mentions row.a column ∨ Mentions row.b column ∨
        Mentions row.c column) :
    column < u64Base ∨
      column ∈ suffixAllocation u64Base candidateBase selectorBase count := by
  have u64Separated :
      PiRlcCanonicalCandidatesBatchHonest.u64End u64Base count ≤
        candidateBase := by
    omega
  have candidateSeparated :
      PiRlcCanonicalSelectorBatchHonest.candidateEnd candidateBase count ≤
        selectorBase := by
    omega
  have u64Boundary :
      candidateBase =
        u64Base + count * PiRlcCanonicalU64.lanesPerScalar *
          CanonicalU64Recipe.auxiliaryCount := by
    simpa [PiRlcCanonicalCandidatesBatchHonest.u64End] using
      u64Contiguous.symm
  have candidateBoundary :
      selectorBase =
        candidateBase +
          count * PiRlcCanonicalCandidates.candidatesPerScalar *
            PiRlcCanonicalCandidate.auxiliaryCount := by
    simpa [PiRlcCanonicalSelectorBatchHonest.candidateEnd] using
      candidateContiguous.symm
  have candidatePositive : 0 < candidateBase := by
    have baseToEnd :
        u64Base ≤
          PiRlcCanonicalCandidatesBatchHonest.u64End u64Base count := by
      unfold PiRlcCanonicalCandidatesBatchHonest.u64End
      exact Nat.le_add_right _ _
    exact Nat.lt_of_lt_of_le positive
      (Nat.le_trans baseToEnd u64Separated)
  have selectorPositive : 0 < selectorBase := by
    have baseToEnd :
        candidateBase ≤
          PiRlcCanonicalSelectorBatchHonest.candidateEnd candidateBase count := by
      unfold PiRlcCanonicalSelectorBatchHonest.candidateEnd
      exact Nat.le_add_right _ _
    exact Nat.lt_of_lt_of_le candidatePositive
      (Nat.le_trans baseToEnd candidateSeparated)
  let candidateBelow :=
    PiRlcCanonicalCandidatesBatchHonest.sourcesBelow_of_u64End duplexBase
      u64Base candidateBase count initialBuilder u64Separated
  let sourcesBefore :=
    PiRlcCanonicalSelectorBatchHonest.sourcesBeforeSelector_of_candidateEnd
      duplexBase u64Base candidateBase selectorBase count initialBuilder
      candidateSeparated
  simp only [suffixRows, List.mem_append] at member
  rcases member with (inU64 | inCandidates) | inSelectors
  · have upper :=
      u64Rows_mentions_lt_end duplexBase u64Base count initialBuilder u64Below
        positive row inU64 column mentioned
    by_cases external : column < u64Base
    · exact Or.inl external
    · right
      apply u64Allocation_mem_suffix
      rw [PiRlcCanonicalU64.allocation_mem_iff]
      exact ⟨Nat.le_of_not_gt external, by
        simpa [PiRlcCanonicalCandidatesBatchHonest.u64End] using upper⟩
  · have upper :=
      candidateRows_mentions_lt_end duplexBase u64Base candidateBase count
        initialBuilder candidateBelow candidatePositive row inCandidates column
        mentioned
    by_cases external : column < u64Base
    · exact Or.inl external
    · right
      by_cases beforeCandidate : column < candidateBase
      · apply u64Allocation_mem_suffix
        rw [PiRlcCanonicalU64.allocation_mem_iff]
        exact ⟨Nat.le_of_not_gt external, by omega⟩
      · apply candidateAllocation_mem_suffix
        rw [PiRlcCanonicalCandidates.allocation_mem_iff]
        exact ⟨Nat.le_of_not_gt beforeCandidate, upper⟩
  · have upper :=
      selectorRows_mentions_lt_end duplexBase u64Base candidateBase
        selectorBase count initialBuilder sourcesBefore selectorPositive row
        inSelectors column mentioned
    by_cases external : column < u64Base
    · exact Or.inl external
    · right
      by_cases beforeCandidate : column < candidateBase
      · apply u64Allocation_mem_suffix
        rw [PiRlcCanonicalU64.allocation_mem_iff]
        exact ⟨Nat.le_of_not_gt external, by omega⟩
      · by_cases beforeSelector : column < selectorBase
        · apply candidateAllocation_mem_suffix
          rw [PiRlcCanonicalCandidates.allocation_mem_iff]
          exact ⟨Nat.le_of_not_gt beforeCandidate, by omega⟩
        · apply selectorAllocation_mem_suffix
          rw [PiRlcCanonicalSelector.allocation_mem_iff]
          exact ⟨Nat.le_of_not_gt beforeSelector, upper⟩

end Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerConservation
