import Nightstream.Implementation.R1CS.Canonical.AllocationCoverage
import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerProgram
import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSymbolicMachineCallOrder

/-!
Contract: exact allocation coverage for the fixed-active `Pi_RLC` sampler.

Every column counted by the transcript, canonical-u64, candidate, or selector
allocation must occur in an emitted row.  This is the converse of conservation:
it rejects a correct-looking auxiliary count padded with unconstrained columns.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerCoverage

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.AllocationCoverage
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule

/-! ## One canonical-u64 occurrence -/

theorem canonicalU64
    (layout : CanonicalU64Recipe.Layout) :
    RowsCover
      (CanonicalU64Recipe.rows layout)
      (CanonicalU64Recipe.allocation layout) := by
  intro column member
  unfold CanonicalU64Recipe.allocation at member
  rcases List.mem_append.1 member with inBits | inTail
  · rcases List.mem_map.1 inBits with ⟨index, indexMember, rfl⟩
    let row := bitRow (CanonicalU64Recipe.bitColumn layout index)
    refine ⟨row, ?_, Or.inl ?_⟩
    · unfold row CanonicalU64Recipe.rows CanonicalU64Recipe.bitRows
      exact List.mem_append_left _
        (List.mem_map.2 ⟨index, indexMember, rfl⟩)
    · simp [row, bitRow, Mentions]
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at inTail
    rcases inTail with rfl | rfl
    · let row := bitRow (CanonicalU64Recipe.highFlagColumn layout)
      refine ⟨row, ?_, Or.inl ?_⟩
      · simp [row, CanonicalU64Recipe.rows]
      · simp [row, bitRow, Mentions]
    · let row := CanonicalU64Recipe.inverseRow layout
      refine ⟨row, ?_, Or.inr (Or.inl ?_)⟩
      · simp [row, CanonicalU64Recipe.rows]
      · simp [row, CanonicalU64Recipe.inverseRow, Mentions]

/-! ## One rejection candidate -/

theorem candidate
    (layout : PiRlcCanonicalCandidate.Layout) :
    RowsCover
      (PiRlcCanonicalCandidate.rows layout)
      (PiRlcCanonicalCandidate.allocation layout) := by
  intro column member
  unfold PiRlcCanonicalCandidate.allocation at member
  rcases List.mem_map.1 member with ⟨offset, offsetMember, rfl⟩
  have offsetLt : offset < PiRlcCanonicalCandidate.auxiliaryCount :=
    List.mem_range.1 offsetMember
  have cases :
      offset = 0 ∨ offset = 1 ∨ offset = 2 ∨ offset = 3 ∨
      offset = 4 ∨ offset = 5 ∨ offset = 6 ∨
      (7 ≤ offset ∧ offset < 21) ∨ offset = 21 := by
    simp only [PiRlcCanonicalCandidate.auxiliaryCount] at offsetLt
    omega
  rcases cases with
      rfl | rfl | rfl | rfl | rfl | rfl | rfl | middle | rfl
  · let row := bitRow (PiRlcCanonicalCandidate.acceptColumn layout)
    refine ⟨row, ?_, Or.inl ?_⟩
    · simp [row, PiRlcCanonicalCandidate.rows,
        PiRlcCanonicalCandidate.acceptanceRows]
    · simp [row, bitRow, PiRlcCanonicalCandidate.acceptColumn, Mentions]
  · let row :=
      (⟨PiRlcCanonicalCandidate.differenceTerms layout,
        [(PiRlcCanonicalCandidate.inverseColumn layout, 1)],
        [(PiRlcCanonicalCandidate.acceptColumn layout, 1)]⟩ : Row)
    refine ⟨row, ?_, Or.inr (Or.inl ?_)⟩
    · simp [row, PiRlcCanonicalCandidate.rows,
        PiRlcCanonicalCandidate.acceptanceRows]
    · simp [row, PiRlcCanonicalCandidate.acceptanceRows,
        PiRlcCanonicalCandidate.inverseColumn, Mentions]
  · let row :=
      (⟨[(PiRlcCanonicalCandidate.residueColumn layout, 1)],
        [(PiRlcCanonicalCandidate.residueColumn layout, 1),
          (0, goldilocksP - 1)],
        [(PiRlcCanonicalCandidate.productColumn layout 0, 1)]⟩ : Row)
    refine ⟨row, ?_, Or.inl ?_⟩
    · simp [row, PiRlcCanonicalCandidate.rows,
        PiRlcCanonicalCandidate.residueRangeRows]
    · simp [row, PiRlcCanonicalCandidate.residueRangeRows,
        PiRlcCanonicalCandidate.residueColumn, Mentions]
  · let row := PiRlcCanonicalCandidate.quotientRecompositionRow layout
    refine ⟨row, ?_, Or.inl ?_⟩
    · simp [row, PiRlcCanonicalCandidate.rows]
    · simp [row, PiRlcCanonicalCandidate.quotientRecompositionRow,
        PiRlcCanonicalCandidate.quotientColumn, Mentions]
  · let row :=
      (⟨[(PiRlcCanonicalCandidate.residueColumn layout, 1)],
        [(PiRlcCanonicalCandidate.residueColumn layout, 1),
          (0, goldilocksP - 1)],
        [(PiRlcCanonicalCandidate.productColumn layout 0, 1)]⟩ : Row)
    refine ⟨row, ?_, Or.inr (Or.inr ?_)⟩
    · simp [row, PiRlcCanonicalCandidate.rows,
        PiRlcCanonicalCandidate.residueRangeRows]
    · simp [row, PiRlcCanonicalCandidate.residueRangeRows,
        PiRlcCanonicalCandidate.productColumn, Mentions]
  · let row :=
      (⟨[(PiRlcCanonicalCandidate.productColumn layout 0, 1)],
        [(PiRlcCanonicalCandidate.residueColumn layout, 1),
          (0, goldilocksP - 2)],
        [(PiRlcCanonicalCandidate.productColumn layout 1, 1)]⟩ : Row)
    refine ⟨row, ?_, Or.inr (Or.inr ?_)⟩
    · simp [row, PiRlcCanonicalCandidate.rows,
        PiRlcCanonicalCandidate.residueRangeRows]
    · simp [row, PiRlcCanonicalCandidate.residueRangeRows,
        PiRlcCanonicalCandidate.productColumn, Mentions]
  · let row :=
      (⟨[(PiRlcCanonicalCandidate.productColumn layout 1, 1)],
        [(PiRlcCanonicalCandidate.residueColumn layout, 1),
          (0, goldilocksP - 3)],
        [(PiRlcCanonicalCandidate.productColumn layout 2, 1)]⟩ : Row)
    refine ⟨row, ?_, Or.inr (Or.inr ?_)⟩
    · simp [row, PiRlcCanonicalCandidate.rows,
        PiRlcCanonicalCandidate.residueRangeRows]
    · simp [row, PiRlcCanonicalCandidate.residueRangeRows,
        PiRlcCanonicalCandidate.productColumn, Mentions]
  · rcases middle with ⟨lower, upper⟩
    let bitOffset := offset - 7
    have bitOffsetLt :
        bitOffset < PiRlcCanonicalCandidate.quotientBitCount := by
      simp only [bitOffset, PiRlcCanonicalCandidate.quotientBitCount]
      omega
    let row :=
      bitRow (PiRlcCanonicalCandidate.quotientBitColumn layout bitOffset)
    refine ⟨row, ?_, Or.inl ?_⟩
    · unfold row PiRlcCanonicalCandidate.rows
        PiRlcCanonicalCandidate.quotientBitRows
      apply List.mem_append_left
      apply List.mem_append_right
      exact List.mem_map.2
        ⟨bitOffset, List.mem_range.2 bitOffsetLt, rfl⟩
    · simp [row, bitRow, bitOffset,
        PiRlcCanonicalCandidate.quotientBitColumn, Mentions]
      omega
  · let row := PiRlcCanonicalCandidate.cumulativeRow layout
    refine ⟨row, ?_, Or.inl ?_⟩
    · simp [row, PiRlcCanonicalCandidate.rows]
    · simp [row, PiRlcCanonicalCandidate.cumulativeRow,
        PiRlcCanonicalCandidate.cumulativeColumn, Mentions]

/-! ## One selector scalar -/

theorem selectorScalar
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (offset : Nat)
    (offsetLt : offset < PiRlcCanonicalSelector.scalarAuxiliaryCount) :
    ∃ row ∈
        PiRlcCanonicalSelector.scalarRows duplexBase u64Base candidateBase
          selectorBase initial coordinate,
      Mentions row.a
          (PiRlcCanonicalSelector.scalarBase selectorBase coordinate + offset) ∨
        Mentions row.b
          (PiRlcCanonicalSelector.scalarBase selectorBase coordinate + offset) ∨
        Mentions row.c
          (PiRlcCanonicalSelector.scalarBase selectorBase coordinate + offset) := by
  by_cases inHeader : offset < 5
  · have cases :
        offset = 0 ∨ offset = 1 ∨ offset = 2 ∨ offset = 3 ∨ offset = 4 := by
      omega
    rcases cases with rfl | rfl | rfl | rfl | rfl
    · let row : Row :=
        ⟨[(PiRlcCanonicalSelector.slackColumn selectorBase coordinate, 1)],
          [(0, 1)],
          PiRlcCanonicalSelector.slackTerms selectorBase coordinate⟩
      refine ⟨row, ?_, Or.inl ?_⟩
      · simp [row, PiRlcCanonicalSelector.scalarRows,
          PiRlcCanonicalSelector.acceptanceBoundRows]
      · simp [row, PiRlcCanonicalSelector.slackColumn,
          PiRlcCanonicalSelector.scalarBase, Mentions]
    · let row :=
        bitRow
          (PiRlcCanonicalSelector.slackBitColumn selectorBase coordinate 0)
      refine ⟨row, ?_, Or.inl ?_⟩
      · unfold row PiRlcCanonicalSelector.scalarRows
          PiRlcCanonicalSelector.acceptanceBoundRows
        apply List.mem_append_left
        apply List.mem_append_left
        exact List.mem_map.2 ⟨0, List.mem_range.2 (by decide), rfl⟩
      · simp [row, bitRow, PiRlcCanonicalSelector.slackBitColumn,
          PiRlcCanonicalSelector.scalarBase, Mentions]
    · let row :=
        bitRow
          (PiRlcCanonicalSelector.slackBitColumn selectorBase coordinate 1)
      refine ⟨row, ?_, Or.inl ?_⟩
      · unfold row PiRlcCanonicalSelector.scalarRows
          PiRlcCanonicalSelector.acceptanceBoundRows
        apply List.mem_append_left
        apply List.mem_append_left
        exact List.mem_map.2 ⟨1, List.mem_range.2 (by decide), rfl⟩
      · simp [row, bitRow, PiRlcCanonicalSelector.slackBitColumn,
          PiRlcCanonicalSelector.scalarBase, Mentions]
    · let row :=
        bitRow
          (PiRlcCanonicalSelector.slackBitColumn selectorBase coordinate 2)
      refine ⟨row, ?_, Or.inl ?_⟩
      · unfold row PiRlcCanonicalSelector.scalarRows
          PiRlcCanonicalSelector.acceptanceBoundRows
        apply List.mem_append_left
        apply List.mem_append_left
        exact List.mem_map.2 ⟨2, List.mem_range.2 (by decide), rfl⟩
      · simp [row, bitRow, PiRlcCanonicalSelector.slackBitColumn,
          PiRlcCanonicalSelector.scalarBase, Mentions]
    · let row :=
        bitRow
          (PiRlcCanonicalSelector.slackBitColumn selectorBase coordinate 3)
      refine ⟨row, ?_, Or.inl ?_⟩
      · unfold row PiRlcCanonicalSelector.scalarRows
          PiRlcCanonicalSelector.acceptanceBoundRows
        apply List.mem_append_left
        apply List.mem_append_left
        exact List.mem_map.2 ⟨3, List.mem_range.2 (by decide), rfl⟩
      · simp [row, bitRow, PiRlcCanonicalSelector.slackBitColumn,
          PiRlcCanonicalSelector.scalarBase, Mentions]
  · let position : Fin PiRlcCanonicalSelector.outputCount :=
      ⟨(offset - 5) / PiRlcCanonicalSelector.positionAuxiliaryCount, by
        simp only [PiRlcCanonicalSelector.scalarAuxiliaryCount,
          PiRlcCanonicalSelector.outputCount,
          PiRlcCanonicalSelector.positionAuxiliaryCount] at offsetLt ⊢
        omega⟩
    let within :=
      (offset - 5) % PiRlcCanonicalSelector.positionAuxiliaryCount
    have withinLt :
        within < PiRlcCanonicalSelector.positionAuxiliaryCount :=
      Nat.mod_lt _ (by
        simp [PiRlcCanonicalSelector.positionAuxiliaryCount])
    have offsetSplit :
        offset =
          5 + position.val *
              PiRlcCanonicalSelector.positionAuxiliaryCount +
            within := by
      have split :=
        Nat.div_add_mod (offset - 5)
          PiRlcCanonicalSelector.positionAuxiliaryCount
      simp only [position, within] at split ⊢
      rw [Nat.mul_comm] at split
      omega
    by_cases inSelectors :
        within < PiRlcCanonicalSelector.selectionWindow
    · let selectorOffset : Fin PiRlcCanonicalSelector.selectionWindow :=
        ⟨within, inSelectors⟩
      let row :=
        bitRow
          (PiRlcCanonicalSelector.selectorColumn selectorBase coordinate
            position selectorOffset)
      refine ⟨row, ?_, Or.inl ?_⟩
      · unfold row PiRlcCanonicalSelector.scalarRows
          PiRlcCanonicalSelector.positionRows
          PiRlcCanonicalSelector.oneHotRows
        apply List.mem_append_right
        apply List.mem_flatMap.2
        refine ⟨position, List.mem_finRange position, ?_⟩
        apply List.mem_append_left
        apply List.mem_append_left
        apply List.mem_append_left
        exact List.mem_map.2
          ⟨selectorOffset, List.mem_finRange selectorOffset, rfl⟩
      · simp [row, bitRow, PiRlcCanonicalSelector.selectorColumn,
          PiRlcCanonicalSelector.positionBase,
          selectorOffset, offsetSplit, Mentions]
        omega
    · by_cases isOutput : within = 44
      · let row : Row :=
          ⟨[(PiRlcCanonicalSelector.outputColumn selectorBase coordinate
              position, 1)],
            [(0, 1)],
            PiRlcCanonicalSelector.centeredSymbolTerms selectorBase
              coordinate position⟩
        refine ⟨row, ?_, Or.inl ?_⟩
        · unfold row PiRlcCanonicalSelector.scalarRows
            PiRlcCanonicalSelector.positionRows
            PiRlcCanonicalSelector.bindingRows
          apply List.mem_append_right
          apply List.mem_flatMap.2
          refine ⟨position, List.mem_finRange position, ?_⟩
          apply List.mem_append_right
          simp
        · simp [row, PiRlcCanonicalSelector.outputColumn,
            PiRlcCanonicalSelector.positionBase, offsetSplit, isOutput,
            Mentions]
          omega
      · have productBounds :
          11 ≤ within ∧ within < 44 := by
          simp only [PiRlcCanonicalSelector.selectionWindow] at inSelectors
          simp only [PiRlcCanonicalSelector.positionAuxiliaryCount] at withinLt
          omega
        let productIndex := within - 11
        let selectorOffset : Fin PiRlcCanonicalSelector.selectionWindow :=
          ⟨productIndex / 3, by
            simp only [productIndex,
              PiRlcCanonicalSelector.selectionWindow]
            omega⟩
        have stageLt : productIndex % 3 < 3 :=
          Nat.mod_lt _ (by decide)
        have productSplit :
            within = 11 + 3 * selectorOffset.val + productIndex % 3 := by
          have split := Nat.div_add_mod productIndex 3
          simp only [selectorOffset, productIndex] at split ⊢
          omega
        let selected :=
          PiRlcCanonicalSelector.selectorColumn selectorBase coordinate
            position selectorOffset
        let sourceCandidate :=
          PiRlcCanonicalSelector.candidateAt position selectorOffset
        have rowAtMember :
            ∀ row ∈
              PiRlcCanonicalSelector.productRowsAt
                duplexBase u64Base candidateBase selectorBase initial
                coordinate position selectorOffset,
              row ∈
                PiRlcCanonicalSelector.scalarRows
                  duplexBase u64Base candidateBase selectorBase initial
                  coordinate := by
          intro row member
          unfold PiRlcCanonicalSelector.scalarRows
            PiRlcCanonicalSelector.positionRows
            PiRlcCanonicalSelector.productRows
          apply List.mem_append_right
          apply List.mem_flatMap.2
          refine ⟨position, List.mem_finRange position, ?_⟩
          apply List.mem_append_left
          apply List.mem_append_right
          exact List.mem_flatMap.2
            ⟨selectorOffset, List.mem_finRange selectorOffset, member⟩
        have stageCases :
            productIndex % 3 = 0 ∨ productIndex % 3 = 1 ∨
              productIndex % 3 = 2 := by
          omega
        rcases stageCases with stage | stage | stage
        · let row : Row :=
            ⟨[(selected, 1)],
              [(PiRlcCanonicalSelector.symbolSource duplexBase u64Base
                candidateBase initial coordinate sourceCandidate, 1)],
              [(PiRlcCanonicalSelector.symbolProductColumn selectorBase
                coordinate position selectorOffset, 1)]⟩
          refine ⟨row, rowAtMember row ?_, Or.inr (Or.inr ?_)⟩
          · simp [row, PiRlcCanonicalSelector.productRowsAt, selected,
              sourceCandidate]
          · simp [row, PiRlcCanonicalSelector.symbolProductColumn,
              PiRlcCanonicalSelector.positionBase, selectorOffset,
              offsetSplit, productSplit, stage, Mentions]
            omega
        · let row : Row :=
            ⟨[(selected, 1)],
              [(PiRlcCanonicalSelector.acceptSource duplexBase u64Base
                candidateBase initial coordinate sourceCandidate, 1)],
              [(PiRlcCanonicalSelector.acceptProductColumn selectorBase
                coordinate position selectorOffset, 1)]⟩
          refine ⟨row, rowAtMember row ?_, Or.inr (Or.inr ?_)⟩
          · simp [row, PiRlcCanonicalSelector.productRowsAt, selected,
              sourceCandidate]
          · simp [row, PiRlcCanonicalSelector.acceptProductColumn,
              PiRlcCanonicalSelector.symbolProductColumn,
              PiRlcCanonicalSelector.positionBase, selectorOffset,
              offsetSplit, productSplit, stage, Mentions]
            omega
        · let row : Row :=
            ⟨[(selected, 1)],
              PiRlcCanonicalSelector.prefixSource duplexBase u64Base
                candidateBase initial coordinate sourceCandidate,
              [(PiRlcCanonicalSelector.prefixProductColumn selectorBase
                coordinate position selectorOffset, 1)]⟩
          refine ⟨row, rowAtMember row ?_, Or.inr (Or.inr ?_)⟩
          · simp [row, PiRlcCanonicalSelector.productRowsAt, selected,
              sourceCandidate]
          · simp [row, PiRlcCanonicalSelector.prefixProductColumn,
              PiRlcCanonicalSelector.symbolProductColumn,
              PiRlcCanonicalSelector.positionBase, selectorOffset,
              offsetSplit, productSplit, stage, Mentions]
            omega

/-! ## Batched sampler families -/

private theorem canonicalU64Allocation_mem_iff
    (layout : CanonicalU64Recipe.Layout) (column : Nat) :
    column ∈ CanonicalU64Recipe.allocation layout ↔
      layout.base ≤ column ∧
        column < layout.base + CanonicalU64Recipe.auxiliaryCount := by
  constructor
  · exact CanonicalU64Recipe.allocation_in_window layout column
  · intro window
    have offsetLt :
        column - layout.base < CanonicalU64Recipe.auxiliaryCount := by
      omega
    unfold CanonicalU64Recipe.allocation
    by_cases isBit : column - layout.base < 64
    · apply List.mem_append_left
      exact List.mem_map.2
        ⟨column - layout.base,
          List.mem_range.2 isBit,
          by simp [CanonicalU64Recipe.bitColumn]; omega⟩
    · apply List.mem_append_right
      simp only [List.mem_cons, List.not_mem_nil, or_false]
      simp only [CanonicalU64Recipe.auxiliaryCount] at offsetLt
      rcases (show column - layout.base = 64 ∨
          column - layout.base = 65 by omega) with localEq | localEq
      · left
        simp [CanonicalU64Recipe.highFlagColumn]
        omega
      · right
        simp [CanonicalU64Recipe.inverseColumn]
        omega

theorem u64Batch
    (duplexBase u64Base count : Nat)
    (initial : SymbolicDuplex.Builder) :
    RowsCover
      (PiRlcCanonicalU64.rows duplexBase u64Base count initial)
      (PiRlcCanonicalU64.allocation u64Base count) := by
  intro column member
  have window :=
    (PiRlcCanonicalU64.allocation_mem_iff
      u64Base count column).1 member
  let offset := column - u64Base
  let occurrence :=
    offset / CanonicalU64Recipe.auxiliaryCount
  let localOffset :=
    offset % CanonicalU64Recipe.auxiliaryCount
  have offsetEq : column = u64Base + offset := by
    simp only [offset]
    omega
  have localLt :
      localOffset < CanonicalU64Recipe.auxiliaryCount :=
    Nat.mod_lt _ (by
      simp [CanonicalU64Recipe.auxiliaryCount])
  have offsetSplit :
      offset =
        occurrence * CanonicalU64Recipe.auxiliaryCount + localOffset := by
    have split :=
      Nat.div_add_mod offset CanonicalU64Recipe.auxiliaryCount
    simp only [occurrence, localOffset] at split ⊢
    rw [Nat.mul_comm] at split
    exact split.symm
  have occurrenceLt :
      occurrence < count * PiRlcCanonicalU64.lanesPerScalar := by
    simp only [PiRlcCanonicalU64.lanesPerScalar,
      CanonicalU64Recipe.auxiliaryCount] at window localLt offsetSplit ⊢
    omega
  let coordinate : Fin count :=
    ⟨occurrence / PiRlcCanonicalU64.lanesPerScalar, by
      have occurrenceModLt :
          occurrence % PiRlcCanonicalU64.lanesPerScalar <
            PiRlcCanonicalU64.lanesPerScalar :=
        Nat.mod_lt _ (by simp [PiRlcCanonicalU64.lanesPerScalar])
      have occurrenceSplit :=
        Nat.div_add_mod occurrence PiRlcCanonicalU64.lanesPerScalar
      simp only [PiRlcCanonicalU64.lanesPerScalar] at occurrenceLt occurrenceModLt occurrenceSplit
      simp only [PiRlcCanonicalU64.lanesPerScalar]
      omega⟩
  let position : Fin PiRlcCanonicalU64.lanesPerScalar :=
    ⟨occurrence % PiRlcCanonicalU64.lanesPerScalar,
      Nat.mod_lt _ (by simp [PiRlcCanonicalU64.lanesPerScalar])⟩
  have occurrenceSplit :
      occurrence =
        coordinate.val * PiRlcCanonicalU64.lanesPerScalar +
          position.val := by
    have split :=
      Nat.div_add_mod occurrence PiRlcCanonicalU64.lanesPerScalar
    simp only [coordinate, position] at split ⊢
    rw [Nat.mul_comm] at split
    exact split.symm
  let layout :=
    PiRlcCanonicalU64.laneLayout duplexBase u64Base initial
      coordinate position
  have localMember :
      column ∈ CanonicalU64Recipe.allocation layout := by
    rw [canonicalU64Allocation_mem_iff]
    simp only [layout, PiRlcCanonicalU64.laneLayout,
      PiRlcCanonicalU64.occurrenceIndex]
    simp only [PiRlcCanonicalU64.lanesPerScalar,
      CanonicalU64Recipe.auxiliaryCount] at offsetEq offsetSplit occurrenceSplit localLt ⊢
    omega
  rcases canonicalU64 layout column localMember with
    ⟨row, rowMember, mentioned⟩
  refine ⟨row, ?_, mentioned⟩
  unfold PiRlcCanonicalU64.rows PiRlcCanonicalU64.scalarRows
  apply List.mem_flatMap.2
  refine ⟨coordinate, List.mem_finRange coordinate, ?_⟩
  apply List.mem_flatMap.2
  exact ⟨position, List.mem_finRange position, rowMember⟩

theorem candidateBatch
    (duplexBase u64Base candidateBase count : Nat)
    (initial : SymbolicDuplex.Builder) :
    RowsCover
      (PiRlcCanonicalCandidates.rows
        duplexBase u64Base candidateBase count initial)
      (PiRlcCanonicalCandidates.allocation candidateBase count) := by
  intro column member
  have window :=
    (PiRlcCanonicalCandidates.allocation_mem_iff
      candidateBase count column).1 member
  let offset := column - candidateBase
  let occurrence :=
    offset / PiRlcCanonicalCandidate.auxiliaryCount
  let localOffset :=
    offset % PiRlcCanonicalCandidate.auxiliaryCount
  have offsetEq : column = candidateBase + offset := by
    simp only [offset]
    omega
  have localLt :
      localOffset < PiRlcCanonicalCandidate.auxiliaryCount :=
    Nat.mod_lt _ (by
      simp [PiRlcCanonicalCandidate.auxiliaryCount])
  have offsetSplit :
      offset =
        occurrence * PiRlcCanonicalCandidate.auxiliaryCount +
          localOffset := by
    have split :=
      Nat.div_add_mod offset PiRlcCanonicalCandidate.auxiliaryCount
    simp only [occurrence, localOffset] at split ⊢
    rw [Nat.mul_comm] at split
    exact split.symm
  have occurrenceLt :
      occurrence <
        count * PiRlcCanonicalCandidates.candidatesPerScalar := by
    simp only [PiRlcCanonicalCandidates.candidatesPerScalar,
      PiRlcCanonicalCandidate.auxiliaryCount] at window localLt offsetSplit ⊢
    omega
  let coordinate : Fin count :=
    ⟨occurrence / PiRlcCanonicalCandidates.candidatesPerScalar, by
      have occurrenceModLt :
          occurrence % PiRlcCanonicalCandidates.candidatesPerScalar <
            PiRlcCanonicalCandidates.candidatesPerScalar :=
        Nat.mod_lt _ (by
          simp [PiRlcCanonicalCandidates.candidatesPerScalar])
      have occurrenceSplit :=
        Nat.div_add_mod occurrence
          PiRlcCanonicalCandidates.candidatesPerScalar
      simp only [PiRlcCanonicalCandidates.candidatesPerScalar] at occurrenceLt occurrenceModLt occurrenceSplit
      simp only [PiRlcCanonicalCandidates.candidatesPerScalar]
      omega⟩
  let candidateIndex :
      Fin PiRlcCanonicalCandidates.candidatesPerScalar :=
    ⟨occurrence % PiRlcCanonicalCandidates.candidatesPerScalar,
      Nat.mod_lt _ (by
        simp [PiRlcCanonicalCandidates.candidatesPerScalar])⟩
  have occurrenceSplit :
      occurrence =
        coordinate.val * PiRlcCanonicalCandidates.candidatesPerScalar +
          candidateIndex.val := by
    have split :=
      Nat.div_add_mod occurrence
        PiRlcCanonicalCandidates.candidatesPerScalar
    simp only [coordinate, candidateIndex] at split ⊢
    rw [Nat.mul_comm] at split
    exact split.symm
  let layout :=
    PiRlcCanonicalCandidates.candidateLayout
      duplexBase u64Base candidateBase initial coordinate candidateIndex
  have localMember :
      column ∈ PiRlcCanonicalCandidate.allocation layout := by
    rw [PiRlcCanonicalCandidate.allocation_mem_iff]
    simp only [layout, PiRlcCanonicalCandidates.candidateLayout,
      PiRlcCanonicalCandidates.occurrenceBase,
      PiRlcCanonicalCandidates.occurrenceIndex]
    simp only [PiRlcCanonicalCandidates.candidatesPerScalar,
      PiRlcCanonicalCandidate.auxiliaryCount] at offsetEq offsetSplit occurrenceSplit localLt ⊢
    omega
  rcases candidate layout column localMember with
    ⟨row, rowMember, mentioned⟩
  refine ⟨row, ?_, mentioned⟩
  unfold PiRlcCanonicalCandidates.rows
    PiRlcCanonicalCandidates.scalarRows
  apply List.mem_flatMap.2
  refine ⟨coordinate, List.mem_finRange coordinate, ?_⟩
  apply List.mem_flatMap.2
  exact ⟨candidateIndex, List.mem_finRange candidateIndex, rowMember⟩

theorem selectorBatch
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initial : SymbolicDuplex.Builder) :
    RowsCover
      (PiRlcCanonicalSelector.rows
        duplexBase u64Base candidateBase selectorBase count initial)
      (PiRlcCanonicalSelector.allocation selectorBase count) := by
  intro column member
  have window :=
    (PiRlcCanonicalSelector.allocation_mem_iff
      selectorBase count column).1 member
  let offset := column - selectorBase
  have offsetEq : column = selectorBase + offset := by
    simp only [offset]
    omega
  let coordinate : Fin count :=
    ⟨offset / PiRlcCanonicalSelector.scalarAuxiliaryCount, by
      have offsetModLt :
          offset % PiRlcCanonicalSelector.scalarAuxiliaryCount <
            PiRlcCanonicalSelector.scalarAuxiliaryCount :=
        Nat.mod_lt _ (by
          simp [PiRlcCanonicalSelector.scalarAuxiliaryCount,
            PiRlcCanonicalSelector.outputCount,
            PiRlcCanonicalSelector.positionAuxiliaryCount])
      have offsetSplit :=
        Nat.div_add_mod offset
          PiRlcCanonicalSelector.scalarAuxiliaryCount
      simp only [PiRlcCanonicalSelector.scalarAuxiliaryCount,
        PiRlcCanonicalSelector.outputCount,
        PiRlcCanonicalSelector.positionAuxiliaryCount] at window offsetEq offsetModLt offsetSplit
      simp only [PiRlcCanonicalSelector.scalarAuxiliaryCount,
        PiRlcCanonicalSelector.outputCount,
        PiRlcCanonicalSelector.positionAuxiliaryCount]
      omega⟩
  let localOffset :=
    offset % PiRlcCanonicalSelector.scalarAuxiliaryCount
  have localLt :
      localOffset < PiRlcCanonicalSelector.scalarAuxiliaryCount :=
    Nat.mod_lt _ (by
      simp [PiRlcCanonicalSelector.scalarAuxiliaryCount,
        PiRlcCanonicalSelector.outputCount,
        PiRlcCanonicalSelector.positionAuxiliaryCount])
  have offsetSplit :
      offset =
        coordinate.val * PiRlcCanonicalSelector.scalarAuxiliaryCount +
          localOffset := by
    have split :=
      Nat.div_add_mod offset
        PiRlcCanonicalSelector.scalarAuxiliaryCount
    simp only [coordinate, localOffset] at split ⊢
    rw [Nat.mul_comm] at split
    exact split.symm
  rcases selectorScalar duplexBase u64Base candidateBase selectorBase
      initial coordinate localOffset localLt with
    ⟨row, rowMember, mentioned⟩
  refine ⟨row, ?_, ?_⟩
  · unfold PiRlcCanonicalSelector.rows
    exact List.mem_flatMap.2
      ⟨coordinate, List.mem_finRange coordinate, rowMember⟩
  · simp only [PiRlcCanonicalSelector.scalarBase]
      at mentioned
    have mentionedColumnEq :
        selectorBase +
              coordinate.val *
                PiRlcCanonicalSelector.scalarAuxiliaryCount +
            localOffset =
          column := by
      omega
    rw [mentionedColumnEq] at mentioned
    exact mentioned

theorem suffix
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initial : SymbolicDuplex.Builder) :
    RowsCover
      (PiRlcCanonicalSamplerHonest.suffixRows
        duplexBase u64Base candidateBase selectorBase count initial)
      (PiRlcCanonicalSamplerHonest.suffixAllocation
        u64Base candidateBase selectorBase count) := by
  unfold PiRlcCanonicalSamplerHonest.suffixRows
    PiRlcCanonicalSamplerHonest.suffixAllocation
  apply AllocationCoverage.append
  · apply AllocationCoverage.append
    · exact u64Batch duplexBase u64Base count initial
    · exact candidateBatch
        duplexBase u64Base candidateBase count initial
  · exact selectorBatch
      duplexBase u64Base candidateBase selectorBase count initial

theorem transcript
    (duplexBase : Nat) (constants : Constants) (lanes : State) :
    RowsCover
      (PiRlcCanonicalSamplerProgram.transcriptRows
        duplexBase constants lanes)
      (PiRlcCanonicalSamplerProgram.transcriptAllocation duplexBase) := by
  intro column member
  have compactMember :
      column ∈
        SymbolicDuplexPhysical.temporaryColumns
          duplexBase
          (PiRlcCanonicalSymbolicMachineHonest.fixedBuilder
            duplexBase lanes).entries.length := by
    simpa only [
      PiRlcCanonicalSamplerProgram.transcriptAllocation,
      PiRlcCanonicalSymbolicMachineHonest.fixedAllocation,
      PiRlcCanonicalSymbolicMachineHonest.fixedBuilder_entries_length] using
      member
  rcases SymbolicDuplexPhysical.temporaryColumns_written_of_calls
      duplexBase constants
      (PiRlcCanonicalSymbolicMachineHonest.fixedBuilder duplexBase lanes)
      (PiRlcCanonicalSymbolicMachineCallOrder.fixedBuilder
        duplexBase lanes)
      column compactMember with
    ⟨row, rowMember, mentioned⟩
  exact ⟨row, by
    simpa only [PiRlcCanonicalSamplerProgram.transcriptRows] using rowMember,
    Or.inr (Or.inr mentioned)⟩

theorem samplerProgram
    (duplexBase : Nat) (constants : Constants) (lanes : State) :
    RowsCover
      (PiRlcCanonicalSamplerProgram.rows duplexBase constants lanes)
      (PiRlcCanonicalSamplerProgram.allocation duplexBase) := by
  unfold PiRlcCanonicalSamplerProgram.rows
    PiRlcCanonicalSamplerProgram.allocation
  apply AllocationCoverage.append
  · exact transcript duplexBase constants lanes
  · exact suffix duplexBase
      (PiRlcCanonicalSamplerProgram.u64Base duplexBase)
      (PiRlcCanonicalSamplerProgram.candidateBase duplexBase)
      (PiRlcCanonicalSamplerProgram.selectorBase duplexBase)
      PiRlcCanonicalSamplerProgram.coordinateCount
      (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)

end Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerCoverage
