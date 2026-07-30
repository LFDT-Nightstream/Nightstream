import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerProgram

/-!
Contract: expose the exact dense membership interval of the complete
fixed-active ΠRLC sampler allocation.

The allocation remains the concatenation emitted by its four physical
families; this theorem proves that the concatenation has neither gaps nor
padding.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerAllocation

open Nightstream.Implementation.R1CS.Canonical

private theorem u64Base_eq (base : Nat) :
    PiRlcCanonicalSamplerProgram.u64Base base = base + 26400 := by
  simp [PiRlcCanonicalSamplerProgram.u64Base,
    PiRlcCanonicalSamplerProgram.transcriptCalls,
    SymbolicDuplex.stride]

private theorem candidateBase_eq (base : Nat) :
    PiRlcCanonicalSamplerProgram.candidateBase base = base + 42240 := by
  simp [PiRlcCanonicalSamplerProgram.candidateBase,
    PiRlcCanonicalCandidatesBatchHonest.u64End,
    u64Base_eq, PiRlcCanonicalSamplerProgram.coordinateCount,
    PiRlcCanonicalU64.lanesPerScalar,
    CanonicalU64Recipe.auxiliaryCount]

private theorem selectorBase_eq (base : Nat) :
    PiRlcCanonicalSamplerProgram.selectorBase base = base + 63360 := by
  simp [PiRlcCanonicalSamplerProgram.selectorBase,
    PiRlcCanonicalSelectorBatchHonest.candidateEnd,
    candidateBase_eq, PiRlcCanonicalSamplerProgram.coordinateCount,
    PiRlcCanonicalCandidates.candidatesPerScalar,
    PiRlcCanonicalCandidate.auxiliaryCount]

/-- Exact dense membership of the selected sampler's four-part allocation. -/
theorem allocation_mem_iff (base column : Nat) :
    column ∈ PiRlcCanonicalSamplerProgram.allocation base ↔
      base ≤ column ∧
        column <
          base + PiRlcCanonicalSamplerProgram.cost.auxiliaryColumns := by
  rw [PiRlcCanonicalSamplerProgram.allocation]
  simp only [List.mem_append]
  constructor
  · intro member
    rcases member with inTranscript | inSuffix
    · have bounds :=
        (SymbolicDuplexPhysical.temporaryColumns_mem_iff base 75 column).1
          (by
            simpa [PiRlcCanonicalSamplerProgram.transcriptAllocation,
              PiRlcCanonicalSymbolicMachineHonest.fixedAllocation] using
              inTranscript)
      rw [SymbolicDuplex.stride_eq] at bounds
      simpa [PiRlcCanonicalSamplerProgram.cost] using
        ⟨bounds.1, by omega⟩
    · unfold PiRlcCanonicalSamplerProgram.suffixAllocation
        PiRlcCanonicalSamplerHonest.suffixAllocation at inSuffix
      simp only [List.mem_append] at inSuffix
      rcases inSuffix with (inU64 | inCandidate) | inSelector
      · have bounds :=
          (PiRlcCanonicalU64.allocation_mem_iff
            (PiRlcCanonicalSamplerProgram.u64Base base)
            PiRlcCanonicalSamplerProgram.coordinateCount column).1 inU64
        simp only [u64Base_eq,
          PiRlcCanonicalSamplerProgram.coordinateCount,
          PiRlcCanonicalU64.lanesPerScalar,
          CanonicalU64Recipe.auxiliaryCount] at bounds
        simpa [PiRlcCanonicalSamplerProgram.cost] using
          ⟨by omega, by omega⟩
      · have bounds :=
          (PiRlcCanonicalCandidates.allocation_mem_iff
            (PiRlcCanonicalSamplerProgram.candidateBase base)
            PiRlcCanonicalSamplerProgram.coordinateCount column).1
            inCandidate
        simp only [candidateBase_eq,
          PiRlcCanonicalSamplerProgram.coordinateCount,
          PiRlcCanonicalCandidates.candidatesPerScalar,
          PiRlcCanonicalCandidate.auxiliaryCount] at bounds
        simpa [PiRlcCanonicalSamplerProgram.cost] using
          ⟨by omega, by omega⟩
      · have bounds :=
          (PiRlcCanonicalSelector.allocation_mem_iff
            (PiRlcCanonicalSamplerProgram.selectorBase base)
            PiRlcCanonicalSamplerProgram.coordinateCount column).1
            inSelector
        simp only [selectorBase_eq,
          PiRlcCanonicalSamplerProgram.coordinateCount,
          PiRlcCanonicalSelector.scalarAuxiliaryCount,
          PiRlcCanonicalSelector.outputCount,
          PiRlcCanonicalSelector.positionAuxiliaryCount] at bounds
        simpa [PiRlcCanonicalSamplerProgram.cost] using
          ⟨by omega, by omega⟩
  · intro bounds
    by_cases beforeU64 :
        column < PiRlcCanonicalSamplerProgram.u64Base base
    · left
      have transcriptMember :
          column ∈ SymbolicDuplexPhysical.temporaryColumns base 75 :=
        (SymbolicDuplexPhysical.temporaryColumns_mem_iff base 75 column).2
          ⟨bounds.1, by simpa [u64Base_eq] using beforeU64⟩
      simpa [PiRlcCanonicalSamplerProgram.transcriptAllocation,
        PiRlcCanonicalSymbolicMachineHonest.fixedAllocation] using
        transcriptMember
    · right
      unfold PiRlcCanonicalSamplerProgram.suffixAllocation
        PiRlcCanonicalSamplerHonest.suffixAllocation
      simp only [List.mem_append]
      by_cases beforeCandidate :
          column < PiRlcCanonicalSamplerProgram.candidateBase base
      · left
        left
        apply
          (PiRlcCanonicalU64.allocation_mem_iff
            (PiRlcCanonicalSamplerProgram.u64Base base)
            PiRlcCanonicalSamplerProgram.coordinateCount column).2
        simp only [u64Base_eq, candidateBase_eq,
          PiRlcCanonicalSamplerProgram.coordinateCount,
          PiRlcCanonicalU64.lanesPerScalar,
          CanonicalU64Recipe.auxiliaryCount] at beforeU64 beforeCandidate ⊢
        omega
      · by_cases beforeSelector :
          column < PiRlcCanonicalSamplerProgram.selectorBase base
        · left
          right
          apply
            (PiRlcCanonicalCandidates.allocation_mem_iff
              (PiRlcCanonicalSamplerProgram.candidateBase base)
              PiRlcCanonicalSamplerProgram.coordinateCount column).2
          simp only [candidateBase_eq, selectorBase_eq,
            PiRlcCanonicalSamplerProgram.coordinateCount,
            PiRlcCanonicalCandidates.candidatesPerScalar,
            PiRlcCanonicalCandidate.auxiliaryCount] at beforeCandidate beforeSelector ⊢
          omega
        · right
          apply
            (PiRlcCanonicalSelector.allocation_mem_iff
              (PiRlcCanonicalSamplerProgram.selectorBase base)
              PiRlcCanonicalSamplerProgram.coordinateCount column).2
          simp only [selectorBase_eq,
            PiRlcCanonicalSamplerProgram.coordinateCount,
            PiRlcCanonicalSelector.scalarAuxiliaryCount,
            PiRlcCanonicalSelector.outputCount,
            PiRlcCanonicalSelector.positionAuxiliaryCount,
            PiRlcCanonicalSamplerProgram.cost] at beforeSelector bounds ⊢
          omega

end Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerAllocation
