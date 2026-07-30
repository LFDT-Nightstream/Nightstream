import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidateConservation
import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerProgram
import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelectorConservation

/-!
Contract: exact categorical column conservation for the complete Lean-owned
fixed-active `Pi_RLC` sampler program.

Every operand of every emitted row is either in the caller-owned authoritative
prefix strictly below `duplexBase`, or in the program's exact declared
allocation.  In particular, the downstream sampler cannot read an arbitrary
column merely because it happens to lie below a later family boundary.

Assurance tier: model-level canonical encoding.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerProgramConservation

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerProgram

def Allowed (duplexBase column : Nat) : Prop :=
  column < duplexBase ∨
    column ∈ PiRlcCanonicalSamplerProgram.allocation duplexBase

private theorem allowed_prefix
    (duplexBase column : Nat) (before : column < duplexBase) :
    Allowed duplexBase column :=
  Or.inl before

private theorem allowed_transcript
    (duplexBase column : Nat)
    (member :
      column ∈ PiRlcCanonicalSamplerProgram.transcriptAllocation
        duplexBase) :
    Allowed duplexBase column := by
  right
  unfold PiRlcCanonicalSamplerProgram.allocation
  exact List.mem_append_left _ member

private theorem allowed_suffix
    (duplexBase column : Nat)
    (member :
      column ∈ PiRlcCanonicalSamplerProgram.suffixAllocation duplexBase) :
    Allowed duplexBase column := by
  right
  unfold PiRlcCanonicalSamplerProgram.allocation
  exact List.mem_append_right _ member

private theorem allowed_constant
    (duplexBase : Nat) (positive : 0 < duplexBase) :
    Allowed duplexBase 0 :=
  allowed_prefix duplexBase 0 positive

private theorem u64Allocation_allowed
    (duplexBase column : Nat)
    (member :
      column ∈ PiRlcCanonicalU64.allocation
        (u64Base duplexBase) coordinateCount) :
    Allowed duplexBase column := by
  apply allowed_suffix
  unfold PiRlcCanonicalSamplerProgram.suffixAllocation
    PiRlcCanonicalSamplerHonest.suffixAllocation
  exact List.mem_append_left _
    (List.mem_append_left _ member)

private theorem candidateAllocation_allowed
    (duplexBase column : Nat)
    (member :
      column ∈ PiRlcCanonicalCandidates.allocation
        (candidateBase duplexBase) coordinateCount) :
    Allowed duplexBase column := by
  apply allowed_suffix
  unfold PiRlcCanonicalSamplerProgram.suffixAllocation
    PiRlcCanonicalSamplerHonest.suffixAllocation
  exact List.mem_append_left _
    (List.mem_append_right _ member)

private theorem selectorAllocation_allowed
    (duplexBase column : Nat)
    (member :
      column ∈ PiRlcCanonicalSelector.allocation
        (selectorBase duplexBase) coordinateCount) :
    Allowed duplexBase column := by
  apply allowed_suffix
  unfold PiRlcCanonicalSamplerProgram.suffixAllocation
    PiRlcCanonicalSamplerHonest.suffixAllocation
  exact List.mem_append_right _ member

private theorem mentions_has_coefficient
    (comb : LinCombNormal.LinComb) (column : Nat)
    (mentioned : Mentions comb column) :
    ∃ coefficient, (column, coefficient) ∈ comb := by
  unfold Mentions at mentioned
  rcases List.mem_map.mp mentioned with
    ⟨⟨source, coefficient⟩, member, equal⟩
  simp only at equal
  subst source
  exact ⟨coefficient, member⟩

private theorem bitColumn_mem_allocation
    (layout : CanonicalU64Recipe.Layout)
    (index : Nat) (bounded : index < 64) :
    CanonicalU64Recipe.bitColumn layout index ∈
      CanonicalU64Recipe.allocation layout := by
  unfold CanonicalU64Recipe.allocation
  apply List.mem_append_left
  exact List.mem_map.mpr
    ⟨index, List.mem_range.mpr bounded, rfl⟩

private theorem u64Rows_conservation
    (duplexBase : Nat) (lanes : State)
    (positive : 0 < duplexBase)
    (row : Row)
    (rowMember :
      row ∈
        PiRlcCanonicalU64.rows duplexBase (u64Base duplexBase)
          coordinateCount
          (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes))
    (column : Nat)
    (mentioned :
      Mentions row.a column ∨ Mentions row.b column ∨
        Mentions row.c column) :
    Allowed duplexBase column := by
  rcases List.mem_flatMap.mp rowMember with
    ⟨coordinate, _, scalarMember⟩
  rcases List.mem_flatMap.mp scalarMember with
    ⟨position, _, localMember⟩
  let layout :=
    PiRlcCanonicalU64.laneLayout duplexBase (u64Base duplexBase)
      (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)
      coordinate position
  have classify :
      column = 0 ∨
        (∃ inputCoefficient,
          (column, inputCoefficient) ∈ layout.input) ∨
        column ∈ CanonicalU64Recipe.allocation layout := by
    rcases mentioned with inA | inB | inC
    · rcases mentions_has_coefficient row.a column inA with
        ⟨coefficient, member⟩
      exact CanonicalU64Recipe.rows_conservation layout row localMember
        column coefficient (Or.inl member)
    · rcases mentions_has_coefficient row.b column inB with
        ⟨coefficient, member⟩
      exact CanonicalU64Recipe.rows_conservation layout row localMember
        column coefficient (Or.inr (Or.inl member))
    · rcases mentions_has_coefficient row.c column inC with
        ⟨coefficient, member⟩
      exact CanonicalU64Recipe.rows_conservation layout row localMember
        column coefficient (Or.inr (Or.inr member))
  rcases classify with constant | input | localAllocation
  · subst column
    exact allowed_constant duplexBase positive
  · rcases input with ⟨coefficient, inputMember⟩
    have transcriptMember :=
      PiRlcCanonicalU64Placement.laneInput_member_temporaryColumns
        duplexBase coordinateCount
        (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)
        rfl coordinate position column coefficient
        (by
          simpa [layout, PiRlcCanonicalU64.laneLayout] using inputMember)
    apply allowed_transcript
    simpa [PiRlcCanonicalSamplerProgram.transcriptAllocation,
      PiRlcCanonicalSymbolicMachineHonest.fixedAllocation,
      PiRlcCanonicalSymbolicMachineHonest.initialBuilder,
      SymbolicDuplex.start, coordinateCount] using transcriptMember
  · apply u64Allocation_allowed
    exact PiRlcCanonicalU64.lane_allocation_mem
      duplexBase (u64Base duplexBase)
      (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)
      coordinate position column
      (by simpa [layout] using localAllocation)

private theorem sourceBit_allowed
    (duplexBase : Nat) (lanes : State)
    (coordinate : Fin coordinateCount)
    (candidate : Fin PiRlcCanonicalCandidates.candidatesPerScalar)
    (index : Fin PiRlcCanonicalCandidate.sourceBitCount) :
    Allowed duplexBase
      ((PiRlcCanonicalCandidates.candidateLayout
        duplexBase (u64Base duplexBase) (candidateBase duplexBase)
        (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)
        coordinate candidate).sourceBit index) := by
  apply u64Allocation_allowed
  apply PiRlcCanonicalU64.lane_allocation_mem
    duplexBase (u64Base duplexBase)
    (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)
    coordinate (PiRlcCanonicalCandidates.lanePosition candidate)
  unfold PiRlcCanonicalCandidates.candidateLayout
  exact bitColumn_mem_allocation _ _
    (PiRlcCanonicalCandidates.sourceBitIndex_lt candidate index)

private theorem prior_allowed
    (duplexBase : Nat) (lanes : State)
    (coordinate : Fin coordinateCount)
    (candidate : Fin PiRlcCanonicalCandidates.candidatesPerScalar)
    (column : Nat)
    (mentioned :
      Mentions
        (PiRlcCanonicalCandidates.candidateLayout
          duplexBase (u64Base duplexBase) (candidateBase duplexBase)
          (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)
          coordinate candidate).prior column) :
    Allowed duplexBase column := by
  change
    Mentions
      (PiRlcCanonicalCandidates.prior
        (candidateBase duplexBase) coordinate candidate) column at mentioned
  unfold PiRlcCanonicalCandidates.prior at mentioned
  split at mentioned
  · simp [Mentions] at mentioned
  · have equal :=
      (mentions_single
        (PiRlcCanonicalCandidates.occurrenceBase
          (candidateBase duplexBase) coordinate candidate - 1)
        column 1).mp mentioned
    subst column
    apply candidateAllocation_allowed
    rw [PiRlcCanonicalCandidates.allocation_mem_iff]
    have coordinateLt := coordinate.isLt
    have candidateLt := candidate.isLt
    simp only [PiRlcCanonicalCandidates.occurrenceBase,
      PiRlcCanonicalCandidates.occurrenceIndex,
      PiRlcCanonicalCandidates.candidatesPerScalar,
      PiRlcCanonicalCandidate.auxiliaryCount] at *
    omega

private theorem candidateRows_conservation
    (duplexBase : Nat) (lanes : State)
    (positive : 0 < duplexBase)
    (row : Row)
    (rowMember :
      row ∈
        PiRlcCanonicalCandidates.rows duplexBase (u64Base duplexBase)
          (candidateBase duplexBase) coordinateCount
          (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes))
    (column : Nat)
    (mentioned :
      Mentions row.a column ∨ Mentions row.b column ∨
        Mentions row.c column) :
    Allowed duplexBase column := by
  rcases List.mem_flatMap.mp rowMember with
    ⟨coordinate, _, scalarMember⟩
  rcases List.mem_flatMap.mp scalarMember with
    ⟨candidate, _, localMember⟩
  let layout :=
    PiRlcCanonicalCandidates.candidateLayout
      duplexBase (u64Base duplexBase) (candidateBase duplexBase)
      (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)
      coordinate candidate
  have classify :=
    PiRlcCanonicalCandidateConservation.rows_conservation
      layout row localMember column mentioned
  rcases classify with constant | source | prior | localAllocation
  · subst column
    exact allowed_constant duplexBase positive
  · rcases source with ⟨index, rfl⟩
    exact sourceBit_allowed duplexBase lanes coordinate candidate index
  · exact prior_allowed duplexBase lanes coordinate candidate column
      (by simpa [layout] using prior)
  · apply candidateAllocation_allowed
    exact PiRlcCanonicalCandidates.occurrence_allocation_mem
      duplexBase (u64Base duplexBase) (candidateBase duplexBase)
      (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)
      coordinate candidate column
      (by simpa [layout] using localAllocation)

private theorem selectorRows_conservation
    (duplexBase : Nat) (lanes : State)
    (positive : 0 < duplexBase)
    (row : Row)
    (rowMember :
      row ∈
        PiRlcCanonicalSelector.rows duplexBase (u64Base duplexBase)
          (candidateBase duplexBase) (selectorBase duplexBase)
          coordinateCount
          (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes))
    (column : Nat)
    (mentioned :
      Mentions row.a column ∨ Mentions row.b column ∨
        Mentions row.c column) :
    Allowed duplexBase column := by
  have classify :=
    PiRlcCanonicalSelectorConservation.rows_conservation
      duplexBase (u64Base duplexBase) (candidateBase duplexBase)
      (selectorBase duplexBase) coordinateCount
      (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)
      row rowMember column mentioned
  rcases classify with constant | candidate | selector
  · subst column
    exact allowed_constant duplexBase positive
  · exact candidateAllocation_allowed duplexBase column candidate
  · exact selectorAllocation_allowed duplexBase column selector

/-- Exact downstream conservation: every suffix operand is either in the
authoritative prefix, the exact transcript allocation, or the exact suffix
allocation. -/
theorem suffixRows_conservation
    (duplexBase : Nat) (lanes : State)
    (positive : 0 < duplexBase)
    (row : Row)
    (rowMember : row ∈ suffixRows duplexBase lanes)
    (column : Nat)
    (mentioned :
      Mentions row.a column ∨ Mentions row.b column ∨
        Mentions row.c column) :
    Allowed duplexBase column := by
  unfold suffixRows PiRlcCanonicalSamplerHonest.suffixRows at rowMember
  simp only [List.mem_append] at rowMember
  rcases rowMember with (inU64 | inCandidates) | inSelectors
  · exact u64Rows_conservation duplexBase lanes positive row inU64 column
      mentioned
  · exact candidateRows_conservation duplexBase lanes positive row
      inCandidates column mentioned
  · exact selectorRows_conservation duplexBase lanes positive row
      inSelectors column mentioned

/-- Every operand of every row in the complete fixed-active sampler belongs
to the authoritative prefix or the program's exact declared allocation. -/
theorem rows_conservation
    (duplexBase : Nat) (constants : Constants) (lanes : State)
    (positive : 0 < duplexBase)
    (lanesInPrefix :
      ∀ lane : Fin width,
        SymbolicDuplexPlacement.ValueInPrefix duplexBase (lanes lane))
    (row : Row)
    (rowMember : row ∈ rows duplexBase constants lanes)
    (column : Nat)
    (mentioned :
      Mentions row.a column ∨ Mentions row.b column ∨
        Mentions row.c column) :
    Allowed duplexBase column := by
  unfold rows at rowMember
  rcases List.mem_append.mp rowMember with inTranscript | inSuffix
  · rcases
      PiRlcCanonicalSymbolicMachineHonest.fixedRows_conservation
        duplexBase constants lanes positive lanesInPrefix row inTranscript
        column mentioned with inPrefix | inAllocation
    · exact allowed_prefix duplexBase column inPrefix
    · exact allowed_transcript duplexBase column
        (by simpa [PiRlcCanonicalSamplerProgram.transcriptAllocation] using
          inAllocation)
  · exact suffixRows_conservation duplexBase lanes positive row inSuffix
      column mentioned

end Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerProgramConservation
