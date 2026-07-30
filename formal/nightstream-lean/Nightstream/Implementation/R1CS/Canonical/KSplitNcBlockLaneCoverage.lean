import Nightstream.Implementation.R1CS.Canonical.KFrameAllocationCoverage
import Nightstream.Implementation.R1CS.Canonical.KSplitNcBlockLaneRows

/-!
Contract: exact allocation coverage for the FE and block×lane NC claimed
chains inside the operational Split-NC verifier.

The allocation is reconstructed from the fixed-phase Horner frames actually
emitted by the two chains.  Source, challenge, boundary, and terminal columns
remain shared reads and are not counted here.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KSplitNcBlockLaneCoverage

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.R1CS.Canonical.AllocationCoverage
open Nightstream.Implementation.R1CS.Canonical.KFrameAllocationCoverage
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhaseSemanticOccurrence
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

/-- The mixed-width FE prefix covers both of its exact Horner intervals. -/
theorem fe
    {degree : Nat}
    (columns : KSplitNcFeRows.Columns degree)
    (base : Nat) :
    RowsCover
      (KSplitNcFeRows.rows columns base)
      (KFrames.frameColumns base
        (columns.rowRounds.length * degree +
          columns.laneRounds.length * 2)) := by
  intro column member
  rw [KFrames.frameColumns_mem_iff] at member
  by_cases inRows :
      column < base + 3 * (columns.rowRounds.length * degree)
  · have rowColumn :
        column ∈
          KFrames.frameColumns base
            (columns.rowRounds.length * degree) := by
      rw [KFrames.frameColumns_mem_iff]
      omega
    have covered :=
      fixedPhase
        (carried columns.initial)
        columns.rowSource.rowRounds
        columns.rowSource.rowChallenges
        (carried columns.boundary)
        base
        (by
          simpa [KSplitNcFeRows.Columns.rowSource,
            SourceColumns.rowRounds,
            SourceColumns.rowChallenges] using
            columns.rowSameLength)
    have rowRoundsLength :
        columns.rowSource.rowRounds.length =
          columns.rowRounds.length := by
      simp [KSplitNcFeRows.Columns.rowSource,
        SourceColumns.rowRounds]
    rw [rowRoundsLength] at covered
    rcases covered column rowColumn with ⟨row, rowMember, mentioned⟩
    exact
      ⟨row, List.mem_append_left _ rowMember, mentioned⟩
  · have laneColumn :
        column ∈
          KFrames.frameColumns (KSplitNcFeRows.laneBase columns base)
            (columns.laneRounds.length * 2) := by
      rw [KFrames.frameColumns_mem_iff]
      unfold KSplitNcFeRows.laneBase
      have rowWidth :
          columns.rowRounds.length * (3 * degree) =
            3 * (columns.rowRounds.length * degree) := by
        calc
          columns.rowRounds.length * (3 * degree) =
              (columns.rowRounds.length * 3) * degree := by
                rw [Nat.mul_assoc]
          _ = (3 * columns.rowRounds.length) * degree := by
                rw [Nat.mul_comm columns.rowRounds.length 3]
          _ = 3 * (columns.rowRounds.length * degree) := by
                rw [Nat.mul_assoc]
      rw [rowWidth]
      omega
    have covered :=
      fixedPhase
        (carried columns.boundary)
        columns.laneSource.rowRounds
        columns.laneSource.rowChallenges
        (carried columns.terminal)
        (KSplitNcFeRows.laneBase columns base)
        (by
          simpa [KSplitNcFeRows.Columns.laneSource,
            SourceColumns.rowRounds,
            SourceColumns.rowChallenges] using
            columns.laneSameLength)
    have laneRoundsLength :
        columns.laneSource.rowRounds.length =
          columns.laneRounds.length := by
      simp [KSplitNcFeRows.Columns.laneSource,
        SourceColumns.rowRounds]
    rw [laneRoundsLength] at covered
    rcases covered column laneColumn with ⟨row, rowMember, mentioned⟩
    exact
      ⟨row, List.mem_append_right _ rowMember, mentioned⟩

/-- The degree-four NC suffix covers its exact Horner interval. -/
theorem nc
    (columns : KSplitNcNcRows.Columns)
    (base : Nat) :
    RowsCover
      (KSplitNcNcRows.rows columns base)
      (KFrames.frameColumns base (columns.rounds.length * 4)) := by
  have covered :=
    fixedPhase
      (carried columns.current)
      columns.rowRounds
      columns.rowChallenges
      (carried columns.terminal)
      base
      (by
        simpa [SourceColumns.rowRounds,
          SourceColumns.rowChallenges] using
          columns.sameLength)
  have roundsLength :
      columns.rowRounds.length = columns.rounds.length := by
    simp [SourceColumns.rowRounds]
  simpa only [roundsLength] using covered

/-- The combined FE→NC numeric program covers every auxiliary column counted
by its cost. -/
theorem rows
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domains : Domains}
    (columns : KSplitNcBlockLaneRows.Columns input domains)
    (base : Nat) :
    RowsCover
      (KSplitNcBlockLaneRows.rows columns base)
      (KFrames.frameColumns base
        (columns.fe.rowRounds.length * SumCheck.Fe.Drow input +
          columns.fe.laneRounds.length * 2 +
          columns.nc.rounds.length * 4)) := by
  intro column member
  rw [KFrames.frameColumns_mem_iff] at member
  let feFrames :=
    columns.fe.rowRounds.length * SumCheck.Fe.Drow input +
      columns.fe.laneRounds.length * 2
  have feAuxiliary :
      (KSplitNcFeRows.cost columns.fe).auxiliaryColumns =
        3 * feFrames := by
    rw [KSplitNcFeRows.auxiliary_count]
    have rowWidth :
        columns.fe.rowRounds.length * (3 * SumCheck.Fe.Drow input) =
          3 *
            (columns.fe.rowRounds.length * SumCheck.Fe.Drow input) := by
      calc
        columns.fe.rowRounds.length * (3 * SumCheck.Fe.Drow input) =
            (columns.fe.rowRounds.length * 3) *
              SumCheck.Fe.Drow input := by
                rw [Nat.mul_assoc]
        _ = (3 * columns.fe.rowRounds.length) *
              SumCheck.Fe.Drow input := by
                rw [Nat.mul_comm columns.fe.rowRounds.length 3]
        _ = 3 *
              (columns.fe.rowRounds.length * SumCheck.Fe.Drow input) := by
                rw [Nat.mul_assoc]
    rw [rowWidth]
    simp only [feFrames, Nat.mul_add]
    omega
  by_cases inFe : column < base + 3 * feFrames
  · have feColumn :
        column ∈ KFrames.frameColumns base feFrames := by
      rw [KFrames.frameColumns_mem_iff]
      omega
    rcases fe columns.fe base column feColumn with
      ⟨row, rowMember, mentioned⟩
    exact
      ⟨row, List.mem_append_left _ rowMember, mentioned⟩
  · have ncColumn :
        column ∈
          KFrames.frameColumns
            (KSplitNcBlockLaneRows.ncBase columns base)
            (columns.nc.rounds.length * 4) := by
      rw [KFrames.frameColumns_mem_iff]
      unfold KSplitNcBlockLaneRows.ncBase
      rw [feAuxiliary]
      omega
    rcases nc columns.nc
        (KSplitNcBlockLaneRows.ncBase columns base)
        column ncColumn with ⟨row, rowMember, mentioned⟩
    exact
      ⟨row, List.mem_append_right _ rowMember, mentioned⟩

end Nightstream.Implementation.R1CS.Canonical.KSplitNcBlockLaneCoverage
