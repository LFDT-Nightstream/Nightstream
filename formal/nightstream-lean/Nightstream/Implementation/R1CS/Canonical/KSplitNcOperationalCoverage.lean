import Nightstream.Implementation.R1CS.Canonical.KSplitNcBlockLaneCoverage
import Nightstream.Implementation.R1CS.Canonical.KSplitNcEndpointCoverage
import Nightstream.Implementation.R1CS.Canonical.KSplitNcOperationalRows
import Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscriptCallOrder

/-!
Contract: every auxiliary declared by the selected operational Split-NC
program is used by an emitted row.

The proof follows the actual three-part placement: transcript permutations,
numeric FE/NC chains, and verifier-owned endpoints.  It does not infer use
from equal row or column counts.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1000000

namespace Nightstream.Implementation.R1CS.Canonical.KSplitNcOperationalCoverage

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.AllocationCoverage
open Nightstream.Implementation.R1CS.Canonical.KSplitNcBlockLaneCoverage
open Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexPhysical
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

private theorem dense_mem_iff (base width column : Nat) :
    column ∈ (List.range width).map (fun offset => base + offset) ↔
      base ≤ column ∧ column < base + width := by
  constructor
  · intro member
    rcases List.mem_map.1 member with ⟨offset, inRange, rfl⟩
    exact ⟨by omega, by
      have bound := List.mem_range.1 inRange
      omega⟩
  · intro bounds
    exact List.mem_map.2
      ⟨column - base, List.mem_range.2 (by omega), by omega⟩

private theorem liftGroup
    {groups : List (List Row)} {group : List Row} {columns : List Nat}
    (groupMember : group ∈ groups)
    (covered : RowsCover group columns) :
    RowsCover groups.flatten columns := by
  intro column member
  rcases covered column member with ⟨row, rowMember, mentioned⟩
  exact
    ⟨row, List.mem_flatten.2 ⟨group, groupMember, rowMember⟩, mentioned⟩

private theorem transcript
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (constants : Poseidon2Schedule.Constants)
    (input : KSplitNcOperationalRows.Input polynomialInput domains) :
    RowsCover
      (KSplitNcOperationalRows.transcriptRows constants input)
      (SymbolicDuplexPhysical.temporaryColumns
        input.transcript.transcriptBase
        (KSplitNcTranscript.replay input.transcript).afterOutput.entries.length) := by
  intro column member
  rcases temporaryColumns_written_of_calls
      input.transcript.transcriptBase constants
      (KSplitNcTranscript.outputBuilder input.transcript)
      (KSplitNcTranscriptCallOrder.outputBuilder input.transcript)
      column
      (by simpa only [KSplitNcTranscript.replay] using member) with
    ⟨row, rowMember, mentioned⟩
  exact ⟨row, rowMember, Or.inr (Or.inr mentioned)⟩

/-- Exact converse-to-conservation theorem for the complete operational
Split-NC row program. -/
theorem rows
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (constants : Poseidon2Schedule.Constants)
    (input : KSplitNcOperationalRows.Input polynomialInput domains) :
    RowsCover
      (KSplitNcOperationalRows.rows constants input)
      (KSplitNcOperationalRows.columns input) := by
  intro column member
  unfold KSplitNcOperationalRows.columns at member
  rw [dense_mem_iff] at member
  by_cases inTranscript :
      column < KSplitNcOperationalRows.numericBase input
  · have localColumn :
        column ∈
          SymbolicDuplexPhysical.temporaryColumns
            input.transcript.transcriptBase
            (KSplitNcTranscript.replay
              input.transcript).afterOutput.entries.length := by
      rw [temporaryColumns_mem_iff]
      constructor
      · exact member.1
      · simpa only [KSplitNcOperationalRows.numericBase] using inTranscript
    unfold KSplitNcOperationalRows.rows
    exact liftGroup
      (by simp [KSplitNcOperationalRows.rowGroups])
      (transcript constants input) column localColumn
  · by_cases inNumeric :
      column < KSplitNcOperationalRows.endpointBase input
    · let numericColumns :=
        KSplitNcTranscript.numericColumns input.transcript
      let frameCount :=
        numericColumns.fe.rowRounds.length *
            SumCheck.Fe.Drow polynomialInput +
          numericColumns.fe.laneRounds.length * 2 +
          numericColumns.nc.rounds.length * 4
      have localColumn :
          column ∈
            KFrames.frameColumns
              (KSplitNcOperationalRows.numericBase input)
              frameCount := by
        rw [KFrames.frameColumns_mem_iff]
        constructor
        · exact Nat.le_of_not_gt inTranscript
        · have upper := inNumeric
          have auxiliary :
              (KSplitNcBlockLaneRows.cost numericColumns).auxiliaryColumns =
                3 * frameCount := by
            rw [KSplitNcBlockLaneRows.auxiliary_count]
            have rowWidth :
                numericColumns.fe.rowRounds.length *
                    (3 * SumCheck.Fe.Drow polynomialInput) =
                  3 *
                    (numericColumns.fe.rowRounds.length *
                      SumCheck.Fe.Drow polynomialInput) := by
              calc
                numericColumns.fe.rowRounds.length *
                      (3 * SumCheck.Fe.Drow polynomialInput) =
                    (numericColumns.fe.rowRounds.length * 3) *
                      SumCheck.Fe.Drow polynomialInput := by
                        rw [Nat.mul_assoc]
                _ = (3 * numericColumns.fe.rowRounds.length) *
                      SumCheck.Fe.Drow polynomialInput := by
                        rw [Nat.mul_comm
                          numericColumns.fe.rowRounds.length 3]
                _ = 3 *
                      (numericColumns.fe.rowRounds.length *
                        SumCheck.Fe.Drow polynomialInput) := by
                        rw [Nat.mul_assoc]
            rw [rowWidth]
            unfold frameCount
            omega
          unfold KSplitNcOperationalRows.endpointBase at upper
          rw [auxiliary] at upper
          exact upper
      have covered :=
        KSplitNcBlockLaneCoverage.rows
          (KSplitNcTranscript.numericColumns input.transcript)
          (KSplitNcOperationalRows.numericBase input)
      unfold numericColumns frameCount at localColumn
      unfold KSplitNcOperationalRows.rows
      exact liftGroup
        (by
          simp [KSplitNcOperationalRows.rowGroups,
            KSplitNcOperationalRows.numericRows])
        covered column localColumn
    · have localColumn :
          column ∈
            KSplitNcEndpoints.columns
              (KSplitNcOperationalRows.endpointInput input) := by
        unfold KSplitNcEndpoints.columns
        rw [dense_mem_iff]
        constructor
        · change KSplitNcOperationalRows.endpointBase input ≤ column
          exact Nat.le_of_not_gt inNumeric
        · have endEq :
              input.transcript.transcriptBase +
                  KSplitNcOperationalRows.allocationWidth input =
                KSplitNcOperationalRows.endpointBase input +
                  KSplitNcEndpoints.allocationWidth
                    (KSplitNcOperationalRows.endpointInput input) := by
            unfold KSplitNcOperationalRows.allocationWidth
              KSplitNcOperationalRows.endpointBase
              KSplitNcOperationalRows.numericBase
            omega
          rw [endEq] at member
          exact member.2
      unfold KSplitNcOperationalRows.rows
      exact liftGroup
        (by
          simp [KSplitNcOperationalRows.rowGroups,
            KSplitNcOperationalRows.endpointRows])
        (KSplitNcEndpointCoverage.endpoints
          (KSplitNcOperationalRows.endpointInput input))
        column localColumn

end Nightstream.Implementation.R1CS.Canonical.KSplitNcOperationalCoverage
