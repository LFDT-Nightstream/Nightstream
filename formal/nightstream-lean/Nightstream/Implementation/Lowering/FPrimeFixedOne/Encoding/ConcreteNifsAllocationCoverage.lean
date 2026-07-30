import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRawProgram
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.RawAllocationCoverage
import Nightstream.Implementation.R1CS.Canonical.KSplitNcClaimedEndpointCoverage
import Nightstream.Implementation.R1CS.Canonical.KSplitNcOperationalCoverage
import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerAllocation
import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerCoverage

/-!
Contract: every auxiliary declared by the complete selected raw `nifsVerify`
program is mentioned by an emitted row.

The proof follows the Lean-owned component allocations.  It does not infer
coverage from equal row/column counts or from the dense outer frame alone.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1200000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsAllocationCoverage

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.RawAllocationCoverage
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.AllocationCoverage
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

private abbrev TranscriptState := Poseidon2Duplex.State

section SelectedFrame

variable {shape : SemanticShape}
variable {Digest AppState Witness Encoded RunningWitness FreshWitness : Type}
variable [DecidableEq AppState] [DecidableEq Encoded]
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}
variable {keys : Fin 1 →
  SelectedKey shape TranscriptState publicRingColumns publicFits verifierRows}
variable {defaultRunning :
  SelectedRunning shape publicRingColumns publicFits verifierRows}
variable {machine :
  Machine
    (SelectedKey shape TranscriptState publicRingColumns publicFits
      verifierRows)
    Digest AppState Witness
    (SelectedRunning shape publicRingColumns publicFits verifierRows)
    (SelectedFresh shape publicRingColumns publicFits verifierRows)
    Encoded 1}
variable {terminalRelations :
  TerminalRelations
    (SelectedKey shape TranscriptState publicRingColumns publicFits
      verifierRows)
    (SelectedRunning shape publicRingColumns publicFits verifierRows)
    RunningWitness
    (SelectedFresh shape publicRingColumns publicFits verifierRows)
    FreshWitness 1}
variable {terminalChecks :
  Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
    terminalRelations}
variable {widths : Widths} {footprints : Footprints}

local notation "Selected" =>
  ConcreteNifsParameters.selected keys defaultRunning machine
    terminalRelations terminalChecks widths footprints

private abbrev FamilyFor
    (application : Poseidon23ApplicationProfile Selected) :=
  application.family Selected

private abbrev FrameFor
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)} :=
  CallFrame (signature := signature Selected)
    (FamilyFor application) Call.nifsVerify
    (Refs.cons runningRef
      (Refs.cons freshRef (Refs.cons proofRef .nil)))

/-- The canonical claimed-value list is exactly the first ten temporary
sources of the selected call frame. -/
theorem endpointColumns_eq
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    KSplitNcClaimedEndpointCoverage.columns
        (ConcreteNifsOperationalOccurrence.input application profile frame) =
      (List.range 10).map (temporarySource frame) := by
  simp [KSplitNcClaimedEndpointCoverage.columns,
    ConcreteNifsOperationalOccurrence.input,
    ConcreteNifsOperationalOccurrence.transcriptInput,
    KSplitNcTranscript.numericColumns,
    ConcreteNifsOperationalOccurrence.temporaryK,
    KFixedPhaseEndpointCoverage.columns,
    KFixedPhaseSemanticOccurrence.carried, List.range_succ]

/-- The operational ΠCCS rows cover the enclosing frame's five claimed
quadratic-extension values. -/
theorem operationalEndpoints
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    NumericRowsCover
      (ConcreteNifsOperationalOccurrence.rows application profile frame)
      ((List.range 10).map (temporarySource frame)) := by
  apply RawAllocationCoverage.of_rows_cover
  rw [← endpointColumns_eq application profile frame]
  exact KSplitNcClaimedEndpointCoverage.rows profile.constants
    (ConcreteNifsOperationalOccurrence.input application profile frame)

/-- The operational ΠCCS and fixed-active sampler rows cover their exact
component allocation. -/
theorem operationalSampler
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    NumericRowsCover
      (ConcreteNifsOperationalSampler.rows application profile frame)
      (ConcreteNifsOperationalSampler.allocation
        application profile frame) := by
  apply RawAllocationCoverage.of_rows_cover
  intro column member
  unfold ConcreteNifsOperationalSampler.allocation at member
  rcases List.mem_append.1 member with inOperational | inSampler
  · rcases KSplitNcOperationalCoverage.rows profile.constants
        (ConcreteNifsOperationalOccurrence.input application profile frame)
        column inOperational with
      ⟨row, rowMember, mentioned⟩
    refine ⟨row, ?_, mentioned⟩
    unfold ConcreteNifsOperationalSampler.rows
    exact List.mem_append_left _
      (List.mem_append_left _ rowMember)
  · rcases PiRlcCanonicalSamplerCoverage.samplerProgram
        (ConcreteNifsOperationalSampler.samplerBase
          application profile frame)
        profile.constants
        (ConcreteNifsOperationalSampler.samplerLanes
          application profile frame)
        column inSampler with
      ⟨row, rowMember, mentioned⟩
    refine ⟨row, ?_, mentioned⟩
    unfold ConcreteNifsOperationalSampler.rows
    exact List.mem_append_left _
      (List.mem_append_right _ rowMember)

private theorem dense_mem_iff (base width column : Nat) :
    column ∈ (List.range width).map (fun offset => base + offset) ↔
      base ≤ column ∧ column < base + width := by
  constructor
  · intro member
    rcases List.mem_map.1 member with ⟨offset, inRange, rfl⟩
    exact ⟨by omega, by
      have offsetLt := List.mem_range.1 inRange
      omega⟩
  · intro bounds
    exact List.mem_map.2
      ⟨column - base, List.mem_range.2 (by omega), by omega⟩

private theorem samplerBase_eq
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    ConcreteNifsOperationalSampler.samplerBase application profile frame =
      temporarySource frame 10 +
        KSplitNcOperationalRows.allocationWidth
          (ConcreteNifsOperationalOccurrence.input
            application profile frame) := by
  rfl

/-- The operational ΠCCS and fixed-active sampler allocation is one exact
dense interval between the claimed endpoints and the action products. -/
theorem operationalSamplerAllocation_mem_iff
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (column : Nat) :
    column ∈ ConcreteNifsOperationalSampler.allocation
        application profile frame ↔
      temporarySource frame 10 ≤ column ∧
        column <
          ConcreteNifsRawProgram.actionBase
            application profile frame := by
  unfold ConcreteNifsOperationalSampler.allocation
  rw [List.mem_append]
  constructor
  · intro member
    rcases member with inOperational | inSampler
    · have bounds :
          temporarySource frame 10 ≤ column ∧
            column <
              ConcreteNifsOperationalSampler.samplerBase
                application profile frame := by
        rw [samplerBase_eq application profile frame]
        apply
          (dense_mem_iff (temporarySource frame 10)
            (KSplitNcOperationalRows.allocationWidth
              (ConcreteNifsOperationalOccurrence.input
                application profile frame)) column).1
        simpa [KSplitNcOperationalRows.columns,
          ConcreteNifsOperationalOccurrence.input,
          ConcreteNifsOperationalOccurrence.transcriptInput] using
          inOperational
      refine ⟨bounds.1, ?_⟩
      unfold ConcreteNifsRawProgram.actionBase
      omega
    · have bounds :=
        (PiRlcCanonicalSamplerAllocation.allocation_mem_iff
          (ConcreteNifsOperationalSampler.samplerBase
            application profile frame) column).1 inSampler
      refine ⟨?_, ?_⟩
      · rw [samplerBase_eq application profile frame] at bounds
        omega
      · unfold ConcreteNifsRawProgram.actionBase
        exact bounds.2
  · intro bounds
    by_cases beforeSampler :
        column <
          ConcreteNifsOperationalSampler.samplerBase
            application profile frame
    · left
      have dense :
          column ∈
            (List.range
              (KSplitNcOperationalRows.allocationWidth
                (ConcreteNifsOperationalOccurrence.input
                  application profile frame))).map
              (fun offset => temporarySource frame 10 + offset) := by
        apply
          (dense_mem_iff (temporarySource frame 10)
            (KSplitNcOperationalRows.allocationWidth
              (ConcreteNifsOperationalOccurrence.input
                application profile frame)) column).2
        rw [samplerBase_eq application profile frame] at beforeSampler
        exact ⟨bounds.1, beforeSampler⟩
      simpa [KSplitNcOperationalRows.columns,
        ConcreteNifsOperationalOccurrence.input,
        ConcreteNifsOperationalOccurrence.transcriptInput] using dense
    · right
      apply
        (PiRlcCanonicalSamplerAllocation.allocation_mem_iff
          (ConcreteNifsOperationalSampler.samplerBase
            application profile frame) column).2
      constructor
      · omega
      · simpa [ConcreteNifsRawProgram.actionBase] using bounds.2

private theorem operationalSamplerEndpoints
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    NumericRowsCover
      (ConcreteNifsOperationalSampler.rows application profile frame)
      ((List.range 10).map (temporarySource frame)) := by
  intro column member
  rcases operationalEndpoints application profile frame column member with
    ⟨row, rowMember, mentioned⟩
  refine ⟨row, ?_, mentioned⟩
  unfold ConcreteNifsOperationalSampler.rows
  exact List.mem_append_left _
    (List.mem_append_left _ rowMember)

theorem actionBase_eq_temporarySource
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    ConcreteNifsRawProgram.actionBase application profile frame =
      temporarySource frame
        (10 +
          (ConcreteNifsOperationalSampler.cost
            application profile frame).auxiliaryColumns) := by
  unfold ConcreteNifsRawProgram.actionBase
  rw [samplerBase_eq application profile frame]
  unfold ConcreteNifsOperationalSampler.cost
    ConcreteNifsOperationalSampler.challengeCost
  simp only [Cost.add_auxiliaryColumns, Nat.add_zero]
  rw [← KSplitNcOperationalRows.allocationWidth_eq_cost]
  unfold temporarySource
  omega

private theorem operationalTranslated_mem_raw
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (row : Nightstream.Implementation.Lowering.Goldilocks.Row)
    (member :
      row ∈ ConcreteNifsRawProgram.translate application frame
        (ConcreteNifsOperationalSampler.rows
          application profile frame)) :
    row ∈ ConcreteNifsRawProgram.rawRows application profile frame := by
  unfold ConcreteNifsRawProgram.rawRows
  exact List.mem_append_left _
    (List.mem_append_left _
      (List.mem_append_left _
        (List.mem_append_left _
          (List.mem_append_right _ member))))

private theorem action_mem_raw
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (row : Nightstream.Implementation.Lowering.Goldilocks.Row)
    (member :
      row ∈ ConcreteNifsPiRlcActionRows.rows
        application profile frame
          (ConcreteNifsRawProgram.actionBase application profile frame)) :
    row ∈ ConcreteNifsRawProgram.rawRows application profile frame := by
  unfold ConcreteNifsRawProgram.rawRows
  exact List.mem_append_left _
    (List.mem_append_left _
      (List.mem_append_right _ member))

/-- **Whole selected-NIFS converse conservation.** Every auxiliary declared
by the complete raw occurrence is mentioned by an emitted row.  In particular,
the dense outer allocation cannot hide an omitted endpoint, sampler cell, or
Phi81 action product. -/
theorem allocation_used
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    TypedRowsCover
      (ConcreteNifsRawProgram.rawRows application profile frame)
      (ConcreteNifsRawProgram.allocation application profile frame) := by
  intro column member
  unfold ConcreteNifsRawProgram.allocation at member
  rcases List.mem_map.1 member with
    ⟨sourceColumn, sourceMember, rfl⟩
  unfold ConcreteNifsRawProgram.allocationSources at sourceMember
  rcases List.mem_map.1 sourceMember with
    ⟨index, indexMember, rfl⟩
  have indexLt := List.mem_range.1 indexMember
  let samplerWidth :=
    (ConcreteNifsOperationalSampler.cost
      application profile frame).auxiliaryColumns
  by_cases inEndpoints : index < 10
  · have sourceInEndpoints :
        temporarySource frame index ∈
          (List.range 10).map (temporarySource frame) :=
      List.mem_map.2
        ⟨index, List.mem_range.2 inEndpoints, rfl⟩
    have typed :=
      RawAllocationCoverage.translate
        (columnMap frame)
        (ConcreteNifsOperationalSampler.rows application profile frame)
        ((List.range 10).map (temporarySource frame))
        (operationalSamplerEndpoints application profile frame)
    have typedMember :
        columnMap frame (temporarySource frame index) ∈
          ((List.range 10).map (temporarySource frame)).map
            (columnMap frame) :=
      List.mem_map.2
        ⟨temporarySource frame index, sourceInEndpoints, rfl⟩
    rcases typed _ typedMember with
      ⟨row, rowMember, mentioned⟩
    exact
      ⟨row,
        operationalTranslated_mem_raw
          application profile frame row rowMember,
        mentioned⟩
  · by_cases inSampler : index < 10 + samplerWidth
    · have sourceInSampler :
          temporarySource frame index ∈
            ConcreteNifsOperationalSampler.allocation
              application profile frame := by
        apply
          (operationalSamplerAllocation_mem_iff
            application profile frame _).2
        constructor
        · unfold temporarySource
          omega
        · rw [actionBase_eq_temporarySource
            application profile frame]
          unfold temporarySource
          dsimp [samplerWidth] at inSampler ⊢
          omega
      have typed :=
        RawAllocationCoverage.translate
          (columnMap frame)
          (ConcreteNifsOperationalSampler.rows application profile frame)
          (ConcreteNifsOperationalSampler.allocation
            application profile frame)
          (operationalSampler application profile frame)
      have typedMember :
          columnMap frame (temporarySource frame index) ∈
            (ConcreteNifsOperationalSampler.allocation
              application profile frame).map (columnMap frame) :=
        List.mem_map.2
          ⟨temporarySource frame index, sourceInSampler, rfl⟩
      rcases typed _ typedMember with
        ⟨row, rowMember, mentioned⟩
      exact
        ⟨row,
          operationalTranslated_mem_raw
            application profile frame row rowMember,
          mentioned⟩
    · let offset := index - (10 + samplerWidth)
      have offsetLt :
          offset <
            (ConcreteNifsPiRlcActionRows.cost
              shape publicRingColumns verifierRows).auxiliaryColumns := by
        unfold ConcreteNifsRawProgram.allocationWidth at indexLt
        dsimp [offset, samplerWidth]
        omega
      have actionColumn :=
        ConcreteNifsPiRlcActionAudit.dense_column_mem
          application profile frame
          (ConcreteNifsRawProgram.actionBase application profile frame)
          offset offsetLt
      rcases ConcreteNifsPiRlcActionAudit.columns_written
          application profile frame
          (ConcreteNifsRawProgram.actionBase application profile frame)
          _ actionColumn with
        ⟨row, rowMember, mentioned⟩
      have sourceEq :
          temporarySource frame index =
            ConcreteNifsRawProgram.actionBase application profile frame +
              offset := by
        rw [actionBase_eq_temporarySource
          application profile frame]
        unfold temporarySource
        dsimp [offset, samplerWidth]
        omega
      rw [sourceEq]
      exact
        ⟨row, action_mem_raw application profile frame row rowMember,
          mentioned⟩

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsAllocationCoverage
