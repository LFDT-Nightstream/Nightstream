import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ActivatedRawProgram
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsAllocationCoverage
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRawConservation

/-!
Contract: activation, footprint, and exact temporary placement for the
Lean-owned selected `nifsVerify` row program.

The intrinsic verifier occupies the leading temporary prefix.  One residual
per intrinsic row occupies the remaining suffix.  This module derives that
split from the declared call footprint and proves that the activated rows use
exactly those coordinates without overlap or hidden allocation.

The footprint alignment is a static representation obligation.  It contains
no verifier result, acceptance proposition, source-authority premise, Rust
measurement, or generated-row fact.

Emits constraints: two activated rows and one residual coordinate per
intrinsic row.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1200000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsActivatedProgram

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.RawAllocationCoverage
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile
open Nightstream.Implementation.R1CS.Canonical
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

/-- Intrinsic selected-verifier cost before activation. -/
def intrinsicCost
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) : Cost :=
  ConcreteNifsRawProgram.cost application profile frame

/-- Complete selected-verifier cost after activation lowering. -/
def cost
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) : Cost :=
  ActivatedRawProgram.cost (intrinsicCost application profile frame)

/-- The exact call footprint computed from the Lean-owned activated program. -/
def footprint
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) : CallFootprint where
  recurringRows := (cost application profile frame).recurringRows
  temporaries :=
    [auxiliaryLayout (cost application profile frame).auxiliaryColumns]

/-- Static equality between the selected vocabulary footprint and the
Lean-derived activated program for every physical occurrence shape.

The quantification prevents a caller from pricing one frame while emitting
another.  Its fields are resource equalities only. -/
structure FootprintAlignment
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application) : Prop where
  exact :
    ∀ {context : Schema (typeSystem Selected)}
      {runningRef : Ref (typeSystem Selected) context (.data .running)}
      {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
      {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
      (frame : FrameFor application
        (context := context) (runningRef := runningRef)
        (freshRef := freshRef) (proofRef := proofRef)),
      footprints.nifsVerify = footprint application profile frame

/-- The signature exposes exactly the Lean-derived activated footprint. -/
theorem selected_footprint_exact
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    (alignment : FootprintAlignment application profile)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    (signature Selected).callFootprint Call.nifsVerify =
      footprint application profile frame := by
  exact alignment.exact frame

/-- Activation residuals are the exact suffix after the raw verifier's
intrinsic allocation. -/
def residuals
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) : List ColumnId :=
  frame.temporaries.ids.drop
    (ConcreteNifsRawProgram.allocationWidth application profile frame)

/-- Exact activated raw row list. -/
def rawRows
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) : List Row :=
  ActivatedRawProgram.rawRows frame.active
    (ConcreteNifsRawProgram.rawRows application profile frame)
    (residuals application profile frame)

/-- Stable positional ownership of every activated selected-verifier row. -/
def rows
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) : List OwnedRow :=
  ActivatedRawProgram.rows frame.owner frame.active
    (ConcreteNifsRawProgram.rawRows application profile frame)
    (residuals application profile frame)

/-- The selected temporary receipt has exactly the activated auxiliary
coordinate count. -/
theorem temporaries_length
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    (alignment : FootprintAlignment application profile)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    frame.temporaries.ids.length =
      (cost application profile frame).auxiliaryColumns := by
  have layoutsEqual :
      ((signature Selected).callFootprint Call.nifsVerify).temporaries =
        (footprint application profile frame).temporaries :=
    congrArg CallFootprint.temporaries
      (selected_footprint_exact application profile alignment frame)
  rw [LayoutBundles.ids, List.length_map,
    LayoutBundles.columns_length, layoutsEqual]
  simp [footprint, auxiliaryLayout, ownedLayout]

/-- The selected footprint always has room for the intrinsic raw prefix. -/
theorem frameFits
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    (alignment : FootprintAlignment application profile)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    ConcreteNifsRawProgram.FrameFits application profile frame := by
  unfold ConcreteNifsRawProgram.FrameFits
  rw [ConcreteNifsRawProgram.allocationWidth_eq_cost,
    temporaries_length application profile alignment frame]
  unfold cost intrinsicCost
  rw [ActivatedRawProgram.cost_auxiliary]
  omega

/-- There is exactly one activation residual per intrinsic row. -/
theorem residuals_length
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    (alignment : FootprintAlignment application profile)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    (residuals application profile frame).length =
      (ConcreteNifsRawProgram.rawRows application profile frame).length := by
  unfold residuals
  rw [List.length_drop,
    temporaries_length application profile alignment frame,
    ConcreteNifsRawProgram.allocationWidth_eq_cost,
    ConcreteNifsRawProgram.rawRows_length]
  unfold cost intrinsicCost
  rw [ActivatedRawProgram.cost_auxiliary]
  omega

/-- The intrinsic allocation and activation suffix exhaust the declared
temporary receipt in order. -/
theorem allocation_append_residuals
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    (alignment : FootprintAlignment application profile)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    ConcreteNifsRawProgram.allocation application profile frame ++
        residuals application profile frame =
      frame.temporaries.ids := by
  rw [ConcreteNifsRawProgram.allocation_eq_temporaryPrefix
    application profile frame
      (frameFits application profile alignment frame)]
  exact List.take_append_drop _ _

/-- The activation residual suffix is duplicate-free. -/
theorem residuals_nodup
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    (residuals application profile frame).Nodup := by
  unfold residuals
  exact ((List.nodup_append.1 frame.allocationsNodup).2.1).drop

/-- Residuals cannot alias a visible input, output, activation, or constant
wire. -/
theorem residuals_disjoint_visible
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    IdsDisjoint (residuals application profile frame) frame.visibleIds := by
  intro column residualMember visibleMember
  exact frame.temporariesDisjointVisible column
    (List.mem_of_mem_drop residualMember) visibleMember

/-- Residuals cannot alias the intrinsic allocation prefix. -/
theorem residuals_disjoint_allocation
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    (alignment : FootprintAlignment application profile)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    IdsDisjoint
      (residuals application profile frame)
      (ConcreteNifsRawProgram.allocation application profile frame) := by
  have temporaryNodup :
      frame.temporaries.ids.Nodup :=
    (List.nodup_append.1 frame.allocationsNodup).2.1
  have combinedNodup :
      (ConcreteNifsRawProgram.allocation application profile frame ++
        residuals application profile frame).Nodup := by
    rw [allocation_append_residuals application profile alignment frame]
    exact temporaryNodup
  have cross := (List.nodup_append.1 combinedNodup).2.2
  intro column residualMember allocationMember
  exact cross column allocationMember column residualMember rfl

/-- Every residual is fresh from every intrinsic row dependency. -/
theorem residuals_fresh
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    (alignment : FootprintAlignment application profile)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    IdsDisjoint
      (residuals application profile frame)
      ((ConcreteNifsRawProgram.rawRows application profile frame).flatMap
        fun row => row.columnIds) := by
  intro column residualMember supportMember
  rcases List.mem_flatMap.1 supportMember with
    ⟨row, rowMember, columnMember⟩
  have allowed :=
    ConcreteNifsRawConservation.rawRows_supported_exact
      application profile frame row rowMember column columnMember
  rcases List.mem_append.1 allowed with visibleMember | allocationMember
  · exact residuals_disjoint_visible application profile frame
      column residualMember visibleMember
  · exact residuals_disjoint_allocation
      application profile alignment frame
      column residualMember allocationMember

/-- The activation coordinate is never one of its residual witnesses. -/
theorem active_not_residual
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    frame.active ∉ residuals application profile frame := by
  intro residualMember
  exact residuals_disjoint_visible application profile frame
    frame.active residualMember (by simp [CallFrame.visibleIds])

/-- Every activated row uses only visible coordinates or the exact complete
temporary receipt. -/
theorem rows_supported
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    (alignment : FootprintAlignment application profile)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (row : OwnedRow)
    (member : row ∈ rows application profile frame)
    (column : ColumnId)
    (columnMember : column ∈ row.columnIds) :
    column ∈ frame.visibleIds ++ frame.temporaries.ids := by
  have supported :=
    ActivatedRawProgram.rows_supported frame.owner frame.active
      (ConcreteNifsRawProgram.rawRows application profile frame)
      (residuals application profile frame)
      (frame.visibleIds ++
        ConcreteNifsRawProgram.allocation application profile frame)
      (ConcreteNifsRawConservation.rawRows_supported_exact
        application profile frame)
      row member column columnMember
  rcases List.mem_cons.1 supported with rfl | supported
  · exact List.mem_append_left _ (by simp [CallFrame.visibleIds])
  · rcases List.mem_append.1 supported with
      visibleOrAllocation | residualMember
    · rcases List.mem_append.1 visibleOrAllocation with
        visibleMember | allocationMember
      · exact List.mem_append_left _ visibleMember
      · rw [← allocation_append_residuals
          application profile alignment frame]
        exact List.mem_append_right frame.visibleIds
          (List.mem_append_left _ allocationMember)
    · rw [← allocation_append_residuals
        application profile alignment frame]
      exact List.mem_append_right frame.visibleIds
        (List.mem_append_right _ residualMember)

/-- The complete activated program covers every declared temporary
coordinate, both intrinsic and residual. -/
theorem allocation_coverage
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    (alignment : FootprintAlignment application profile)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    TypedRowsCover
      (rawRows application profile frame)
      frame.temporaries.ids := by
  rw [← allocation_append_residuals
    application profile alignment frame]
  exact ActivatedRawProgram.allocation_coverage frame.active
    (ConcreteNifsRawProgram.rawRows application profile frame)
    (residuals application profile frame)
    (ConcreteNifsRawProgram.allocation application profile frame)
    (residuals_length application profile alignment frame).symm
    (ConcreteNifsAllocationCoverage.allocation_used
      application profile frame)

/-- The emitted activated rows have the exact selected footprint row count. -/
theorem rows_length
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    (alignment : FootprintAlignment application profile)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    (rows application profile frame).length =
      ((signature Selected).callFootprint
        Call.nifsVerify).recurringRows := by
  unfold rows
  rw [ActivatedRawProgram.rows_length _ _ _ _
      (residuals_length application profile alignment frame).symm,
    ConcreteNifsRawProgram.rawRows_length]
  have exactFootprint :=
    selected_footprint_exact application profile alignment frame
  rw [exactFootprint]
  unfold footprint cost intrinsicCost
  rw [ActivatedRawProgram.cost_rows]

/-- All activated row identifiers belong to the exact call owner. -/
theorem rows_owned
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (row : OwnedRow)
    (member : row ∈ rows application profile frame) :
    row.id.owner = frame.owner :=
  ActivatedRawProgram.rows_owned _ _ _ _ _ member

/-- Activated row identifiers are positionally unique. -/
theorem rowIds_nodup
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    ((rows application profile frame).map fun row => row.id).Nodup :=
  ActivatedRawProgram.rowIds_nodup _ _ _ _

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsActivatedProgram
