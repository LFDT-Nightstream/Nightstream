import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSampler
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOutputRows
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiDecRows
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionAudit
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcPointRows
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsProofCanonicalityRows
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRunningAuthorityRows

/-!
Contract: the complete ungated deterministic row program for one selected
fixed-active NIFS occurrence.

The ordered program is assembled exclusively from Lean-owned component row
lists. Numeric rows cross the single call-frame column map once; the Phi81
action rows are already typed in that same namespace. Ten leading
temporaries own the five quadratic-extension claimed-chain values, including
the internal FE row/lane boundary.

This module owns construction, positional row ownership, and the exact raw
row/allocation receipt. It does not own activation, honest completion,
deterministic verifier refinement, paper events, or a `CallRecipe`.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1200000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRawProgram

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
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

/-- Ten caller-frame temporaries own the five `K` claimed-chain values. -/
def claimedValueCost : Cost where
  recurringRows := 0
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := 10

/-- First numeric source after the operational ΠCCS and fixed-active sampler
allocations. -/
def actionBase
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) : Nat :=
  ConcreteNifsOperationalSampler.samplerBase application profile frame +
    PiRlcCanonicalSamplerProgram.cost.auxiliaryColumns

/-- Translate a numeric component through the sole selected call-frame map. -/
def translate
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (source : List Nightstream.Implementation.R1CS.Row) :
    List Nightstream.Implementation.Lowering.Goldilocks.Row :=
  source.map
    (Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.row
      (columnMap frame))

/-- Numeric translation introduces no dependency outside the call's visible
coordinates and declared temporary bundle. -/
theorem translate_supported
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (source : List Nightstream.Implementation.R1CS.Row) :
    RawRowsSupportedBy
      (frame.visibleIds ++ frame.temporaries.ids)
      (translate application frame source) := by
  intro row rowMember column columnMember
  rcases List.mem_map.1 rowMember with
    ⟨numericRow, _, rowEqual⟩
  rw [← rowEqual] at columnMember
  have exactColumns :=
    NumericRowBridge.row_columnIds (columnMap frame) numericRow
  have mappedMember :
      column ∈
        (numericRow.a ++ numericRow.b ++ numericRow.c).map
          (fun sourceTerm => columnMap frame sourceTerm.1) :=
    (congrArg (fun columns : List ColumnId => column ∈ columns)
      exactColumns).mp columnMember
  rcases List.mem_map.1 mappedMember with ⟨term, _, rfl⟩
  exact columnMap_supported frame term.1

private theorem supported_append
    (allowed : List ColumnId)
    (left right : List Nightstream.Implementation.Lowering.Goldilocks.Row)
    (leftSupported : RawRowsSupportedBy allowed left)
    (rightSupported : RawRowsSupportedBy allowed right) :
    RawRowsSupportedBy allowed (left ++ right) := by
  intro row rowMember column columnMember
  rcases List.mem_append.1 rowMember with inLeft | inRight
  · exact leftSupported row inLeft column columnMember
  · exact rightSupported row inRight column columnMember

/-- Complete ungated selected-NIFS row list in verifier execution order. -/
def rawRows
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    List Nightstream.Implementation.Lowering.Goldilocks.Row :=
  ConcreteNifsProofCanonicalityRows.rows application profile frame ++
    translate application frame
      (ConcreteNifsRunningAuthorityRows.rows application profile frame) ++
    translate application frame
      (ConcreteNifsOperationalSampler.rows application profile frame) ++
    translate application frame
      (ConcreteNifsPiRlcPointRows.rows application profile frame) ++
    ConcreteNifsPiRlcActionRows.rows application profile frame
      (actionBase application profile frame) ++
    translate application frame
      (ConcreteNifsPiDecRows.rows application profile frame) ++
    translate application frame
      (ConcreteNifsOutputRows.rows application profile frame)

/-- Every dependency of the complete raw program is either visible before
the occurrence or belongs to the exact declared temporary bundle. -/
theorem rawRows_supported
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    RawRowsSupportedBy
      (frame.visibleIds ++ frame.temporaries.ids)
      (rawRows application profile frame) := by
  unfold rawRows
  apply supported_append
  · apply supported_append
    · apply supported_append
      · apply supported_append
        · apply supported_append
          · apply supported_append
            · intro row rowMember column columnMember
              exact List.mem_append_left _
                (ConcreteNifsProofCanonicalityRows.rows_supported
                  application profile frame row rowMember column columnMember)
            · exact translate_supported application frame _
          · exact translate_supported application frame _
        · exact translate_supported application frame _
      · exact ConcreteNifsPiRlcActionAudit.rows_supported
          application profile frame (actionBase application profile frame)
    · exact translate_supported application frame _
  · exact translate_supported application frame _

/-- Stable positional ownership for the complete raw program. -/
def rows
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    List OwnedRow :=
  DirectCalls.ownRows frame.owner (rawRows application profile frame)

/-- Receipt-folded intrinsic cost before activation. -/
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
  claimedValueCost +
    ConcreteNifsProofCanonicalityRows.cost +
    ConcreteNifsRunningAuthorityRows.cost
      shape publicRingColumns verifierRows +
    ConcreteNifsOperationalSampler.cost application profile frame +
    ConcreteNifsPiRlcPointRows.cost shape.rowVariables +
    ConcreteNifsPiRlcActionRows.cost
      shape publicRingColumns verifierRows +
    ConcreteNifsPiDecRows.cost shape publicRingColumns verifierRows +
    ConcreteNifsOutputRows.cost shape publicRingColumns verifierRows

/-- Exact number of raw auxiliary coordinates. -/
def allocationWidth
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) : Nat :=
  10 +
    (ConcreteNifsOperationalSampler.cost application profile frame).auxiliaryColumns +
    (ConcreteNifsPiRlcActionRows.cost
      shape publicRingColumns verifierRows).auxiliaryColumns

/-- Numeric allocation sources are one dense temporary interval. -/
def allocationSources
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) : List Nat :=
  (List.range (allocationWidth application profile frame)).map
    (temporarySource frame)

/-- Exact typed allocation in source order. -/
def allocation
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) : List ColumnId :=
  (allocationSources application profile frame).map (columnMap frame)

theorem rawRows_length
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    (rawRows application profile frame).length =
      (cost application profile frame).recurringRows := by
  unfold rawRows translate cost claimedValueCost
  simp only [List.length_append, List.length_map, Cost.add_recurringRows]
  rw [ConcreteNifsProofCanonicalityRows.cost_rows,
    ConcreteNifsRunningAuthorityRows.cost_rows,
    ConcreteNifsOperationalSampler.rows_cost,
    ← ConcreteNifsPiRlcPointRows.cost_rows_eq,
    ConcreteNifsPiRlcActionRows.rows_length,
    ConcreteNifsPiDecRows.cost_rows,
    ← ConcreteNifsOutputRows.cost_rows_eq]
  omega

theorem rows_length
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    (rows application profile frame).length =
      (cost application profile frame).recurringRows := by
  simpa [rows] using rawRows_length application profile frame

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
    (row : OwnedRow) (member : row ∈ rows application profile frame) :
    row.id.owner = frame.owner :=
  DirectCalls.ownRows_owner _ _ _ member

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
  DirectCalls.ownRows_ids_nodup _ _

theorem rows_supported
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
    (member : row ∈ rows application profile frame)
    (column : ColumnId)
    (columnMember : column ∈ row.columnIds) :
    column ∈ frame.visibleIds ++ frame.temporaries.ids :=
  DirectCalls.ownRows_supported frame.owner
    (rawRows application profile frame)
    (frame.visibleIds ++ frame.temporaries.ids)
    (rawRows_supported application profile frame)
    row member column columnMember

theorem allocationWidth_eq_cost
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    allocationWidth application profile frame =
      (cost application profile frame).auxiliaryColumns := by
  simp [allocationWidth, cost, claimedValueCost,
    ConcreteNifsProofCanonicalityRows.cost,
    ConcreteNifsRunningAuthorityRows.cost,
    ConcreteNifsPiRlcPointRows.cost,
    ConcreteNifsPiDecRows.cost,
    ConcreteNifsOutputRows.cost]

theorem allocation_length
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    (allocation application profile frame).length =
      (cost application profile frame).auxiliaryColumns := by
  rw [allocation, List.length_map, allocationSources, List.length_map,
    List.length_range, allocationWidth_eq_cost]

/-- The selected call frame has room for the raw verifier's derived temporary
prefix.  Activation owns the disjoint suffix, so raw-program fit is a bound,
not equality with the complete call-frame bundle. -/
def FrameFits
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) : Prop :=
  allocationWidth application profile frame <=
    frame.temporaries.ids.length

/-- Under the derived footprint, the raw allocation is exactly the ordered
temporary prefix.  In particular, `columnMap` never reaches its fallback on
an allocated source, while later activation may use the untouched suffix. -/
theorem allocation_eq_temporaryPrefix
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (fits : FrameFits application profile frame) :
    allocation application profile frame =
      frame.temporaries.ids.take
        (allocationWidth application profile frame) := by
  apply List.ext_getElem
  · simp [allocation, allocationSources, Nat.min_eq_left fits]
  · intro index allocationBound temporaryBound
    simp only [allocation, allocationSources, List.getElem_map,
      List.getElem_range, List.getElem_take]
    have indexLtWidth :
        index < allocationWidth application profile frame := by
      simpa [allocation, allocationSources] using allocationBound
    exact columnMap_temporarySource frame (Nat.lt_of_lt_of_le indexLtWidth fits)

/-- The exact raw allocation contains no duplicate physical identity. -/
theorem allocation_nodup
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (fits : FrameFits application profile frame) :
    (allocation application profile frame).Nodup := by
  rw [allocation_eq_temporaryPrefix application profile frame fits]
  exact ((List.nodup_append.1 frame.allocationsNodup).2.1).take

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRawProgram
