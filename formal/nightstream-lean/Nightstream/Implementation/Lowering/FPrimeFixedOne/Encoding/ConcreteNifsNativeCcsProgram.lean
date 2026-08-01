import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsVerifyCallRecipe
import Nightstream.Implementation.Lowering.Goldilocks.NativeCcsProgram

/-!
Contract: native CCS selection for the complete selected `nifsVerify`
occurrence.

Assurance tier: model-level.

Owns:
- the intrinsic selected-verifier rows, with the recursive branch activation
  placed in the fourth CCS matrix;
- the exact intrinsic temporary prefix, with no activation residual suffix;
- active soundness, active honest completeness, inactive satisfiability,
  positional ownership, support, and exact row/allocation counts.

Does not own: the complete Step postcompiler, a proof-free manifest, Rust
matrix emission, or a deployment application.

Emits constraints: one native CCS row per intrinsic selected-verifier row and
no activation residual rows or columns.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2200000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsProgram

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsProgram
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsSelector
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

private abbrev TranscriptState := Poseidon2Duplex.State
private abbrev Field := Nightstream.SuperNeo.Concrete.F

private theorem cost_ext
    {left right : Cost}
    (rows : left.recurringRows = right.recurringRows)
    (committed : left.committedColumns = right.committedColumns)
    (publicEq : left.publicColumns = right.publicColumns)
    (auxiliary : left.auxiliaryColumns = right.auxiliaryColumns) :
    left = right := by
  cases left
  cases right
  simp_all

private theorem columnCost_append
    (left right : List OwnedColumn) :
    columnCost (left ++ right) = columnCost left + columnCost right := by
  unfold columnCost
  rw [List.map_append, Cost.sum_append]

private theorem rowCost_eq_length (rows : List OwnedRow) :
    rowCost rows = ⟨rows.length, 0, 0, 0⟩ := by
  induction rows with
  | nil =>
      rfl
  | cons _ tail inductionHypothesis =>
      change Cost.oneRow + rowCost tail =
        ⟨tail.length + 1, 0, 0, 0⟩
      rw [inductionHypothesis]
      apply cost_ext <;> simp [Cost.oneRow] <;> omega

private theorem columnCost_auxiliary
    (columns : List OwnedColumn)
    (auxiliary :
      ∀ column, column ∈ columns →
        column.ownership = .auxiliaryColumn) :
    columnCost columns = ⟨0, 0, 0, columns.length⟩ := by
  induction columns with
  | nil =>
      rfl
  | cons head tail inductionHypothesis =>
      have headAuxiliary :=
        auxiliary head List.mem_cons_self
      have tailAuxiliary :
          ∀ column, column ∈ tail →
            column.ownership = .auxiliaryColumn := by
        intro column member
        exact auxiliary column (List.mem_cons_of_mem head member)
      unfold columnCost
      simp only [List.map_cons, Cost.sum]
      rw [headAuxiliary]
      change
        Cost.oneColumn .auxiliaryColumn + columnCost tail =
          ⟨0, 0, 0, (head :: tail).length⟩
      rw [inductionHypothesis tailAuxiliary]
      apply cost_ext <;>
        simp [Cost.oneColumn] <;> omega

private theorem columnBundle_cost
    {layout : Layout}
    (bundle : ColumnBundle layout) :
    columnCost bundle.columns = layout.cost := by
  unfold columnCost Layout.cost
  rw [← bundle.ownerships_exact]
  simp only [List.map_map, Function.comp_def]

private theorem schemaBundles_cost
    {types : TypeSystem}
    {schema : Schema types}
    (bundles : SchemaBundles schema) :
    columnCost bundles.columns = schema.cost := by
  induction bundles with
  | nil =>
      rfl
  | @cons port tail head rest inductionHypothesis =>
      simp only [SchemaBundles.columns, SchemaBundles.portColumns,
        List.flatten_cons, Schema.cost, List.map_cons, Cost.sum]
      rw [columnCost_append, columnBundle_cost]
      have restCost :
          columnCost rest.portColumns.flatten = tail.cost := by
        simpa only [SchemaBundles.columns] using inductionHypothesis
      rw [restCost]
      rfl

private theorem layoutBundles_ownership_member
    {layouts : List Layout}
    (bundles : LayoutBundles layouts)
    (column : OwnedColumn)
    (member : column ∈ bundles.columns) :
    column.ownership ∈ layouts.flatMap Layout.owners := by
  induction bundles with
  | nil =>
      simp [LayoutBundles.columns, LayoutBundles.bundleColumns] at member
  | @cons layout tail head rest inductionHypothesis =>
      simp only [LayoutBundles.columns, LayoutBundles.bundleColumns,
        List.flatten_cons, List.mem_append] at member
      simp only [List.flatMap_cons, List.mem_append]
      rcases member with headMember | restMember
      · left
        have mapped :
            column.ownership ∈
              head.columns.map (fun item => item.ownership) :=
          List.mem_map.mpr ⟨column, headMember, rfl⟩
        simpa [head.ownerships_exact] using mapped
      · right
        exact inductionHypothesis restMember

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

/-- The native program allocates only the intrinsic temporary prefix. -/
def temporaryAllocations
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    List OwnedColumn :=
  frame.temporaries.columns.take
    (ConcreteNifsRawProgram.allocationWidth application profile frame)

/-- Exact intrinsic call allocation. The removed suffix contains only the old
R1CS activation residuals. -/
def allocations
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    List OwnedColumn :=
  frame.outputs.columns ++ temporaryAllocations application profile frame

theorem allocations_nodup
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    ((allocations application profile frame).map fun column => column.id
      ).Nodup := by
  have allNodup :
      (frame.outputs.ids ++ frame.temporaries.ids).Nodup := by
    simpa [CallFrame.allocations, SchemaBundles.ids, LayoutBundles.ids,
      List.map_append] using frame.allocationsNodup
  unfold allocations temporaryAllocations
  rw [List.map_append, List.map_take]
  change
    (frame.outputs.ids ++
      frame.temporaries.ids.take
        (ConcreteNifsRawProgram.allocationWidth application profile frame)
      ).Nodup
  exact List.nodup_append.2
    ⟨(List.nodup_append.1 allNodup).1,
      ((List.nodup_append.1 allNodup).2.1.take),
      by
        intro left leftMember right rightMember equal
        exact (List.nodup_append.1 allNodup).2.2
          left leftMember right (List.mem_of_mem_take rightMember) equal⟩

private theorem allocations_owned
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (column : OwnedColumn)
    (member : column ∈ allocations application profile frame) :
    column.id.owner = frame.owner := by
  apply frame.allocationsOwned column
  rcases List.mem_append.1 member with outputMember | temporaryMember
  · exact List.mem_append_left _ outputMember
  · exact List.mem_append_right _
      (List.mem_of_mem_take temporaryMember)

/-- Intrinsic R1CS receipt retained as the source image of the native CCS
row. It is not emitted as an activated R1CS receipt. -/
def sourceReceipt
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    InstructionReceipt where
  owner := frame.owner
  kind := .call
  allocations := allocations application profile frame
  rows := ConcreteNifsRawProgram.rows application profile frame
  allocationsOwned := allocations_owned application profile frame
  rowsOwned := ConcreteNifsRawProgram.rows_owned application profile frame

/-- The selected receipt uses the existing recursive-branch activation as the
fourth CCS matrix input. -/
def selectedReceipt
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    SelectedReceipt where
  receipt := sourceReceipt application profile frame
  selector := frame.active

theorem sourceReceipt_columnIds_nodup
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    (sourceReceipt application profile frame).columnIds.Nodup :=
  allocations_nodup application profile frame

theorem sourceReceipt_rowIds_nodup
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    (sourceReceipt application profile frame).rowIds.Nodup :=
  ConcreteNifsRawProgram.rowIds_nodup application profile frame

@[simp] theorem selectedReceipt_rows
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    (selectedReceipt application profile frame).rows =
      select frame.active
        (ConcreteNifsRawProgram.rows application profile frame) :=
  rfl

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
    ((selectedReceipt application profile frame).rows).length =
      (ConcreteNifsRawProgram.cost application profile frame).recurringRows := by
  rw [SelectedReceipt.rows_length]
  exact ConcreteNifsRawProgram.rows_length application profile frame

private theorem outputCost_exact
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    physicalCost frame.outputs.columns [] =
      ((signature Selected).callOutputs Call.nifsVerify).cost := by
  unfold physicalCost
  rw [schemaBundles_cost, rowCost_eq_length]
  apply cost_ext <;> simp [Cost.zero]

/-- Every declared temporary of the selected NIFS call is auxiliary. This
classifies both the intrinsic prefix retained by native CCS and the legacy
activation-residual suffix that native CCS removes. -/
theorem temporaryColumn_auxiliary
    (application : Poseidon23ApplicationProfile Selected)
    (certificate :
      ConcreteNifsVerifyCallRecipe.Certification application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (column : OwnedColumn)
    (member :
      column ∈ frame.temporaries.columns) :
    column.ownership = .auxiliaryColumn := by
  have ownershipMember :=
    layoutBundles_ownership_member frame.temporaries column member
  have layoutsEqual :=
    congrArg CallFootprint.temporaries
      (ConcreteNifsActivatedProgram.selected_footprint_exact
        application certificate.operational certificate.footprint frame)
  rw [layoutsEqual] at ownershipMember
  have exactPair :
      (ConcreteNifsActivatedProgram.cost
          application certificate.operational frame).auxiliaryColumns ≠ 0 ∧
        column.ownership = .auxiliaryColumn := by
    simpa [ConcreteNifsActivatedProgram.footprint, auxiliaryLayout,
      ownedLayout] using ownershipMember
  exact exactPair.2

private theorem temporaryAllocations_auxiliary
    (application : Poseidon23ApplicationProfile Selected)
    (certificate :
      ConcreteNifsVerifyCallRecipe.Certification application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (column : OwnedColumn)
    (member :
      column ∈ temporaryAllocations
        application certificate.operational frame) :
    column.ownership = .auxiliaryColumn :=
  temporaryColumn_auxiliary application certificate frame column
    (List.mem_of_mem_take member)

private theorem temporaryAllocations_length
    (application : Poseidon23ApplicationProfile Selected)
    (certificate :
      ConcreteNifsVerifyCallRecipe.Certification application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    (temporaryAllocations
      application certificate.operational frame).length =
        (ConcreteNifsRawProgram.cost
          application certificate.operational frame).auxiliaryColumns := by
  have fits :=
    ConcreteNifsActivatedProgram.frameFits
      application certificate.operational certificate.footprint frame
  have fitsColumns :
      ConcreteNifsRawProgram.allocationWidth
          application certificate.operational frame ≤
        frame.temporaries.columns.length := by
    unfold ConcreteNifsRawProgram.FrameFits at fits
    change
      ConcreteNifsRawProgram.allocationWidth
          application certificate.operational frame ≤
        (frame.temporaries.columns.map fun column => column.id).length at fits
    simpa only [List.length_map] using fits
  unfold temporaryAllocations
  rw [List.length_take, Nat.min_eq_left fitsColumns,
    ConcreteNifsRawProgram.allocationWidth_eq_cost]

private theorem intrinsicPhysicalCost_exact
    (application : Poseidon23ApplicationProfile Selected)
    (certificate :
      ConcreteNifsVerifyCallRecipe.Certification application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    physicalCost
        (temporaryAllocations
          application certificate.operational frame)
        (ConcreteNifsRawProgram.rows
          application certificate.operational frame) =
      ConcreteNifsRawProgram.cost
        application certificate.operational frame := by
  unfold physicalCost
  rw [columnCost_auxiliary _
      (temporaryAllocations_auxiliary application certificate frame),
    temporaryAllocations_length application certificate frame,
    rowCost_eq_length,
    ConcreteNifsRawProgram.rows_length]
  apply cost_ext <;>
    simp [ConcreteNifsRawProgram.cost,
      ConcreteNifsRawProgram.claimedValueCost,
      ConcreteNifsProofCanonicalityRows.cost,
      ConcreteNifsRunningAuthorityRows.cost,
      ConcreteNifsOperationalSampler.cost,
      KSplitNcOperationalRows.cost,
      KSplitNcOperationalRows.endpointCost,
      KSplitNcTranscript.cost,
      SymbolicDuplex.cost,
      KSplitNcBlockLaneRows.cost,
      KSplitNcFeRows.cost,
      KSplitNcNcRows.cost,
      KFixedPhaseSumCheck.chainCost,
      PiRlcCanonicalSamplerProgram.cost,
      ConcreteNifsOperationalSampler.challengeCost,
      ConcreteNifsPiRlcPointRows.cost,
      ConcreteNifsPiRlcActionRows.cost,
      ConcreteNifsPiDecRows.cost,
      ConcreteNifsOutputRows.cost]

/-- The native selected receipt costs exactly the call output allocation plus
the intrinsic verifier. No activation residual resource remains. -/
theorem selectedReceipt_cost_exact
    (application : Poseidon23ApplicationProfile Selected)
    (certificate :
      ConcreteNifsVerifyCallRecipe.Certification application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    (selectedReceipt application certificate.operational frame).cost =
      ((signature Selected).callOutputs Call.nifsVerify).cost +
        ConcreteNifsRawProgram.cost
          application certificate.operational frame := by
  unfold SelectedReceipt.cost selectedReceipt sourceReceipt
    InstructionReceipt.cost allocations
  change
    physicalCost
        (frame.outputs.columns ++
          temporaryAllocations application certificate.operational frame)
        (ConcreteNifsRawProgram.rows
          application certificate.operational frame) =
      ((signature Selected).callOutputs Call.nifsVerify).cost +
        ConcreteNifsRawProgram.cost
          application certificate.operational frame
  calc
    physicalCost
        (frame.outputs.columns ++
          temporaryAllocations application certificate.operational frame)
        (ConcreteNifsRawProgram.rows
          application certificate.operational frame) =
      physicalCost frame.outputs.columns [] +
        physicalCost
          (temporaryAllocations application certificate.operational frame)
          (ConcreteNifsRawProgram.rows
            application certificate.operational frame) := by
        simpa only [List.nil_append] using
          physicalCost_append frame.outputs.columns
            (temporaryAllocations application certificate.operational frame)
            [] (ConcreteNifsRawProgram.rows
              application certificate.operational frame)
    _ =
      ((signature Selected).callOutputs Call.nifsVerify).cost +
        ConcreteNifsRawProgram.cost
          application certificate.operational frame := by
      rw [outputCost_exact application certificate.operational frame,
        intrinsicPhysicalCost_exact application certificate frame]

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
    (((selectedReceipt application profile frame).rows).map
      fun row => row.source.id).Nodup := by
  apply SelectedReceipt.row_ids_nodup
  exact ConcreteNifsRawProgram.rowIds_nodup application profile frame

theorem selector_supported
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (row : SelectedRow)
    (member : row ∈ (selectedReceipt application profile frame).rows)
    (column : ColumnId)
    (columnMember : column ∈ row.columnIds) :
    column ∈ frame.visibleIds ++ frame.temporaries.ids := by
  have supported :=
    select_supported frame.active
      (ConcreteNifsRawProgram.rows application profile frame)
      (frame.visibleIds ++ frame.temporaries.ids)
      (ConcreteNifsRawProgram.rows_supported application profile frame)
      row member column columnMember
  rcases List.mem_cons.1 supported with rfl | sourceMember
  · exact List.mem_append_left _ (by simp [CallFrame.visibleIds])
  · exact sourceMember

/-- Active native CCS satisfaction reaches the exact selected verifier result.
No residual witness appears in the hypothesis. -/
theorem active_soundness
    (application : Poseidon23ApplicationProfile Selected)
    (certificate :
      ConcreteNifsVerifyCallRecipe.Certification application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (assignment : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (constantOne : assignment frame.one = 1)
    (activeOne : assignment frame.active = 1)
    (decoded :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil))))
    (satisfied :
      NativeCcsSelector.Satisfies
        (selectedReceipt application certificate.operational frame).rows
        assignment) :
    ∃ outputs :
        Schema.Values (typeSystem Selected)
          ((signature Selected).callOutputs Call.nifsVerify),
      callEval Selected Call.nifsVerify
          (.cons running (.cons fresh (.cons proof .nil))) =
        some outputs ∧
      frame.outputs.Decodes (FamilyFor application) assignment outputs := by
  have ownedSatisfied :
      Goldilocks.Satisfies
        (ConcreteNifsRawProgram.rows
          application certificate.operational frame)
        assignment :=
    NativeCcsSelector.active_sound frame.active
      (ConcreteNifsRawProgram.rows application certificate.operational frame)
      assignment activeOne satisfied
  have rawSatisfied :
      RawSatisfies
        (ConcreteNifsRawProgram.rawRows
          application certificate.operational frame)
        assignment :=
    (satisfies_ownRows_iff frame.owner
      (ConcreteNifsRawProgram.rawRows
        application certificate.operational frame)
      assignment).mp
      (by simpa [ConcreteNifsRawProgram.rows] using ownedSatisfied)
  rcases
      ConcreteNifsRawSemantics.call_result_and_output_of_rawRows
        certificate.prime application certificate.operational frame assignment
        running fresh proof constantOne decoded rawSatisfied with
    ⟨output, evaluated, decodedOutput⟩
  exact ⟨.cons output .nil, evaluated, decodedOutput⟩

/-- Honest active execution fills only the intrinsic temporary prefix and
satisfies the selected CCS rows. -/
theorem active_honest_completeness
    (application : Poseidon23ApplicationProfile Selected)
    (certificate :
      ConcreteNifsVerifyCallRecipe.Certification application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (assignment : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (output :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (constantOne : assignment frame.one = 1)
    (_activeOne : assignment frame.active = 1)
    (encodedInputs :
      frame.operands.Encodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil))))
    (encodedOutput :
      frame.outputs.Encodes (FamilyFor application) assignment
        (.cons output .nil))
    (evaluated :
      callEval Selected Call.nifsVerify
          (.cons running (.cons fresh (.cons proof .nil))) =
        some (.cons output .nil)) :
    ∃ completed : ColumnId → Field,
      AgreesOn frame.visibleIds assignment completed ∧
        ChangesOnly frame.temporaries.ids assignment completed ∧
        NativeCcsSelector.Satisfies
          (selectedReceipt application certificate.operational frame).rows
          completed := by
  have fits :=
    ConcreteNifsActivatedProgram.frameFits
      application certificate.operational certificate.footprint frame
  rcases
      ConcreteNifsRawHonest.rows_honest
        certificate.prime certificate.field application
        certificate.operational frame assignment running fresh proof output
        fits constantOne encodedInputs encodedOutput evaluated with
    ⟨completed, agrees, changes, rawSatisfied⟩
  have ownedSatisfied :
      Goldilocks.Satisfies
        (ConcreteNifsRawProgram.rows
          application certificate.operational frame)
        completed :=
    (satisfies_ownRows_iff frame.owner
      (ConcreteNifsRawProgram.rawRows
        application certificate.operational frame)
      completed).mpr
      (by simpa [ConcreteNifsRawProgram.rows] using rawSatisfied)
  exact ⟨completed, agrees, changes,
    NativeCcsSelector.complete frame.active
      (ConcreteNifsRawProgram.rows
        application certificate.operational frame)
      completed ownedSatisfied⟩

/-- Inactive native selection needs no residual completion and changes no
coordinate. -/
theorem inactive_satisfiable
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (assignment : ColumnId → Field)
    (activeZero : assignment frame.active = 0) :
    AgreesOn frame.visibleIds assignment assignment ∧
      ChangesOnly frame.temporaries.ids assignment assignment ∧
      NativeCcsSelector.Satisfies
        (selectedReceipt application profile frame).rows assignment := by
  exact ⟨
    fun _ _ => rfl,
    fun _ _ => rfl,
    NativeCcsSelector.inactive_satisfies frame.active
      (ConcreteNifsRawProgram.rows application profile frame)
      assignment activeZero
  ⟩

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsProgram
