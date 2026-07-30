import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRawProgram
import Nightstream.Implementation.Lowering.Goldilocks.SelectedBranchSupport

/-!
Contract: install the five verifier-derived extension-field claims that form
the leading ten temporary coordinates of one selected `nifsVerify` call.

The write list is the actual temporary prefix owned by the call frame.  It is
not a parallel numeric fixture.  Under `FrameFits`, every written coordinate
is recovered exactly through the sole global column map, while every visible
input and output column is preserved.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsClaimedValuesHonest

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

/-- The five semantic values occupying the exact claimed-chain ABI order. -/
structure Values where
  feInitial : K
  feBoundary : K
  feTerminal : K
  ncInitial : K
  ncTerminal : K

/-- Exact base-field serialization of the five values. -/
def Values.coordinates (values : Values) : List Field :=
  [values.feInitial.c0, values.feInitial.c1,
    values.feBoundary.c0, values.feBoundary.c1,
    values.feTerminal.c0, values.feTerminal.c1,
    values.ncInitial.c0, values.ncInitial.c1,
    values.ncTerminal.c0, values.ncTerminal.c1]

def Values.get (values : Values) : Nat → K
  | 0 => values.feInitial
  | 1 => values.feBoundary
  | 2 => values.feTerminal
  | 3 => values.ncInitial
  | _ => values.ncTerminal

private theorem concreteK_eq
    (left right : K)
    (c0Equal : left.c0 = right.c0)
    (c1Equal : left.c1 = right.c1) :
    left = right := by
  cases left
  cases right
  simp only at c0Equal c1Equal
  cases c0Equal
  cases c1Equal
  rfl

private theorem field_mod_self (value : Field) :
    value.val % Nightstream.Implementation.R1CS.goldilocksP =
      value.val := by
  apply Nat.mod_eq_of_lt
  simpa [Nightstream.Implementation.R1CS.goldilocksP,
    Nightstream.SuperNeo.Concrete.goldilocksModulus] using value.isLt

@[simp] theorem Values.coordinates_length (values : Values) :
    values.coordinates.length = 10 := by
  simp [Values.coordinates]

/-- The actual physical prefix receiving the claimed values. -/
def columns
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    List ColumnId :=
  (ConcreteNifsRawProgram.allocation application profile frame).take 10

/-- Install the ten canonical residues and preserve every other coordinate. -/
def seed
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
    (values : Values) : ColumnId → Field :=
  writeColumns assignment (columns application profile frame)
    values.coordinates

private theorem ten_le_allocationWidth
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    10 ≤ ConcreteNifsRawProgram.allocationWidth application profile frame := by
  unfold ConcreteNifsRawProgram.allocationWidth
  omega

theorem columns_length
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    (columns application profile frame).length = 10 := by
  unfold columns
  rw [List.length_take,
    ConcreteNifsRawProgram.allocation,
    List.length_map,
    ConcreteNifsRawProgram.allocationSources,
    List.length_map, List.length_range]
  exact Nat.min_eq_left
    (ten_le_allocationWidth application profile frame)

theorem columns_nodup
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame) :
    (columns application profile frame).Nodup :=
  (ConcreteNifsRawProgram.allocation_nodup
    application profile frame fits).take

theorem columns_eq_temporaryPrefix
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame) :
    columns application profile frame =
      frame.temporaries.ids.take 10 := by
  unfold columns
  rw [ConcreteNifsRawProgram.allocation_eq_temporaryPrefix
    application profile frame fits, List.take_take]
  rw [Nat.min_eq_left
    (ten_le_allocationWidth application profile frame)]

theorem columns_subset_temporaries
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame) :
    ∀ column ∈ columns application profile frame,
      column ∈ frame.temporaries.ids := by
  rw [columns_eq_temporaryPrefix application profile frame fits]
  intro column member
  exact List.mem_of_mem_take member

theorem seed_changesOnly
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
    (values : Values)
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame) :
    ChangesOnly frame.temporaries.ids assignment
      (seed application profile frame assignment values) := by
  intro column notTemporary
  apply writeColumns_of_not_mem
  intro written
  exact notTemporary
    (columns_subset_temporaries application profile frame fits column written)

theorem seed_agrees_visible
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
    (values : Values)
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame) :
    AgreesOn frame.visibleIds assignment
      (seed application profile frame assignment values) := by
  apply writeColumns_agreesOn assignment
    (columns application profile frame) frame.visibleIds values.coordinates
  intro column written visible
  exact frame.temporariesDisjointVisible column
    (columns_subset_temporaries application profile frame fits column written)
    visible

/-- The physical write list recovers the exact ten serialized residues. -/
theorem seed_coordinates
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
    (values : Values)
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame) :
    (columns application profile frame).map
        (seed application profile frame assignment values) =
      values.coordinates := by
  apply writeColumns_map_eq
  · rw [columns_length, Values.coordinates_length]
  · exact columns_nodup application profile frame fits

/-- Exact physical recovery of one claimed-value base-field coordinate. -/
theorem seed_temporaryField
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
    (values : Values)
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame)
    (index : Nat)
    (indexLt : index < 10) :
    seed application profile frame assignment values
        (columnMap frame (temporarySource frame index)) =
      values.coordinates.getD index 0 := by
  have temporaryLt : index < frame.temporaries.ids.length :=
    Nat.lt_of_lt_of_le indexLt
      (Nat.le_trans
        (ten_le_allocationWidth application profile frame) fits)
  rw [columnMap_temporarySource frame temporaryLt]
  have recovered :=
    seed_coordinates application profile frame assignment values fits
  have columnsLt :
      index < (columns application profile frame).length := by
    rw [columns_length]
    exact indexLt
  have mappedLt :
      index <
        ((columns application profile frame).map
          (seed application profile frame assignment values)).length := by
    simpa using columnsLt
  have valuesLt : index < values.coordinates.length := by
    rw [Values.coordinates_length]
    exact indexLt
  have atIndex := congrArg
    (fun coordinates : List Field => coordinates.getD index 0)
    recovered
  change
    ((columns application profile frame).map
        (seed application profile frame assignment values)).getD index 0 =
      values.coordinates.getD index 0
    at atIndex
  rw [← List.getElem_eq_getD
      (l := (columns application profile frame).map
        (seed application profile frame assignment values))
      (i := index) (h := mappedLt) 0,
    ← List.getElem_eq_getD
      (l := values.coordinates) (i := index) (h := valuesLt) 0]
    at atIndex
  simp only [List.getElem_map] at atIndex
  rw [List.getElem_eq_getD
      (l := columns application profile frame)
      (i := index) (h := columnsLt) frame.one,
    List.getElem_eq_getD
      (l := values.coordinates) (i := index) (h := valuesLt) 0]
    at atIndex
  have tenLeTemporaries : 10 ≤ frame.temporaries.ids.length :=
    Nat.le_trans
      (ten_le_allocationWidth application profile frame) fits
  have takeLt : index < (frame.temporaries.ids.take 10).length := by
    rw [List.length_take, Nat.min_eq_left tenLeTemporaries]
    exact indexLt
  have columnGetD :
      (columns application profile frame).getD index frame.one =
        frame.temporaries.ids.getD index frame.one := calc
    _ = (frame.temporaries.ids.take 10).getD index frame.one :=
      congrArg
        (fun ids : List ColumnId => ids.getD index frame.one)
        (columns_eq_temporaryPrefix application profile frame fits)
    _ = (frame.temporaries.ids.take 10)[index] :=
      (List.getElem_eq_getD
        (l := frame.temporaries.ids.take 10)
        (i := index) (h := takeLt) frame.one).symm
    _ = frame.temporaries.ids[index] := by
      simp only [List.getElem_take]
    _ = frame.temporaries.ids.getD index frame.one :=
      List.getElem_eq_getD
        (l := frame.temporaries.ids)
        (i := index) (h := temporaryLt) frame.one
  rw [columnGetD] at atIndex
  rw [List.getElem_eq_getD
    (l := frame.temporaries.ids)
    (i := index) (h := temporaryLt) frame.one]
  exact atIndex

/-- Decoding either coordinate pair recovers the corresponding claimed
extension-field value. -/
theorem seed_temporaryK
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
    (values : Values)
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame)
    (index : Nat)
    (indexLt : index < 5) :
    KSplitNcTranscriptSemantics.decodedColumns
        (NumericRowBridge.numericAssignment (columnMap frame)
          (seed application profile frame assignment values))
        (ConcreteNifsOperationalOccurrence.temporaryK
          (FamilyFor application) frame index) =
      values.get index := by
  have indexCases :
      index = 0 ∨ index = 1 ∨ index = 2 ∨ index = 3 ∨ index = 4 := by
    omega
  rcases indexCases with rfl | rfl | rfl | rfl | rfl <;>
    apply concreteK_eq <;>
    apply Fin.ext <;>
    simp [KSplitNcTranscriptSemantics.decodedColumns,
      ConcreteNifsOperationalOccurrence.temporaryK,
      Values.get, Values.coordinates,
      Nightstream.Implementation.R1CS.ProjectionProgram.KColumns.value,
      Nightstream.Implementation.R1CS.ProjectionProgram.baseAt,
      Nightstream.Implementation.R1CS.ProjectionProgram.residue,
      KConcreteFixedPhaseBridge.ofProjection,
      NumericRowBridge.numericAssignment,
      field_mod_self,
      seed_temporaryField application profile frame assignment values fits]

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsClaimedValuesHonest
