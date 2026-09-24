import NightstreamFPrime.Export.Stage1.PerApplicationFixedPoint
import NightstreamFPrime.Export.Stage1.Wide.PiRLCGeometry
import NightstreamFPrime.Layout.ProductionRelation.ColumnMap
import NightstreamFPrime.Export.Stage1.Wide.ProductCoordinates

/-! Candidate coordinate order. Common retained intervals keep their internal
order; the complete wide PiRLC allocation follows them. Old sampler and
First-54 intervals have no image. The map is used only after support checks. -/

namespace NightstreamFPrime.Export.Stage1.Wide.RetainedLayout

open NightstreamFPrime.Layout
open ProductionRelation

abbrev Program := Lifecycle.Stage1.Application.Program

def hashEnd (program : Program) : Nat := LaterPoseidonRetainedBlocks.samplerStart program
def sharedStart (program : Program) : Nat := PiRLCRetainedGeometry.prefixLogicalWidth program
def sharedEnd (program : Program) : Nat := PiDECRetainedGeometry.completeLogicalWidth program
def applicationStart (program : Program) : Nat := ApplicationOrdinaryGeometry.witnessStart program
def applicationCount (program : Program) : Nat := ApplicationSelectedBlocks.retainedCoordinateCount program

theorem boundaries (program : Program) :
    hashEnd program = 113904174 ∧ sharedStart program = 121293360 ∧
      sharedEnd program = 140293745 ∧ applicationStart program = 149282257 := by
  constructor
  · rw [hashEnd, LaterPoseidonRetainedBlocks.samplerStart_eq]
    unfold PiRLCRetainedGeometry.laterPoseidonStart PiRLCRetainedGeometry.outputPoseidonStart
      PiRLCRetainedGeometry.priorPoseidonStart PiRLCRetainedGeometry.priorPoseidonBlock
      PiRLCRetainedGeometry.outputPoseidonBlock
    simp only [LowNormBlock.Block.lift_coordinateCount]
    rfl
  constructor
  · exact PiRLCRetainedGeometry.prefixLogicalWidth_eq program
  constructor
  · have full := PiRLCSamplerOrdinaryRetainedGeometry.completeLogicalWidth_eq program
    unfold PiRLCSamplerOrdinaryRetainedGeometry.completeLogicalWidth
      PiRLCSamplerOrdinaryRetainedGeometry.freshStart
      PiRLCSamplerOrdinaryRetainedGeometry.logicalStart
      PiRLCSamplerOrdinaryRetainedGeometry.prefixLogicalWidth at full
    rw [PiRLCSamplerOrdinaryRetainedBlocks.logicalBlock_coordinateCount,
      PiRLCSamplerOrdinaryRetainedBlocks.freshBlock_coordinateCount] at full
    unfold sharedEnd
    omega
  · exact PiRLCSamplerOrdinaryRetainedGeometry.completeLogicalWidth_eq program

def commonCount (program : Program) : Nat :=
  hashEnd program + (sharedEnd program - sharedStart program) + applicationCount program

def logicalWidth (program : Program) : Nat := commonCount program + PiRLCGeometry.coordinateCount

theorem commonCount_eq (program : Program) :
    commonCount program = 132904559 + applicationCount program := by
  obtain ⟨hash, start, stop, _⟩ := boundaries program
  simp only [commonCount, hash, start, stop]

theorem logicalWidth_eq (program : Program) :
    logicalWidth program = 137331104 + applicationCount program := by
  rw [logicalWidth, commonCount_eq, PiRLCGeometry.coordinateCount_eq]
  omega

theorem referenceWidth_eq (program : Program) :
    PerApplicationFixedPoint.logicalWidth program = 149282257 + applicationCount program :=
  ApplicationRetainedGeometry.completeLogicalWidth_eq program

def outputStart (program : Program) : Nat := commonCount program + 135813
def quotientStart (program : Program) : Nat := outputStart program + 2145366

/-- Each successful branch names a retained coordinate, never a helper. -/
def column? (program : Program) (source : Nat) : Option Nat :=
  if source < hashEnd program then some source
  else if sharedStart program ≤ source ∧ source < sharedEnd program then
    some (hashEnd program + (source - sharedStart program))
  else if applicationStart program ≤ source ∧
      source < applicationStart program + applicationCount program then
    some (hashEnd program + (sharedEnd program - sharedStart program) +
      (source - applicationStart program))
  else if output : 119147994 ≤ source ∧ source < 121293360 then
    some (outputStart program +
      (ProductCoordinates.coordinate ⟨source - 119147994, by omega⟩).val)
  else if quotient : 114443652 ≤ source ∧ source < 116589018 then
    some (quotientStart program +
      (ProductCoordinates.coordinate ⟨source - 114443652, by omega⟩).val)
  else none

/-- Exact domain of the coordinate map. This predicate concerns stored
entries, including entries whose coefficient might later cancel. -/
def Live (program : Program) (source : Nat) : Prop :=
  source < hashEnd program ∨
  (sharedStart program ≤ source ∧ source < sharedEnd program) ∨
  (applicationStart program ≤ source ∧ source < applicationStart program + applicationCount program) ∨
  (119147994 ≤ source ∧ source < 121293360) ∨
  (114443652 ≤ source ∧ source < 116589018)

theorem live_iff_mapped (program : Program) (source : Nat) :
    Live program source ↔ (column? program source).isSome = true := by
  unfold Live column?
  split_ifs <;> simp_all

theorem column?_lt (program : Program) (source target : Nat)
    (mapped : column? program source = some target) : target < logicalWidth program := by
  obtain ⟨hash, start, stop, app⟩ := boundaries program
  unfold column? at mapped
  rw [hash, start, stop, app] at mapped
  rw [logicalWidth_eq]
  simp only [quotientStart, outputStart, commonCount_eq] at mapped
  split_ifs at mapped <;> simp only [Option.some.injEq] at mapped <;> omega

theorem column?_injective (program : Program) {left right target : Nat}
    (leftMap : column? program left = some target)
    (rightMap : column? program right = some target) : left = right := by
  obtain ⟨hash, start, stop, app⟩ := boundaries program
  unfold column? at leftMap rightMap
  rw [hash, start, stop, app] at leftMap rightMap
  simp only [quotientStart, outputStart, commonCount_eq] at leftMap rightMap
  split_ifs at leftMap <;> split_ifs at rightMap <;>
    simp only [Option.some.injEq] at leftMap rightMap <;>
    first | omega | skip
  all_goals
    have mappedEq := Nat.add_left_cancel (leftMap.trans rightMap.symm)
    have inputEq := ProductCoordinates.coordinate_injective (Fin.ext mappedEq)
    have values := congrArg Fin.val inputEq
    dsimp only at values
    omega

/-- A removed source coordinate has no value of this type. -/
def column (program : Program) (source : Fin (PerApplicationFixedPoint.logicalWidth program))
    (live : Live program source.val) : Fin (logicalWidth program) :=
  let mapped := (live_iff_mapped program source.val).mp live
  ⟨(column? program source.val).get mapped,
    column?_lt program source.val _ (Option.some_get mapped).symm⟩

theorem column_mapped (program : Program)
    (source : Fin (PerApplicationFixedPoint.logicalWidth program)) (live : Live program source.val) :
    column? program source.val = some (column program source live).val :=
  (Option.some_get ((live_iff_mapped program source.val).mp live)).symm

theorem column_injective (program : Program)
    (left right : Fin (PerApplicationFixedPoint.logicalWidth program)) (leftLive rightLive)
    (equal : column program left leftLive = column program right rightLive) : left = right := by
  apply Fin.ext
  apply column?_injective program (column_mapped program left leftLive)
  rw [congrArg Fin.val equal]
  exact column_mapped program right rightLive

theorem column_of_some (program : Program)
    (source : Fin (PerApplicationFixedPoint.logicalWidth program)) (live : Live program source.val)
    (target : Nat) (mapped : column? program source.val = some target) :
    (column program source live).val = target := by
  have selected := Option.some_get ((live_iff_mapped program source.val).mp live)
  exact Option.some.inj (selected.trans mapped)

/-- Rename each stored entry using its own support proof. There is no default
column, filtering, or deletion of unsupported entries. -/
def renameForm (program : Program)
    (form : SparseForm (PerApplicationFixedPoint.logicalWidth program))
    (supported : ∀ entry ∈ form.entries, Live program entry.column.val) :
    SparseForm (logicalWidth program) :=
  form.mapColumnsChecked (column program) supported

theorem renameForm_length (program : Program)
    (form : SparseForm (PerApplicationFixedPoint.logicalWidth program)) (supported) :
    (renameForm program form supported).entries.length = form.entries.length := by
  exact List.length_pmap

theorem renameForm_congr (program : Program)
    {left right : SparseForm (PerApplicationFixedPoint.logicalWidth program)}
    (same : left = right) (leftSupported rightSupported) :
    renameForm program left leftSupported = renameForm program right rightSupported := by
  cases same
  rfl

theorem publicColumn (program : Program) (source : Nat) (bounded : source < 270) :
    column? program source = some source := by
  have hash := (boundaries program).1
  simp only [column?, hash, if_pos (show source < 113904174 by omega)]

end NightstreamFPrime.Export.Stage1.Wide.RetainedLayout
