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

/-- Total form-renaming function. A caller must prove that its entries have a
successful `column?` result; the fallback is not a retained allocation. -/
def column (program : Program) (source : Fin (PerApplicationFixedPoint.logicalWidth program)) :
    Fin (logicalWidth program) :=
  match mapped : column? program source.val with
  | some target => ⟨target, column?_lt program source.val target mapped⟩
  | none => ⟨0, by rw [logicalWidth_eq]; omega⟩

theorem column_of_some (program : Program)
    (source : Fin (PerApplicationFixedPoint.logicalWidth program)) (target : Nat)
    (mapped : column? program source.val = some target) :
    (column program source).val = target := by
  unfold column
  split
  · rename_i result resultEq
    rw [mapped] at resultEq
    exact (Option.some.inj resultEq).symm
  · rename_i resultEq
    rw [mapped] at resultEq
    contradiction

theorem publicColumn (program : Program) (source : Nat) (bounded : source < 270) :
    column? program source = some source := by
  have hash := (boundaries program).1
  simp only [column?, hash, if_pos (show source < 113904174 by omega)]

end NightstreamFPrime.Export.Stage1.Wide.RetainedLayout
