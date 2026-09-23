import NightstreamFPrime.Export.Stage1.ApplicationRetainedBlocks
import NightstreamFPrime.Export.Stage1.ApplicationPoseidonRetainedBlock

/-! Select the proved retained application block from the circuit certificate. -/

namespace NightstreamFPrime.Export.Stage1.ApplicationSelectedBlocks

open NightstreamFPrime.Layout

abbrev Program := Lifecycle.Stage1.Application.Program

def localBlock (application : Program) :
    LowNormBlock.Block (ApplicationRetainedBlocks.sourceWidth application) :=
  match application.compactHashChain with
  | none => ApplicationRetainedBlocks.localBlock application
  | some certificate => ApplicationPoseidonRetainedBlock.block application certificate

def localCount (application : Program) : Nat := (localBlock application).slotCount

theorem localBlock_none (application : Program) (ordinary : application.compactHashChain = none) :
    localBlock application = ApplicationRetainedBlocks.localBlock application := by
  simp only [localBlock, ordinary]

theorem localBlock_some (application : Program)
    (certificate : ApplicationPoseidonRetainedBlock.Certificate application)
    (selected : application.compactHashChain = some certificate) :
    localBlock application = ApplicationPoseidonRetainedBlock.block application certificate := by
  simp only [localBlock, selected]

theorem localBlock_kind (application : Program) : (localBlock application).kind = .field := by
  unfold localBlock
  split <;> rfl

theorem localBlock_coordinateCount (application : Program) :
    (localBlock application).coordinateCount = localCount application * 41 := by
  rw [LowNormBlock.Block.coordinateCount, localBlock_kind]
  rfl

theorem source_after_localStart (application : Program)
    (slot : Fin (localBlock application).slotCount) :
    Layout.Stage1.ApplicationInputs.localStart application ≤ ((localBlock application).source slot).val := by
  revert slot
  unfold localBlock
  split
  · intro slot
    change Layout.Stage1.ApplicationInputs.localStart application ≤
      Layout.Stage1.ApplicationInputs.localStart application + slot.val
    omega
  · rename_i certificate selected
    intro slot
    change Layout.Stage1.ApplicationInputs.localStart application ≤
      ApplicationPoseidonRetainedBlock.witnessStart application (Fin.decodeProd slot).1 +
        (ProductionRelation.PoseidonRetainedSlots.localOutput (Fin.decodeProd slot).2).val
    unfold ApplicationPoseidonRetainedBlock.witnessStart
    omega

theorem localCount_le_ordinary (application : Program) :
    localCount application ≤ ApplicationRetainedBlocks.localCount application := by
  unfold localCount localBlock
  split
  · exact Nat.le_refl _
  · rename_i certificate selected
    rw [ApplicationPoseidonRetainedBlock.block_slotCount]
    have bound := ApplicationPoseidonRetainedBlock.sourceWidth_bound application certificate
    unfold ApplicationRetainedBlocks.localCount ApplicationRetainedBlocks.sourceWidth
    omega

def retainedCoordinateCount (application : Program) : Nat :=
  (ApplicationRetainedBlocks.witnessBlock application).coordinateCount +
    (localBlock application).coordinateCount

theorem retainedCoordinateCount_eq (application : Program) :
    retainedCoordinateCount application = (application.witnessWordCount + localCount application) * 41 := by
  rw [retainedCoordinateCount, localBlock_coordinateCount]
  change application.witnessWordCount * 41 + localCount application * 41 = _
  omega

end NightstreamFPrime.Export.Stage1.ApplicationSelectedBlocks
