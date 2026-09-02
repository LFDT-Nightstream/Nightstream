import NightstreamFPrime.Export.Stage1.PerApplicationPreservation
import NightstreamFPrime.Export.Stage1.PoseidonRetainedBlock
import NightstreamFPrime.Layout.LowNormBlock

/-!
Owns retained low-norm input blocks for the two pilot Poseidon2 hash chains.
Each block keeps exactly the authoritative preimage fields. Intermediate
permutation outputs are derived from retained S-box values and are not copied.

This module does not select later PiCCS or PiRLC permutation inputs.
-/

namespace NightstreamFPrime.Export.Stage1.PoseidonInputRetainedBlock

open NightstreamFPrime.Export.Package
open NightstreamFPrime.Layout

def basePackage (_delay : Unit := ()) : CircuitPackage :=
  PerApplicationPackage.basePackage ()

private theorem priorChain_mem : Data.priorChain ∈ basePackage.hashChains := by
  unfold basePackage PerApplicationPackage.basePackage
  rw [Data.circuitPackage_hashChains]
  simp

private theorem outputChain_mem : Data.outputChain ∈ basePackage.hashChains := by
  unfold basePackage PerApplicationPackage.basePackage
  rw [Data.circuitPackage_hashChains]
  simp

theorem priorInputEnd :
    Data.priorChain.inputStart + Data.priorChain.inputLength ≤
      basePackage.layout.constantColumn :=
  (PerApplicationPreservation.canonicalHashChain_private
    Data.priorChain priorChain_mem).inputEnd

theorem outputInputEnd :
    Data.outputChain.inputStart + Data.outputChain.inputLength ≤
      basePackage.layout.constantColumn :=
  (PerApplicationPreservation.canonicalHashChain_private
    Data.outputChain outputChain_mem).inputEnd

def block (chain : HashChain)
    (inputEnd : chain.inputStart + chain.inputLength ≤
      basePackage.layout.constantColumn) :
    LowNormBlock.Block basePackage.layout.constantColumn where
  kind := .field
  slotCount := chain.inputLength
  source := fun slot =>
    ⟨chain.inputStart + slot.val, by
      have slotBound := slot.isLt
      omega⟩

def priorBlock : LowNormBlock.Block basePackage.layout.constantColumn :=
  block Data.priorChain priorInputEnd

def outputBlock : LowNormBlock.Block basePackage.layout.constantColumn :=
  block Data.outputChain outputInputEnd

@[simp] theorem priorBlock_kind : priorBlock.kind = .field := by
  rfl

@[simp] theorem outputBlock_kind : outputBlock.kind = .field := by
  rfl

@[simp] theorem priorBlock_slotCount : priorBlock.slotCount = 49393 := by
  norm_num [priorBlock, block, Data.priorChain, Data.liftPilotChain,
    PilotData.priorChain, PilotValues.stateHashWords,
    PilotValues.stateHashBaseWords]

@[simp] theorem outputBlock_slotCount : outputBlock.slotCount = 49393 := by
  norm_num [outputBlock, block, Data.outputChain, Data.liftPilotChain,
    PilotData.outputChain, PilotValues.stateHashWords,
    PilotValues.stateHashBaseWords]

theorem priorBlock_source (slot : Fin Data.priorChain.inputLength) :
    priorBlock.source slot =
      ⟨Data.priorChain.inputStart + slot.val, by
        have slotBound := slot.isLt
        exact Nat.lt_of_lt_of_le (Nat.add_lt_add_left slotBound _)
          priorInputEnd⟩ := by
  apply Fin.ext
  rfl

theorem outputBlock_source (slot : Fin Data.outputChain.inputLength) :
    outputBlock.source slot =
      ⟨Data.outputChain.inputStart + slot.val, by
        have slotBound := slot.isLt
        exact Nat.lt_of_lt_of_le (Nat.add_lt_add_left slotBound _)
          outputInputEnd⟩ := by
  apply Fin.ext
  rfl

@[simp] theorem priorBlock_coordinateCount :
    priorBlock.coordinateCount = 2025113 := by
  norm_num [priorBlock, block, LowNormBlock.Block.coordinateCount,
    LowNormSlot.Kind.width, BalancedTernary.width, Data.priorChain,
    Data.liftPilotChain, PilotData.priorChain, PilotValues.stateHashWords,
    PilotValues.stateHashBaseWords]

@[simp] theorem outputBlock_coordinateCount :
    outputBlock.coordinateCount = 2025113 := by
  norm_num [outputBlock, block, LowNormBlock.Block.coordinateCount,
    LowNormSlot.Kind.width, BalancedTernary.width, Data.outputChain,
    Data.liftPilotChain, PilotData.outputChain, PilotValues.stateHashWords,
    PilotValues.stateHashBaseWords]

def retainedCoordinateCount : Nat :=
  priorBlock.coordinateCount + outputBlock.coordinateCount

@[simp] theorem retainedCoordinateCount_eq :
    retainedCoordinateCount = 4050226 := by
  simp [retainedCoordinateCount]

end NightstreamFPrime.Export.Stage1.PoseidonInputRetainedBlock
