import NightstreamFPrime.Export.Stage1.DirectPoseidonFootprint
import NightstreamFPrime.Export.Stage1.Package
import NightstreamFPrime.Export.Stage1.PerApplicationPreservation
import NightstreamFPrime.Export.Stage1.PermutationPlan
import NightstreamFPrime.Layout.ProductionRelation.PoseidonRetainedBlock

/-!
Owns the compact retained low-norm field block for every Poseidon2 invocation
in the current Stage 1 prefix. The order is the prior hash chain, the output
hash chain, then the explicit PiCCS and PiRLC invocation list.

This module selects only the retained S-box source columns. It does not select
the remaining non-Poseidon slots or claim the complete final assignment fit.
-/

namespace NightstreamFPrime.Export.Stage1.PoseidonRetainedBlock

open NightstreamFPrime.Export.Package
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

def basePackage (_delay : Unit := ()) : CircuitPackage :=
  PerApplicationPackage.basePackage ()

def priorInvocationCount : Nat := Data.priorChain.absorbCount + 1
def outputInvocationCount : Nat := Data.outputChain.absorbCount + 1
def pilotInvocationCount : Nat := priorInvocationCount + outputInvocationCount
def laterInvocationCountReference (_delay : Unit := ()) : Nat :=
  basePackage.permutationInvocations.length
def laterInvocationCount : Nat := 7757
def totalInvocationCount : Nat := pilotInvocationCount + laterInvocationCount

@[simp] theorem priorInvocationCount_eq : priorInvocationCount = 12350 := by
  rfl

@[simp] theorem outputInvocationCount_eq : outputInvocationCount = 12350 := by
  rfl

@[simp] theorem pilotInvocationCount_eq : pilotInvocationCount = 24700 := by
  simp [pilotInvocationCount]

@[simp] theorem laterInvocationCount_eq : laterInvocationCount = 7757 := by
  rfl

@[simp] theorem basePackage_permutationInvocations_length :
    basePackage.permutationInvocations.length = laterInvocationCount := by
  unfold basePackage PerApplicationPackage.basePackage laterInvocationCount
  exact Package.circuitPackage_permutation_invocations

theorem basePackage_permutationInvocations_eq :
    basePackage.permutationInvocations = Data.permutationInvocations () := by
  unfold basePackage PerApplicationPackage.basePackage
  exact Data.circuitPackage_permutationInvocations.trans
    Data.components_permutationInvocations

@[simp] theorem data_permutationInvocations_length :
    (Data.permutationInvocations ()).length = laterInvocationCount := by
  rw [← basePackage_permutationInvocations_eq]
  exact basePackage_permutationInvocations_length

theorem laterInvocationCount_eq_reference :
    laterInvocationCount = laterInvocationCountReference () := by
  unfold laterInvocationCountReference
  exact basePackage_permutationInvocations_length.symm

@[simp] theorem totalInvocationCount_eq : totalInvocationCount = 32457 := by
  simp [totalInvocationCount]

private theorem priorChain_mem :
    Data.priorChain ∈ basePackage.hashChains := by
  unfold basePackage PerApplicationPackage.basePackage
  rw [Data.circuitPackage_hashChains]
  simp

private theorem outputChain_mem :
    Data.outputChain ∈ basePackage.hashChains := by
  unfold basePackage PerApplicationPackage.basePackage
  rw [Data.circuitPackage_hashChains]
  simp

/-- Exact source-local start of one prior-hash invocation. -/
def priorWitnessStart (invocation : Fin priorInvocationCount) : Nat :=
  Data.priorChain.witnessStart +
    invocation.val * PoseidonScheduleTrace.localColumnCount

theorem priorWitnessStart_bound (invocation : Fin priorInvocationCount) :
    priorWitnessStart invocation + PoseidonScheduleTrace.localColumnCount ≤
      basePackage.layout.constantColumn := by
  simpa [priorWitnessStart, priorInvocationCount,
    PoseidonScheduleTrace.localColumnCount] using
      PerApplicationPreservation.canonicalHashInvocation_witnessBound
        Data.priorChain priorChain_mem invocation

/-- Exact source-local start of one output-hash invocation. -/
def outputWitnessStart (invocation : Fin outputInvocationCount) : Nat :=
  Data.outputChain.witnessStart +
    invocation.val * PoseidonScheduleTrace.localColumnCount

theorem outputWitnessStart_bound (invocation : Fin outputInvocationCount) :
    outputWitnessStart invocation + PoseidonScheduleTrace.localColumnCount ≤
      basePackage.layout.constantColumn := by
  simpa [outputWitnessStart, outputInvocationCount,
    PoseidonScheduleTrace.localColumnCount] using
      PerApplicationPreservation.canonicalHashInvocation_witnessBound
        Data.outputChain outputChain_mem invocation

/-- Exact source-local start of one explicit PiCCS or PiRLC invocation. -/
def laterWitnessStart (invocation : Fin laterInvocationCount) : Nat :=
  let index : Fin basePackage.permutationInvocations.length :=
    ⟨invocation.val, by simpa [laterInvocationCount] using invocation.isLt⟩
  (basePackage.permutationInvocations.get index).witnessStart

/-- Delayed random-access view of the canonical later witness-start schedule. -/
def laterWitnessStartsArray (_delay : Unit := ()) : Array Nat :=
  (PermutationPlan.canonicalWitnessStarts ()).toArray

@[simp] theorem laterWitnessStartsArray_size :
    (laterWitnessStartsArray ()).size = laterInvocationCount := by
  rw [laterWitnessStartsArray, List.size_toArray,
    PermutationPlan.canonicalWitnessStarts_materializes, List.length_map]
  exact data_permutationInvocations_length

/-- Allocation-bounded executable lookup for the canonical witness start. -/
@[inline] def directLaterWitnessStart
    (invocation : Fin laterInvocationCount) : Nat :=
  ((laterWitnessStartsArray ())[invocation.val]'(by
    simpa using invocation.isLt))

/-- Random access returns the exact list-selected canonical witness start. -/
theorem directLaterWitnessStart_eq_laterWitnessStart
    (invocation : Fin laterInvocationCount) :
    directLaterWitnessStart invocation = laterWitnessStart invocation := by
  change
    ((PermutationPlan.canonicalWitnessStarts ()).toArray[invocation.val]'(by
      rw [List.size_toArray,
        PermutationPlan.canonicalWitnessStarts_materializes, List.length_map,
        data_permutationInvocations_length]
      exact invocation.isLt)) =
    (basePackage.permutationInvocations[invocation.val]'(by
      simpa [laterInvocationCount] using invocation.isLt)).witnessStart
  simp only [List.getElem_toArray,
    PermutationPlan.canonicalWitnessStarts_materializes, List.getElem_map,
    basePackage_permutationInvocations_eq]

/-- Compiled retained-slot emission uses the proved random-access lookup. -/
@[csimp] theorem laterWitnessStart_eq_directLaterWitnessStart :
    @laterWitnessStart = @directLaterWitnessStart := by
  funext invocation
  exact (directLaterWitnessStart_eq_laterWitnessStart invocation).symm

theorem laterWitnessStart_bound (invocation : Fin laterInvocationCount) :
    laterWitnessStart invocation + PoseidonScheduleTrace.localColumnCount ≤
      basePackage.layout.constantColumn := by
  let index : Fin basePackage.permutationInvocations.length :=
    ⟨invocation.val, by simpa [laterInvocationCount] using invocation.isLt⟩
  change
    (basePackage.permutationInvocations.get index).witnessStart + 592 ≤
      basePackage.layout.constantColumn
  exact PerApplicationPreservation.canonicalPermutationInvocation_witnessBound
    (basePackage.permutationInvocations.get index) (List.get_mem _ index)

def priorBlock : LowNormBlock.Block basePackage.layout.constantColumn :=
  Layout.ProductionRelation.PoseidonRetainedBlock.block
    basePackage.layout.constantColumn priorInvocationCount priorWitnessStart
      priorWitnessStart_bound

def outputBlock : LowNormBlock.Block basePackage.layout.constantColumn :=
  Layout.ProductionRelation.PoseidonRetainedBlock.block
    basePackage.layout.constantColumn outputInvocationCount outputWitnessStart
      outputWitnessStart_bound

def laterBlock : LowNormBlock.Block basePackage.layout.constantColumn :=
  Layout.ProductionRelation.PoseidonRetainedBlock.block
    basePackage.layout.constantColumn laterInvocationCount laterWitnessStart
      laterWitnessStart_bound

@[simp] theorem laterBlock_kind : laterBlock.kind = .field := by
  rfl

@[simp] theorem priorBlock_slotCount : priorBlock.slotCount = 1062100 := by
  rw [priorBlock,
    Layout.ProductionRelation.PoseidonRetainedBlock.block_slotCount,
    priorInvocationCount_eq]

@[simp] theorem outputBlock_slotCount : outputBlock.slotCount = 1062100 := by
  rw [outputBlock,
    Layout.ProductionRelation.PoseidonRetainedBlock.block_slotCount,
    outputInvocationCount_eq]

@[simp] theorem laterBlock_slotCount : laterBlock.slotCount = 667102 := by
  rw [laterBlock,
    Layout.ProductionRelation.PoseidonRetainedBlock.block_slotCount,
    laterInvocationCount_eq]

@[simp] theorem priorBlock_coordinateCount :
    priorBlock.coordinateCount = 43546100 := by
  rw [priorBlock,
    Layout.ProductionRelation.PoseidonRetainedBlock.block_coordinateCount,
    priorInvocationCount_eq]

@[simp] theorem outputBlock_coordinateCount :
    outputBlock.coordinateCount = 43546100 := by
  rw [outputBlock,
    Layout.ProductionRelation.PoseidonRetainedBlock.block_coordinateCount,
    outputInvocationCount_eq]

@[simp] theorem laterBlock_coordinateCount :
    laterBlock.coordinateCount = 27351182 := by
  rw [laterBlock,
    Layout.ProductionRelation.PoseidonRetainedBlock.block_coordinateCount,
    laterInvocationCount_eq]

/-- Three fixed blocks preserve the package's canonical invocation order. -/
def retainedBlocks : List
    (LowNormBlock.Block basePackage.layout.constantColumn) :=
  [priorBlock, outputBlock, laterBlock]

def retainedSlotCount : Nat :=
  (retainedBlocks.map fun block => block.slotCount).sum

def retainedCoordinateCount : Nat :=
  (retainedBlocks.map fun block => block.coordinateCount).sum

@[simp] theorem retainedSlotCount_eq : retainedSlotCount = 2791302 := by
  simp [retainedSlotCount, retainedBlocks, priorBlock, outputBlock, laterBlock,
    Layout.ProductionRelation.PoseidonRetainedBlock.block_slotCount,
    priorInvocationCount_eq, outputInvocationCount_eq,
    laterInvocationCount_eq]

@[simp] theorem retainedCoordinateCount_eq :
    retainedCoordinateCount = 114443382 := by
  simp [retainedCoordinateCount, retainedBlocks, priorBlock, outputBlock,
    laterBlock,
    Layout.ProductionRelation.PoseidonRetainedBlock.block_coordinateCount,
    priorInvocationCount_eq, outputInvocationCount_eq,
    laterInvocationCount_eq]

/-- The concrete block is the proof object counted by the earlier footprint
guard. -/
theorem retainedBlock_coordinateCount_eq_footprint :
    retainedCoordinateCount =
      DirectPoseidonFootprint.directSboxCoordinateCount := by
  rw [retainedCoordinateCount_eq,
    DirectPoseidonFootprint.directSboxCoordinateCount_eq]

end NightstreamFPrime.Export.Stage1.PoseidonRetainedBlock
