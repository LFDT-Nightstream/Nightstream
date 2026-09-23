import NightstreamFPrime.Export.Stage1.ApplicationDirectSource
import NightstreamFPrime.Layout.ProductionRelation.PoseidonRetainedBlock

/-! Retain only the three variable-block permutations of the checked hash-chain circuit. -/

namespace NightstreamFPrime.Export.Stage1.ApplicationPoseidonRetainedBlock

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle

abbrev Certificate (application : Stage1.Application.Program) :=
  Stage1.Application.HashChainCircuit application.witnessWordCount application.circuit

def witnessStart (application : Stage1.Application.Program) (invocation : Fin 3) : Nat :=
  Layout.Stage1.ApplicationInputs.localStart application + 5920 + invocation.val * 592

theorem sourceWidth_bound (application : Stage1.Application.Program)
    (certificate : Certificate application) :
    Layout.Stage1.ApplicationInputs.localStart application + 7696 ≤
      ApplicationDirectSource.sourceWidth application := by
  unfold ApplicationDirectSource.sourceWidth ApplicationPackage.r1csFreshStart
    ApplicationPackage.operations
  rw [certificate.localLength]
  omega

theorem witnessStart_bound (application : Stage1.Application.Program)
    (certificate : Certificate application) (invocation : Fin 3) :
    witnessStart application invocation + PoseidonScheduleTrace.localColumnCount ≤
      ApplicationDirectSource.sourceWidth application := by
  have bound := sourceWidth_bound application certificate
  have index := invocation.isLt
  unfold witnessStart PoseidonScheduleTrace.localColumnCount
  omega

def block (application : Stage1.Application.Program) (certificate : Certificate application) :
    LowNormBlock.Block (ApplicationDirectSource.sourceWidth application) :=
  PoseidonRetainedBlock.block (ApplicationDirectSource.sourceWidth application) 3
    (witnessStart application) (witnessStart_bound application certificate)

theorem block_slotCount (application : Stage1.Application.Program)
    (certificate : Certificate application) : (block application certificate).slotCount = 258 := by
  rw [block, PoseidonRetainedBlock.block_slotCount]

theorem block_coordinateCount (application : Stage1.Application.Program)
    (certificate : Certificate application) :
    (block application certificate).coordinateCount = 10578 := by
  rw [block, PoseidonRetainedBlock.block_coordinateCount]

end NightstreamFPrime.Export.Stage1.ApplicationPoseidonRetainedBlock
