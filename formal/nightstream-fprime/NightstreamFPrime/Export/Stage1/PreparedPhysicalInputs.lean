import NightstreamFPrime.Export.Stage1.PreparedPhysicalSourceTasks
import NightstreamFPrime.Export.Stage1.PreparedPhysicalOrdinary

/-! Collect the canonical physical sources without discarding their source proofs. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PreparedPhysicalInputs

open NightstreamFPrime.Export.Package
open PreparedPhysicalSources
open PreparedPhysicalSourceTasks

structure Inputs where
  pilot : { value : CircuitPackage // value = PilotData.circuitPackage () }
  permutations : PreparedPermutationBlocks
  groups : PreparedWitnessGroups
  ordinary : PreparedRowBlocks (OrdinaryRowPlan.canonicalBlocks ())

def prepare : IO Inputs := do
  let pilotTask ← IO.asTask do
    return (⟨PilotData.circuitPackage (), rfl⟩ :
      { value : CircuitPackage // value = PilotData.circuitPackage () })
  let permutationTask ← startPermutationBlocks
  let witnessTasks ← startWitnessGroups
  let statementTask ← startStatementBinding
  let piRlcTask ← startAllPiRlc
  let piDecTask ← startPiDec
  let runningTask ← startRunningTransition
  let pilot ← collect pilotTask
  let permutations ← collect permutationTask
  let groups ← collectWitnessGroups witnessTasks
  let statement ← collect statementTask
  let piRlc ← collect piRlcTask
  let piDec ← collect piDecTask
  let running ← collect runningTask
  let ordinary := PreparedPhysicalOrdinary.assemble statement groups piRlc piDec running
  return { pilot, permutations, groups, ordinary }

end NightstreamFPrime.Export.Stage1.PreparedPhysicalInputs
