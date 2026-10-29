import NightstreamFPrime.Export.Stage1.PreparedPhysicalSources

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PreparedPhysicalSourceTasks

open PreparedPhysicalSources

def collect {Value : Type} (task : Task (Except IO.Error Value)) : IO Value := do
  match ← IO.wait task with
  | .ok value => return value
  | .error error => throw error

def startWitnessGroups : IO PreparedWitnessTasks := do
  let initialClaim ← IO.asTask do
    return prepareWitnessGroup fun _ =>
      PiCCSPackets.initialClaim Data.logicalWidth Data.publicFits
  let sumcheck ← IO.asTask do
    return prepareWitnessGroup fun _ =>
      PiCCSPackets.sumcheck Data.logicalWidth Data.publicFits
  let evalK ← IO.asTask do
    return prepareWitnessGroup fun _ =>
      PiCCSPackets.evalK Data.logicalWidth Data.publicFits
  let evalA ← IO.asTask do
    return prepareWitnessGroup fun _ =>
      PiCCSPackets.evalA Data.logicalWidth Data.publicFits
  let ccs ← IO.asTask do
    return prepareWitnessGroup fun _ =>
      PiCCSPackets.ccs Data.logicalWidth Data.publicFits
  let norm ← IO.asTask do
    return prepareWitnessGroup fun _ =>
      PiCCSPackets.norm Data.logicalWidth Data.publicFits
  let finalIdentity ← IO.asTask do
    return prepareWitnessGroup fun _ =>
      PiCCSPackets.finalIdentity Data.logicalWidth Data.publicFits
  return { initialClaim, sumcheck, evalK, evalA, ccs, norm, finalIdentity }

def collectWitnessGroups (tasks : PreparedWitnessTasks) : IO PreparedWitnessGroups := do
  let initialClaim ← collect tasks.initialClaim
  let sumcheck ← collect tasks.sumcheck
  let evalK ← collect tasks.evalK
  let evalA ← collect tasks.evalA
  let ccs ← collect tasks.ccs
  let norm ← collect tasks.norm
  let finalIdentity ← collect tasks.finalIdentity
  return { initialClaim, sumcheck, evalK, evalA, ccs, norm, finalIdentity }

def startPermutationBlocks :
    IO (Task (Except IO.Error PreparedPermutationBlocks)) :=
  IO.asTask do
    return preparePermutationBlocks ()

def startStatementBinding : IO (PreparedRowTask .statementBinding) :=
  IO.asTask do
    return prepareRowBlock .statementBinding

def startPiDec : IO (PreparedRowTask (OrdinaryRowPlan.piDecBlock ())) :=
  IO.asTask do
    let source := OrdinaryRowPlan.piDecBlock ()
    return prepareRowBlock source

def startRunningTransition :
    IO (PreparedRowTask (OrdinaryRowPlan.runningTransitionBlock ())) :=
  IO.asTask do
    let source := OrdinaryRowPlan.runningTransitionBlock ()
    return prepareRowBlock source

def preparePiRlcSources :
    (sources : List Nat) →
      IO (PreparedRowBlocks (sources.flatMap OrdinaryRowPlan.piRlcSourceBlocks))
  | [] => return { blocks := [], rows_eq := rfl }
  | source :: rest => do
      let headTask : PreparedPiRlcSourceTask source ← IO.asTask
        (do return preparePiRlcSource source)
        (prio := Task.Priority.dedicated)
      let tail ← preparePiRlcSources rest
      let head ← collect headTask
      return head.append tail

def prepareAllPiRlc :
    IO (PreparedRowBlocks (OrdinaryRowPlan.piRlcBlocks ())) :=
  preparePiRlcSources (List.range PiRLCSamplerInvocations.sourceCount)

def startAllPiRlc :
    IO (Task (Except IO.Error (PreparedRowBlocks (OrdinaryRowPlan.piRlcBlocks ())))) :=
  IO.asTask prepareAllPiRlc

end NightstreamFPrime.Export.Stage1.PreparedPhysicalSourceTasks
