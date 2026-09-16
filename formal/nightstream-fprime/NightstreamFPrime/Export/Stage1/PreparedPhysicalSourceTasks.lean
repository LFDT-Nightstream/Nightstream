import NightstreamFPrime.Export.Stage1.PreparedPhysicalSources

/-!
Task results retain the checked source indices.
The runtime schedules the existing independent work; collection preserves
canonical source order. This module does not change emitter records or prove
whole physical-plan coverage.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PreparedPhysicalSourceTasks

open PreparedPhysicalSources

/-- Collect an indexed task result without replacing its provenance fields. -/
def collect {Value : Type} (task : Task (Except IO.Error Value)) : IO Value := do
  match ← IO.wait task with
  | .ok value => return value
  | .error error => throw error

/-- Start the same seven independent packet builders with distinct result types. -/
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

/-- Preserve all seven builder indices when the tasks become stored groups. -/
def collectWitnessGroups (tasks : PreparedWitnessTasks) : IO PreparedWitnessGroups := do
  let initialClaim ← collect tasks.initialClaim
  let sumcheck ← collect tasks.sumcheck
  let evalK ← collect tasks.evalK
  let evalA ← collect tasks.evalA
  let ccs ← collect tasks.ccs
  let norm ← collect tasks.norm
  let finalIdentity ← collect tasks.finalIdentity
  return { initialClaim, sumcheck, evalK, evalA, ccs, norm, finalIdentity }

/-- The canonical permutation list is constructed within its existing task. -/
def startPermutationBlocks :
    IO (Task (Except IO.Error PreparedPermutationBlocks)) :=
  IO.asTask do
    return preparePermutationBlocks ()

def startStatementBinding : IO (PreparedRowTask .statementBinding) :=
  IO.asTask do
    return prepareRowBlock .statementBinding

/-- Keep construction of the complete PiDEC row source inside the task. -/
def startPiDec : IO (PreparedRowTask (OrdinaryRowPlan.piDecBlock ())) :=
  IO.asTask do
    let source := OrdinaryRowPlan.piDecBlock ()
    return prepareRowBlock source

/-- Keep construction of the running-transition row source inside the task. -/
def startRunningTransition :
    IO (PreparedRowTask (OrdinaryRowPlan.runningTransitionBlock ())) :=
  IO.asTask do
    let source := OrdinaryRowPlan.runningTransitionBlock ()
    return prepareRowBlock source

/-- Start every source computation before collecting the recursive tail.
Appending the checked head before the tail fixes the returned block order.
Each source retains the dedicated priority used by the existing emitter. -/
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

/-- All original PiRLC sources, including each final selector block, retain
the existing canonical list index. The source count comes from its owner. -/
def prepareAllPiRlc :
    IO (PreparedRowBlocks (OrdinaryRowPlan.piRlcBlocks ())) :=
  preparePiRlcSources (List.range PiRLCSamplerOrdinaryRows.sourceCount)

/-- Let the per-source preparation overlap with the other physical sources. -/
def startAllPiRlc :
    IO (Task (Except IO.Error (PreparedRowBlocks (OrdinaryRowPlan.piRlcBlocks ())))) :=
  IO.asTask prepareAllPiRlc

end NightstreamFPrime.Export.Stage1.PreparedPhysicalSourceTasks
