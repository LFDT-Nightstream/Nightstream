import NightstreamFPrime.Export.Main
import NightstreamFPrime.Export.RowSemantics

/-!
Typed execution inputs assembled from the selected per-application emitter's
existing builders. Only invocation records, witness batches/instructions and
explicit assertion rows are retained. No permutation/compact row expansion,
caller parsing, witness execution or whole-plan refinement is claimed here.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.StoredPhysicalPlan

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package

inductive Event where
  | hash (chain : HashChain) (ordinal : Nat)
  | permutation (invocation : PermutationInvocation)
  | compact (outputTarget : Nat) (invocation : CompactRowInvocation)
  | batch (value : WitnessBatch)
  | instruction (value : WitnessInstruction)

/-- The native assignment schedule orders these same global write targets. -/
def Event.target : Event → Nat
  | .hash chain ordinal => chain.witnessStart +
      ordinal * PilotData.poseidonSchedule.recipesPerPermutation
  | .permutation invocation => invocation.witnessStart
  | .compact outputTarget _ => outputTarget
  | .batch value => value.start
  | .instruction value => value.target

structure Plan where
  layout : PhysicalLayout
  templates : Array CompactRowTemplate
  events : Array Event
  assertions : Array SparseRow

/-- Prepare the exact selected sources once. The seven packet groups and
ordinary blocks are the emitter's shared classified values. Assertions keep
writePerApplicationAssertionRows order; only execution events are sorted. -/
def prepare : IO Plan := do
  let program := Poseidon2HashChainV1Package.application
  let shift := PerApplicationCachedShift.Context.ofProgram program
  let application := PerApplicationPackage.directApplicationPlan program
  let permutationTask ← IO.asTask Main.preparePermutationBlocks
  let witnessTasks ← Main.prepareWitnessGroups
  let rowTasks ← Main.prepareRowBlocks
  let permutationBlocks ← Main.preparedPermutationBlocks permutationTask
  let groups ← [witnessTasks.initialClaim, witnessTasks.sumcheck,
    witnessTasks.evalK, witnessTasks.evalA, witnessTasks.ccs, witnessTasks.norm,
    witnessTasks.finalIdentity].mapM Main.preparedWitnessGroup
  let statementBinding ← Main.preparedRowBlock rowTasks.statementBinding
  let piRlcSources ← rowTasks.piRlcSources.mapM Main.preparedRowSource
  let lastRows ← [rowTasks.piDec, rowTasks.runningTransition].mapM Main.preparedRowBlock
  let templates := (Data.compactRowTemplates ()).toArray
  let mut events : Array Event := #[]

  -- Main.writePerApplicationInnerPackage: shifted prior/output chains,
  -- expanded invocation records, and compact invocation records.
  for chain in [Data.priorChain, Data.outputChain] do
    let shifted := PerApplicationCachedShift.shiftHashChain shift chain
    for ordinal in [:shifted.absorbCount + 1] do
      events := events.push (.hash shifted ordinal)
  for block in permutationBlocks.blocks do
    for invocation in block.expand do
      events := events.push (.permutation
        (PerApplicationCachedShift.shiftPermutationInvocation shift invocation))
  for invocation in application.permutationInvocations do
    events := events.push (.permutation invocation)
  for block in PackagePlan.canonicalCompactBlocks do
    for invocation in block.expand do
      let shifted := PerApplicationCachedShift.shiftCompactRowInvocation shift invocation
      let some template := templates[shifted.templateIndex]?
        | throw (IO.userError "canonical compact invocation has no template")
      let target := compactInputColumn shifted.inputRanges template.outputInput
      events := events.push (.compact target shifted)

  -- Main.writePerApplicationWitnessBatches.
  for batch in Data.liftPilotBatches (PilotData.priorWordBatches ()) do
    events := events.push (.batch (PerApplicationCachedShift.shiftBatch shift batch))
  for group in groups do
    for batch in group.batches do
      events := events.push (.batch (PerApplicationCachedShift.shiftBatch shift batch))
  for block in WitnessPlan.canonicalBlocks Data.logicalWidth Data.publicFits do
    for batch in block.expand do
      events := events.push (.batch (PerApplicationCachedShift.shiftBatch shift batch))
  for batch in application.witnessBatches do
    events := events.push (.batch batch)

  -- Main.writePerApplicationWitnessInstructions.
  for instruction in Data.liftPilotInstructions (PilotData.witnessInstructions ()) do
    events := events.push (.instruction
      (PerApplicationCachedShift.shiftWitnessInstruction shift instruction))
  for instruction in statementBinding.witnessInstructions do
    events := events.push (.instruction
      (PerApplicationCachedShift.shiftWitnessInstruction shift instruction))
  for group in groups do
    for instruction in group.witnessInstructions do
      events := events.push (.instruction
        (PerApplicationCachedShift.shiftWitnessInstruction shift instruction))
  for blocks in piRlcSources do
    for block in blocks do
      for instruction in block.witnessInstructions do
        events := events.push (.instruction
          (PerApplicationCachedShift.shiftWitnessInstruction shift instruction))
  for block in lastRows do
    for instruction in block.witnessInstructions do
      events := events.push (.instruction
        (PerApplicationCachedShift.shiftWitnessInstruction shift instruction))
  for instruction in application.witnessInstructions do
    events := events.push (.instruction instruction)

  -- Main.writePerApplicationAssertionRows, without unrelated row expansion.
  let mut assertions : Array SparseRow := #[]
  for row in Data.liftPilotRows (PilotData.assertionRows ()) do
    assertions := assertions.push (PerApplicationCachedShift.shiftSparseRow shift row)
  for row in statementBinding.assertionRows do
    assertions := assertions.push (PerApplicationCachedShift.shiftSparseRow shift row)
  for group in groups do
    for row in group.assertionRows do
      assertions := assertions.push (PerApplicationCachedShift.shiftSparseRow shift row)
  for blocks in piRlcSources do
    for block in blocks do
      for row in block.assertionRows do
        assertions := assertions.push (PerApplicationCachedShift.shiftSparseRow shift row)
  for block in lastRows do
    for row in block.assertionRows do
      assertions := assertions.push (PerApplicationCachedShift.shiftSparseRow shift row)
  for row in application.assertionRows do
    assertions := assertions.push row
  for row in NextPreimagePackage.assertionRows
      (PerApplicationPackage.nextPreimageRowStart program) do
    assertions := assertions.push row

  return {
    layout := PerApplicationPackage.directFinalLayout program
    templates := templates
    events := events.qsort (fun left right => decide (left.target < right.target))
    assertions := assertions }

end NightstreamFPrime.Export.Stage1.StoredPhysicalPlan
