import NightstreamFPrime.Export.Stage1.Wide.PhysicalRelabel
import NightstreamFPrime.Export.Stage1.Wide.PhysicalSampler
import NightstreamFPrime.Export.Stage1.PermutationPlan

/-! Build the physical prefix with the total wide sampler. Common phases use
checked relocation; removed sampler rows, hints and templates are not emitted. -/

namespace NightstreamFPrime.Export.Stage1.Wide.PhysicalPackage

open NightstreamFPrime.Export.Package NightstreamFPrime.Layout

def common (_unit : Unit) : CircuitPackage :=
  let rows := PiCCSArithmetic.arithmeticRows Data.logicalWidth Data.publicFits ++
    (PiDECArithmetic.canonicalPlan Data.logicalWidth Data.publicFits).rows ++
    (RunningTransitionArithmetic.canonicalPlan Data.logicalWidth Data.publicFits).rows
  Data.circuitPackageOf rows ((PermutationPlan.piCcsBlocks ()).flatMap PermutationPlan.Block.expand)
    PiRLCCombinationInvocations.invocations
    (WitnessProgram.piCcsBatches Data.logicalWidth Data.publicFits ++
      WitnessProgram.piDecBatches Data.logicalWidth Data.publicFits ++
      WitnessProgram.directRunningTransitionBatches Data.logicalWidth Data.publicFits)

def ofCommon (base : CircuitPackage) : Except String CircuitPackage := do
  let layout : PhysicalLayout := { base.layout with
    rowCount := ← PhysicalRelabel.row base.layout.rowCount
    privateColumnCount := Layout.Stage1.Wide.SourceOrder.privateColumns
    constantColumn := Layout.Stage1.Wide.SourceOrder.constantColumn
    totalColumnCount := Layout.Stage1.Wide.SourceOrder.totalColumns
    privateSegments := ← base.layout.privateSegments.mapM PhysicalRelabel.prefixMap.segment
    publicSegments := ← base.layout.publicSegments.mapM PhysicalRelabel.prefixMap.segment }
  let rangeRows := PhysicalSampler.rows ()
  let batches ← base.witnessBatches.mapM PhysicalRelabel.prefixMap.batch
  let (earlierBatches, laterBatches) := batches.span (fun item =>
    item.start < Layout.Stage1.Wide.SourceOrder.column Layout.Stage1.Wide.PiRLCStarts.samplerLogicalStart)
  let instructions ← base.witnessInstructions.mapM PhysicalRelabel.prefixMap.instruction
  let (earlierInstructions, laterInstructions) := instructions.span
    (fun item => item.rowIndex < Layout.Stage1.Wide.PiRLCStarts.samplerRowStart)
  let assertions ← base.assertionRows.mapM PhysicalRelabel.prefixMap.assertion
  let (earlierAssertions, laterAssertions) := assertions.span
    (fun item => item.rowIndex < Layout.Stage1.Wide.PiRLCStarts.samplerRowStart)
  return { base with
    layout := layout
    relation := productionCcsRelation layout.rowCount layout.totalColumnCount Lifecycle.cubeVariables
    hashChains := ← base.hashChains.mapM PhysicalRelabel.prefixMap.chain
    permutationInvocations := (← base.permutationInvocations.mapM PhysicalRelabel.prefixMap.permutation) ++
      (← PhysicalSampler.permutations ())
    compactRowTemplates := PiRLCCombinationTemplates.templates
    compactRowInvocations := ← base.compactRowInvocations.mapM PhysicalRelabel.prefixMap.compact
    witnessBatches := earlierBatches ++ PhysicalSampler.batches () ++ laterBatches
    witnessInstructions := earlierInstructions ++ Rows.witnessInstructionsTR rangeRows ++ laterInstructions
    assertionRows := earlierAssertions ++ Rows.assertionRowsTR rangeRows ++ laterAssertions }

def circuitPackage (_unit : Unit) : Except String CircuitPackage := ofCommon (common ())

end NightstreamFPrime.Export.Stage1.Wide.PhysicalPackage
