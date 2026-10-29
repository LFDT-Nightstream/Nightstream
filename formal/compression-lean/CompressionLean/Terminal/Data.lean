import NightstreamFPrime.Export.Stage1.PerApplicationPackage

namespace CompressionLean.Terminal

open NightstreamFPrime.Export
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Export.Stage1

def arithmeticRows (_unit : Unit) : List Rows.CompiledRow :=
  PiCCSArithmetic.arithmeticRows Data.logicalWidth Data.publicFits ++
    PiRLCSamplerOrdinaryRows.rows (logicalWidth := Data.logicalWidth)
      (publicFits := Data.publicFits)

def priorAssertions (_unit : Unit) : List SparseRow :=
  Data.liftPilotRows (Rows.assertionRowsTR (PilotData.priorExtraRows ()))

def physicalLayout : PhysicalLayout :=
  { Data.physicalLayout with
    privateSegments := Data.physicalLayout.privateSegments ++
      [⟨PerApplicationPackage.Role.applicationWitness,
          Data.physicalLayout.constantColumn, 0⟩,
        ⟨PerApplicationPackage.Role.applicationLocal,
          Data.physicalLayout.constantColumn, 0⟩] }

def circuitPackage (_unit : Unit) : CircuitPackage where
  schemaVersion := 8
  profile := PilotData.profile
  poseidon := PilotData.poseidonSchedule
  layout := physicalLayout
  relation := productionCcsRelation physicalLayout.rowCount
    physicalLayout.totalColumnCount NightstreamFPrime.Lifecycle.cubeVariables
  permutation := PilotData.permutationTemplate ()
  hashChains := [Data.priorChain]
  permutationInvocations := Data.permutationInvocations ()
  compactRowTemplates := Data.compactRowTemplates ()
  compactRowInvocations := Data.compactRowInvocations ()
  witnessBatches :=
    Data.liftPilotBatches (PilotData.priorWordBatches ()) ++
      WitnessProgram.piCcsBatches Data.logicalWidth Data.publicFits ++
      WitnessProgram.piRlcSamplerBatches Data.logicalWidth Data.publicFits
  witnessInstructions :=
    Data.liftPilotInstructions (PilotData.witnessInstructions ()) ++
      Rows.witnessInstructionsTR (arithmeticRows ())
  assertionRows := priorAssertions () ++ Rows.assertionRowsTR (arithmeticRows ())
  terminal := none

def artifact (_unit : Unit) : Artifact :=
  sealPackage (circuitPackage ())

end CompressionLean.Terminal
