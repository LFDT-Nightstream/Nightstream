import NightstreamFPrime.Export.Stage1.PerApplicationPackage

/-!
Owns the terminal circuit package: the rows of the F′ Stage 1 package that
check the last fold before Π_DEC.

Every component is an F′ Stage 1 component, unchanged: the prior-state hash
chain and its pilot rows, the PiCCS rows and permutations, and the PiRLC
sampler and combination rows. The package keeps F′'s columns and row numbers.
It drops the output hash chain and its digest rows, PiDEC, and the running
transition. It adds no row, column, or template.
-/

namespace CompressionLean.Terminal

open NightstreamFPrime.Export
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Export.Stage1

/-- The ordinary rows of PiCCS and of the PiRLC sampler, in F′ order. -/
def arithmeticRows (_unit : Unit) : List Rows.CompiledRow :=
  PiCCSArithmetic.arithmeticRows Data.logicalWidth Data.publicFits ++
    PiRLCSamplerOrdinaryRows.rows (logicalWidth := Data.logicalWidth)
      (publicFits := Data.publicFits)

/-- The pilot assertions of the prior-state hash, without the output digest
rows. -/
def priorAssertions (_unit : Unit) : List SparseRow :=
  Data.liftPilotRows (Rows.assertionRowsTR (PilotData.priorExtraRows ()))

/-- F′'s Stage 1 layout with the two application segments of a production
package, both empty. -/
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
