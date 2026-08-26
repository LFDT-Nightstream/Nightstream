import NightstreamFPrime.Export.PilotData
import NightstreamFPrime.Export.Stage1.PiCCSArithmetic
import NightstreamFPrime.Export.Stage1.PiCCSInvocations
import NightstreamFPrime.Export.Stage1.PiRLCCombinationInvocations
import NightstreamFPrime.Export.Stage1.PiRLCCombinationTemplates
import NightstreamFPrime.Export.Stage1.PiRLCFirst54Invocations
import NightstreamFPrime.Export.Stage1.PiRLCFirst54Templates
import NightstreamFPrime.Export.Stage1.PiRLCSamplerInvocations
import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryRows
import NightstreamFPrime.Export.Stage1.WitnessProgram

/-!
Owns the executable data of the one Stage 1 pilot + PiCCS + PiRLC package.

The closed pilot is lifted into the combined Spartan column order. The PiCCS
proof-input segment follows the two pilot preimages. All remaining private
columns belong to the one package witness program. The PiCCS transcript uses
compact Poseidon2 invocations. PiRLC reuses that permutation template, uses
compact `First54` recipes, keeps decoder rows ordinary, and then emits the
four compact 17-input combination families in parent order.
-/

namespace NightstreamFPrime.Export.Stage1.Data

open NightstreamFPrime.Export.Package
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

namespace Role

def priorPreimage : Nat := PilotData.Role.priorPreimage
def outputPreimage : Nat := PilotData.Role.outputPreimage
def piCcsFreshCommitment : Nat := 6
def piCcsRoundMessages : Nat := 7
def piCcsOutputEval_K : Nat := 8
def piCcsOutputEval_A : Nat := 9
def witness : Nat := PilotData.Role.witness
def priorPublicInput : Nat := PilotData.Role.priorPublicInput
def outputDigest : Nat := PilotData.Role.outputDigest
def verifierContext : Nat := 10

end Role

/-- Current prefix width used only to instantiate width-erased package data. -/
def logicalWidth : Nat :=
  NightstreamFPrime.Layout.Stage1.Spartan.spartanColumnCount

def publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth := by
  apply Nat.le_trans (m := logicalWidth)
  · norm_num [logicalWidth,
      NightstreamFPrime.Layout.Stage1.Spartan.spartanColumnCount, ringDegree,
      publicRingColumns]
  · exact Phi81CarrierLayout.logicalWidth_le_carrierWidth logicalWidth

def liftPilotTerm (term : SparseTerm) : SparseTerm :=
  ⟨NightstreamFPrime.Layout.Stage1.Spartan.liftPilotColumn term.column,
    term.coefficient⟩

def liftPilotCombination (combination : SparseCombination) :
    SparseCombination :=
  ⟨combination.constant, combination.terms.map liftPilotTerm⟩

def liftPilotRow (row : SparseRow) : SparseRow :=
  ⟨row.rowIndex, liftPilotCombination row.a,
    liftPilotCombination row.b, liftPilotCombination row.c⟩

def liftPilotRows (rows : List SparseRow) : List SparseRow :=
  rows.map liftPilotRow

def liftPilotChain (chain : HashChain) : HashChain :=
  { chain with
    inputStart :=
      NightstreamFPrime.Layout.Stage1.Spartan.liftPilotColumn chain.inputStart
    witnessStart :=
      NightstreamFPrime.Layout.Stage1.Spartan.liftPilotColumn chain.witnessStart
    digestStart :=
      NightstreamFPrime.Layout.Stage1.Spartan.liftPilotColumn chain.digestStart }

def priorChain : HashChain := liftPilotChain PilotData.priorChain
def outputChain : HashChain := liftPilotChain PilotData.outputChain

def proofInputStart : Nat :=
  NightstreamFPrime.Layout.Stage1.Spartan.pilotInputPrivateColumnCount

def witnessStart : Nat :=
  proofInputStart +
    NightstreamFPrime.Layout.Stage1.Spartan.proofInputColumnCount

def witnessLength : Nat :=
  NightstreamFPrime.Layout.Stage1.Spartan.privateColumnCount - witnessStart

def outputEval_KWords : Nat :=
  productionShape.coefficientCount * 2

def outputEval_AWords : Nat :=
  productionShape.matrixCount * productionShape.coefficientCount * 2

def outputEvaluationWordsPerSource : Nat :=
  outputEval_KWords + outputEval_AWords

def outputEvaluationTargetStart : Nat :=
  proofInputStart +
    NightstreamFPrime.Layout.Stage1.PiCCSInputs.freshCommitmentWords +
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.roundMessageWords

/-- One exact v1_1 Pad-evaluation segment for one PiCCS output source. -/
def outputEval_KSegment
    (source : Fin productionShape.sourceCount) : Segment :=
  ⟨Role.piCcsOutputEval_K,
    outputEvaluationTargetStart +
      source.val * outputEvaluationWordsPerSource,
    outputEval_KWords⟩

/-- One exact v1_1 CCS-matrix-evaluation segment for one PiCCS output source. -/
def outputEval_ASegment
    (source : Fin productionShape.sourceCount) : Segment :=
  ⟨Role.piCcsOutputEval_A,
    outputEvaluationTargetStart +
      source.val * outputEvaluationWordsPerSource + outputEval_KWords,
    outputEval_AWords⟩

/-- The 17 output claims remain in source order. Within each source, the
separate Pad family precedes the separate 14-matrix family. -/
def piCcsOutputSegments : List Segment :=
  (List.finRange productionShape.sourceCount).flatMap fun source =>
    [outputEval_KSegment source, outputEval_ASegment source]

def privateSegments : List Segment :=
  [⟨Role.priorPreimage, 0,
      NightstreamFPrime.Layout.PilotValues.stateHashWords⟩,
   ⟨Role.outputPreimage,
      NightstreamFPrime.Layout.PilotValues.stateHashWords,
      NightstreamFPrime.Layout.PilotValues.stateHashWords⟩,
   ⟨Role.piCcsFreshCommitment, proofInputStart,
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.freshCommitmentWords⟩,
   ⟨Role.piCcsRoundMessages,
      proofInputStart +
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.freshCommitmentWords,
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.roundMessageWords⟩] ++
    piCcsOutputSegments ++
      [⟨Role.witness, witnessStart, witnessLength⟩]

@[simp] theorem piCcsOutputSegments_length :
    piCcsOutputSegments.length = 34 := by
  norm_num [piCcsOutputSegments, productionShape, productionProfile,
    NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Shape.sourceCount,
    NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81MatrixSource.phi81Shape]

theorem outputEval_KSegment_mem
    (source : Fin productionShape.sourceCount) :
    outputEval_KSegment source ∈ piCcsOutputSegments := by
  unfold piCcsOutputSegments
  apply List.mem_flatMap.mpr
  exact ⟨source, by simp, by simp⟩

theorem outputEval_ASegment_mem
    (source : Fin productionShape.sourceCount) :
    outputEval_ASegment source ∈ piCcsOutputSegments := by
  unfold piCcsOutputSegments
  apply List.mem_flatMap.mpr
  exact ⟨source, by simp, by simp⟩

def publicSegments : List Segment :=
  [⟨Role.priorPublicInput,
      NightstreamFPrime.Layout.Stage1.Spartan.constantColumn + 1,
      NightstreamFPrime.Layout.PilotValues.priorPublicInputWords⟩,
   ⟨Role.outputDigest,
      NightstreamFPrime.Layout.Stage1.Spartan.constantColumn + 1 +
        NightstreamFPrime.Layout.PilotValues.priorPublicInputWords,
      NightstreamFPrime.Layout.PilotValues.digestWords⟩,
   ⟨Role.verifierContext,
      NightstreamFPrime.Layout.Stage1.Spartan.expectedContextPublicStart,
      NightstreamFPrime.Layout.Stage1.Spartan.expectedContextColumnCount⟩]

def physicalLayout : PhysicalLayout where
  rowCount := 25556958
  privateColumnCount :=
    NightstreamFPrime.Layout.Stage1.Spartan.privateColumnCount
  constantColumn := NightstreamFPrime.Layout.Stage1.Spartan.constantColumn
  publicColumnCount :=
    NightstreamFPrime.Layout.Stage1.Spartan.publicColumnCount
  totalColumnCount :=
    NightstreamFPrime.Layout.Stage1.Spartan.spartanColumnCount
  privateSegments := privateSegments
  publicSegments := publicSegments

def arithmeticRows (_unit : Unit) : List Rows.CompiledRow :=
  PiCCSArithmetic.arithmeticRows logicalWidth publicFits ++
    PiRLCSamplerOrdinaryRows.rows (logicalWidth := logicalWidth)
      (publicFits := publicFits)

def permutationInvocations (_unit : Unit) : List PermutationInvocation :=
  PiCCSInvocations.invocations logicalWidth publicFits ++
    PiRLCSamplerInvocations.invocations
      (logicalWidth := logicalWidth) (publicFits := publicFits)

def compactRowTemplates (_unit : Unit) : List CompactRowTemplate :=
  PiRLCCombinationTemplates.templates ++ PiRLCFirst54Templates.templates

def compactRowInvocations (_unit : Unit) : List CompactRowInvocation :=
  PiRLCFirst54Invocations.invocations ++
    PiRLCCombinationInvocations.invocations

theorem arithmeticRows_eq :
    arithmeticRows () =
      PiCCSArithmetic.arithmeticRows logicalWidth publicFits ++
        PiRLCSamplerOrdinaryRows.rows (logicalWidth := logicalWidth)
          (publicFits := publicFits) := by
  rfl

theorem permutationInvocations_eq :
    permutationInvocations () =
      PiCCSInvocations.invocations logicalWidth publicFits ++
        PiRLCSamplerInvocations.invocations
          (logicalWidth := logicalWidth) (publicFits := publicFits) := by
  rfl

theorem compactRowTemplates_eq :
    compactRowTemplates () =
      PiRLCCombinationTemplates.templates ++
        PiRLCFirst54Templates.templates := by
  rfl

theorem compactRowInvocations_eq :
    compactRowInvocations () =
      PiRLCFirst54Invocations.invocations ++
        PiRLCCombinationInvocations.invocations := by
  rfl

structure Components where
  arithmeticRows : List Rows.CompiledRow
  permutationInvocations : List PermutationInvocation

def Components.of (arithmeticRows : List Rows.CompiledRow)
    (permutationInvocations : List PermutationInvocation) : Components :=
  ⟨arithmeticRows, permutationInvocations⟩

theorem Components.of_arithmeticRows
    (arithmeticRows : List Rows.CompiledRow)
    (permutationInvocations : List PermutationInvocation) :
    (Components.of arithmeticRows permutationInvocations).arithmeticRows =
      arithmeticRows := by
  rfl

theorem Components.of_permutationInvocations
    (arithmeticRows : List Rows.CompiledRow)
    (permutationInvocations : List PermutationInvocation) :
    (Components.of arithmeticRows
      permutationInvocations).permutationInvocations =
        permutationInvocations := by
  rfl

def Components.witnessInstructions (components : Components) :
    List WitnessInstruction :=
  Rows.witnessInstructionsTR components.arithmeticRows

def Components.arithmeticAssertionRows (components : Components) :
    List SparseRow :=
  Rows.assertionRowsTR components.arithmeticRows

def Components.assertionRows (components : Components) : List SparseRow :=
  liftPilotRows (PilotData.assertionRows ()) ++
    components.arithmeticAssertionRows

def Components.toCircuitPackage (components : Components) : CircuitPackage where
  schemaVersion := 7
  profile := PilotData.profile
  poseidon := PilotData.poseidonSchedule
  layout := physicalLayout
  relation := productionCcsRelation physicalLayout.rowCount
    physicalLayout.totalColumnCount Lifecycle.cubeVariables
  permutation := PilotData.permutationTemplate ()
  hashChains := [priorChain, outputChain]
  permutationInvocations := components.permutationInvocations
  compactRowTemplates := compactRowTemplates ()
  compactRowInvocations := compactRowInvocations ()
  witnessBatches := WitnessProgram.batches logicalWidth publicFits
  witnessInstructions := components.witnessInstructions
  assertionRows := components.assertionRows
  terminal := none

theorem Components.toCircuitPackage_layout (components : Components) :
    components.toCircuitPackage.layout = physicalLayout := by
  rfl

theorem Components.toCircuitPackage_relation (components : Components) :
    components.toCircuitPackage.relation =
      productionCcsRelation physicalLayout.rowCount
        physicalLayout.totalColumnCount Lifecycle.cubeVariables := by
  rfl

theorem Components.toCircuitPackage_permutation (components : Components) :
    components.toCircuitPackage.permutation =
      PilotData.permutationTemplate () := by
  rfl

theorem Components.toCircuitPackage_hashChains (components : Components) :
    components.toCircuitPackage.hashChains = [priorChain, outputChain] := by
  rfl

theorem Components.toCircuitPackage_permutationInvocations
    (components : Components) :
    components.toCircuitPackage.permutationInvocations =
      components.permutationInvocations := by
  rfl

theorem Components.toCircuitPackage_compactRowTemplates
    (components : Components) :
    components.toCircuitPackage.compactRowTemplates =
      compactRowTemplates () := by
  rfl

theorem Components.toCircuitPackage_compactRowInvocations
    (components : Components) :
    components.toCircuitPackage.compactRowInvocations =
      compactRowInvocations () := by
  rfl

theorem Components.toCircuitPackage_witnessInstructions
    (components : Components) :
    components.toCircuitPackage.witnessInstructions =
      components.witnessInstructions := by
  rfl

theorem Components.toCircuitPackage_witnessBatches
    (components : Components) :
    components.toCircuitPackage.witnessBatches =
      WitnessProgram.batches logicalWidth publicFits := by
  rfl

theorem Components.toCircuitPackage_assertionRows (components : Components) :
    components.toCircuitPackage.assertionRows = components.assertionRows := by
  rfl

/-- The generic assembler classifies every arithmetic row exactly once. -/
theorem Components.ordinaryRows_length (components : Components) :
    components.toCircuitPackage.witnessInstructions.length +
      components.toCircuitPackage.assertionRows.length =
        (liftPilotRows (PilotData.assertionRows ())).length +
          components.arithmeticRows.length := by
  rw [components.toCircuitPackage_witnessInstructions,
    components.toCircuitPackage_assertionRows]
  unfold Components.witnessInstructions Components.assertionRows
    Components.arithmeticAssertionRows
  rw [List.length_append]
  rw [Rows.witnessInstructionsTR_eq, Rows.assertionRowsTR_eq]
  have partition :=
    Rows.witnessInstructions_length_add_assertionRows_length
      components.arithmeticRows
  calc
    _ = (liftPilotRows (PilotData.assertionRows ())).length +
        ((Rows.witnessInstructions components.arithmeticRows).length +
          (Rows.assertionRows components.arithmeticRows).length) := by
      omega
    _ = (liftPilotRows (PilotData.assertionRows ())).length +
        components.arithmeticRows.length := by rw [partition]

/-- Exact total row coverage for any component lists with the production
counts. This theorem never inspects a concrete component list. -/
theorem Components.rowCoverage (components : Components)
    (arithmeticRows_length : components.arithmeticRows.length = 986251)
    (permutationInvocations_length :
      components.permutationInvocations.length = 7613)
    (templateRows_length :
      (PilotData.permutationTemplate ()).rows.length = 592)
    (pilotAssertionRows_length :
      (liftPilotRows (PilotData.assertionRows ())).length = 58)
    (hashChainRows :
      priorChain.witnessLength + outputChain.witnessLength = 12574080)
    (compactRows_length :
      components.toCircuitPackage.compactRowCount = 7489673) :
    (components.toCircuitPackage.hashChains.map
        (fun chain => chain.witnessLength)).sum +
      components.toCircuitPackage.permutationInvocations.length *
        components.toCircuitPackage.permutation.rows.length +
      components.toCircuitPackage.compactRowCount +
      components.toCircuitPackage.witnessInstructions.length +
      components.toCircuitPackage.assertionRows.length =
        components.toCircuitPackage.layout.rowCount := by
  have ordinaryFixed :
      components.toCircuitPackage.witnessInstructions.length +
        components.toCircuitPackage.assertionRows.length = 986309 := by
    calc
      _ = (liftPilotRows (PilotData.assertionRows ())).length +
          components.arithmeticRows.length :=
        components.ordinaryRows_length
      _ = 58 + 986251 := by
        rw [pilotAssertionRows_length, arithmeticRows_length]
      _ = 986309 := by norm_num
  rw [components.toCircuitPackage_hashChains,
    components.toCircuitPackage_permutationInvocations,
    components.toCircuitPackage_permutation,
    components.toCircuitPackage_layout]
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
    Nat.add_zero]
  rw [permutationInvocations_length, templateRows_length]
  rw [show physicalLayout.rowCount = 25556958 from rfl]
  calc
    _ = (priorChain.witnessLength + outputChain.witnessLength) +
          7613 * 592 +
          components.toCircuitPackage.compactRowCount +
          (components.toCircuitPackage.witnessInstructions.length +
            components.toCircuitPackage.assertionRows.length) := by
      omega
    _ = 12574080 + 7613 * 592 + 7489673 + 986309 := by
      rw [hashChainRows, compactRows_length, ordinaryFixed]
    _ = 25556958 := by norm_num

def components (_unit : Unit) : Components :=
  Components.of (arithmeticRows ()) (permutationInvocations ())

theorem components_arithmeticRows :
    (components ()).arithmeticRows = arithmeticRows () := by
  simpa only [components] using
    Components.of_arithmeticRows (arithmeticRows ())
      (permutationInvocations ())

theorem components_permutationInvocations :
    (components ()).permutationInvocations = permutationInvocations () := by
  simpa only [components] using
    Components.of_permutationInvocations (arithmeticRows ())
      (permutationInvocations ())

def circuitPackage (_unit : Unit) : CircuitPackage :=
  (components ()).toCircuitPackage

theorem circuitPackage_layout :
    (circuitPackage ()).layout = physicalLayout :=
  Components.toCircuitPackage_layout (components ())

theorem circuitPackage_relation :
    (circuitPackage ()).relation =
      productionCcsRelation physicalLayout.rowCount
        physicalLayout.totalColumnCount Lifecycle.cubeVariables :=
  Components.toCircuitPackage_relation (components ())

theorem circuitPackage_permutation :
    (circuitPackage ()).permutation = PilotData.permutationTemplate () :=
  Components.toCircuitPackage_permutation (components ())

theorem circuitPackage_hashChains :
    (circuitPackage ()).hashChains = [priorChain, outputChain] :=
  Components.toCircuitPackage_hashChains (components ())

theorem circuitPackage_permutationInvocations :
    (circuitPackage ()).permutationInvocations =
      (components ()).permutationInvocations :=
  Components.toCircuitPackage_permutationInvocations (components ())

theorem circuitPackage_compactRowTemplates :
    (circuitPackage ()).compactRowTemplates = compactRowTemplates () :=
  Components.toCircuitPackage_compactRowTemplates (components ())

theorem circuitPackage_compactRowInvocations :
    (circuitPackage ()).compactRowInvocations = compactRowInvocations () :=
  Components.toCircuitPackage_compactRowInvocations (components ())

theorem circuitPackage_witnessInstructions :
    (circuitPackage ()).witnessInstructions =
      (components ()).witnessInstructions :=
  Components.toCircuitPackage_witnessInstructions (components ())

theorem circuitPackage_witnessBatches :
    (circuitPackage ()).witnessBatches =
      WitnessProgram.batches logicalWidth publicFits :=
  Components.toCircuitPackage_witnessBatches (components ())

theorem circuitPackage_assertionRows :
    (circuitPackage ()).assertionRows = (components ()).assertionRows :=
  Components.toCircuitPackage_assertionRows (components ())

def relationIdentifier (_unit : Unit) : List F :=
  Package.relationIdentifier (circuitPackage ())

def artifact (_unit : Unit) : Artifact :=
  Package.sealPackage (circuitPackage ())

end NightstreamFPrime.Export.Stage1.Data
