import NightstreamFPrime.Export.Package
import NightstreamFPrime.Export.Stage1.Rows
import NightstreamFPrime.Gadgets.Poseidon2.Permutation
import NightstreamFPrime.Layout.PilotProduction
import NightstreamFPrime.Layout.PilotSpartan
import NightstreamFPrime.Layout.PilotValues
import NightstreamFPrime.Layout.R1CS

/-!
Owns the executable data of the canonical Stage 1 pilot package. Physical
positions and counts come only from `Layout.PilotProduction` and
`Layout.PilotSpartan`; this module serializes them. `Export.Pilot` owns the
proofs that connect the data to the production lifecycle layout.
-/

namespace NightstreamFPrime.Export.PilotData

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Layout
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Lifecycle

namespace Role

def priorPreimage : Nat := 1
def outputPreimage : Nat := 2
def witness : Nat := 3
def priorPublicInput : Nat := 4
def outputDigest : Nat := 5

end Role

def canonicalState : Layer.EState :=
  fun lane => Expr.var lane.val

def canonicalRecipes (_unit : Unit) : List Expr :=
  (Permutation.compile 8 canonicalState Permutation.schedule).recipes

def canonicalConstraints (_unit : Unit) : List Expr :=
  recipeConstraints 8 (canonicalRecipes ())

def canonicalRows (_unit : Unit) : List R1CS.Row :=
  (R1CS.lowerConstraints (canonicalConstraints ()) 600).rows

def columnRef (column : Nat) : ColumnRef :=
  if column < 8 then .input column else .local (column - 8)

def templateTerm (term : Nat × F) : TemplateTerm :=
  ⟨columnRef term.1, term.2.val⟩

def templateCombination (combination : R1CS.LinearCombination) :
    TemplateCombination :=
  ⟨combination.constant.val, combination.terms.map templateTerm⟩

def templateRowsFrom : Nat → List R1CS.Row → List TemplateRow
  | _, [] => []
  | output, row :: rest =>
      ⟨output, templateCombination row.a, templateCombination row.b,
        templateCombination row.c⟩ ::
      templateRowsFrom (output + 1) rest

def templateRows (_unit : Unit) : List TemplateRow :=
  templateRowsFrom 0 (canonicalRows ())

def permutationTemplate (_unit : Unit) : PermutationTemplate where
  inputCount := 8
  localColumnCount := 592
  outputLocalStart := 584
  rows := templateRows ()

def priorHashRowStart : Nat := PilotValues.priorHashRowStart
def priorHashRowCount : Nat := PilotValues.hashWitnessCount
def priorBindingRowStart : Nat := PilotValues.priorBindingRowStart
def outputHashRowStart : Nat := PilotValues.outputHashRowStart

def witnessPrivateLength : Nat := PilotValues.witnessPrivateLength
def priorWitnessStart : Nat := PilotValues.priorWitnessStart
def outputWitnessStart : Nat := PilotValues.outputWitnessStart

def priorChain : HashChain where
  phase := 1
  rowStart := priorHashRowStart
  rowCount := priorHashRowCount
  inputStart := PilotValues.priorPreimageStart
  inputLength := PilotValues.stateHashWords
  witnessStart := priorWitnessStart
  witnessLength := PilotValues.hashWitnessCount
  absorbCount := PilotValues.absorbCount
  digestLength := 0
  digestStart := 0

def outputChain : HashChain where
  phase := 2
  rowStart := outputHashRowStart
  rowCount := PilotValues.hashRowCount
  inputStart := PilotValues.secondPrivateStart
  inputLength := PilotValues.stateHashWords
  witnessStart := outputWitnessStart
  witnessLength := PilotValues.hashWitnessCount
  absorbCount := PilotValues.absorbCount
  digestLength := PilotValues.digestWords
  digestStart := PilotValues.secondPublicStart

def zeroCombination : SparseCombination := ⟨0, []⟩
def oneCombination : SparseCombination := ⟨1, []⟩

def digestRow (chain : HashChain) (lane : Fin 4) : SparseRow :=
  let outputColumn := chain.witnessStart + chain.absorbCount * 592 +
    584 + lane.val
  let expectedColumn := chain.digestStart + lane.val
  ⟨chain.rowStart + chain.witnessLength + lane.val,
    ⟨0, [⟨outputColumn, 1⟩, ⟨expectedColumn, (-1 : F).val⟩]⟩,
    oneCombination, zeroCombination⟩

def digestRows (chain : HashChain) : List SparseRow :=
  List.ofFn (digestRow chain)

def remapExpr : Expr → Expr
  | .var index => .var (PilotSpartan.sourceToSpartan index)
  | .const value => .const value
  | .add left right => .add (remapExpr left) (remapExpr right)
  | .mul left right => .mul (remapExpr left) (remapExpr right)

def remapBatch (batch : WitnessBatch) : WitnessBatch where
  start := PilotSpartan.sourceToSpartan batch.start
  recipes := batch.recipes.map remapExpr
  hints := batch.hints.map fun hint =>
    match hint with
    | .bit source index => .bit (remapExpr source) index
    | .inverseOrZero source => .inverseOrZero (remapExpr source)
    | .quotientFive source => .quotientFive (remapExpr source)
    | .remainderFive source => .remainderFive (remapExpr source)

def priorWordBatches (_unit : Unit) : List WitnessBatch :=
  (witnesses (PriorStateHash.wordOps PilotProduction.priorInterface
    PilotProduction.witnessOffset)).map remapBatch

def priorExtraConstraints (_unit : Unit) : List Expr :=
  flatConstraints
    (PriorStateHash.wordOps PilotProduction.priorInterface
      PilotProduction.witnessOffset ++
    PriorStateHash.bindingAssertions PilotProduction.priorInterface
      PilotProduction.witnessOffset)

theorem priorExtraConstraints_eq :
    priorExtraConstraints () =
      PilotProduction.priorWordConstraintsAll ++
        PilotProduction.priorBindingConstraints := by
  unfold priorExtraConstraints
  rw [flatConstraints_append]
  rfl

def priorExtraRows (_unit : Unit) : List Stage1.Rows.CompiledRow :=
  Stage1.Rows.compileRowsTR
    (PilotSpartan.sourceToSpartan PilotValues.logicalColumnCount)
    priorBindingRowStart
    (PilotSpartan.remapRows
      (Stage1.Rows.lowerConstraintsTR (priorExtraConstraints ())
        PilotValues.logicalColumnCount).rows)

def priorFixedRowStart : Nat :=
  priorBindingRowStart + PilotValues.priorCanonicalRowCount

def markerBindingRow : SparseRow :=
  ⟨priorFixedRowStart,
    ⟨(-1 : F).val, [⟨PilotValues.firstPublicStart, 1⟩]⟩,
    oneCombination, zeroCombination⟩

def tailBindingRows : List SparseRow :=
  List.ofFn fun lane : Fin 13 =>
    ⟨priorFixedRowStart + 1 + lane.val,
      ⟨0, [⟨PilotValues.firstPublicStart + 257 + lane.val, 1⟩]⟩,
      oneCombination, zeroCombination⟩

def bindingRows (_unit : Unit) : List SparseRow :=
  markerBindingRow :: tailBindingRows

def assertionRows (_unit : Unit) : List SparseRow :=
  Stage1.Rows.assertionRowsTR (priorExtraRows ()) ++ digestRows outputChain

def witnessInstructions (_unit : Unit) : List WitnessInstruction :=
  Stage1.Rows.witnessInstructionsTR (priorExtraRows ())

def profile : Profile where
  fieldModulus := goldilocksModulus
  decompositionBase := 2
  decompositionDigits := 16
  decompositionBound := 65536
  freshSources := 1
  runningSources := 16
  piRlcInputs := 17
  piDecChildren := 16
  ccsMatrices := 14
  cubeVariables := 25

def poseidonSchedule : PoseidonSchedule where
  width := Spec.Poseidon2.width
  rate := Spec.Poseidon2.rate
  digestLength := Spec.Poseidon2.digestLen
  initialFullRounds := Spec.Poseidon2.halfFullRounds
  partialRounds := Spec.Poseidon2.partialRounds
  terminalFullRounds := Spec.Poseidon2.halfFullRounds
  recipesPerPermutation := 592
  outputLocalStart := 584

def privateSegments : List Segment :=
  [⟨Role.priorPreimage, PilotValues.priorPreimageStart,
      PilotValues.stateHashWords⟩,
   ⟨Role.outputPreimage, PilotValues.secondPrivateStart,
      PilotValues.stateHashWords⟩,
   ⟨Role.witness, priorWitnessStart, witnessPrivateLength⟩]

def publicSegments : List Segment :=
  [⟨Role.priorPublicInput, PilotValues.firstPublicStart,
      PilotValues.priorPublicInputWords⟩,
   ⟨Role.outputDigest, PilotValues.secondPublicStart,
      PilotValues.digestWords⟩]

def physicalLayout : PhysicalLayout where
  rowCount := PilotValues.physicalRowCount
  privateColumnCount := PilotValues.privateColumnCount
  constantColumn := PilotValues.constantColumn
  publicColumnCount := PilotValues.publicColumnCount
  totalColumnCount := PilotValues.spartanColumnCount
  privateSegments := privateSegments
  publicSegments := publicSegments

def circuitPackageOf (batches : List WitnessBatch)
    (instructions : List WitnessInstruction)
    (assertions : List SparseRow) : CircuitPackage where
  schemaVersion := 8
  profile := profile
  poseidon := poseidonSchedule
  layout := physicalLayout
  relation := productionCcsRelation physicalLayout.rowCount
    physicalLayout.totalColumnCount profile.cubeVariables
  permutation := permutationTemplate ()
  hashChains := [priorChain, outputChain]
  permutationInvocations := []
  compactRowTemplates := []
  compactRowInvocations := []
  witnessBatches := batches
  witnessInstructions := instructions
  assertionRows := assertions
  terminal := none

def circuitPackage (_unit : Unit) : CircuitPackage :=
  circuitPackageOf (priorWordBatches ()) (witnessInstructions ())
    (assertionRows ())

@[simp] theorem circuitPackageOf_witnessInstructions
    (batches : List WitnessBatch) (instructions : List WitnessInstruction)
    (assertions : List SparseRow) :
    (circuitPackageOf batches instructions assertions).witnessInstructions =
      instructions := by
  rfl

@[simp] theorem circuitPackageOf_poseidon
    (batches : List WitnessBatch) (instructions : List WitnessInstruction)
    (assertions : List SparseRow) :
    (circuitPackageOf batches instructions assertions).poseidon =
      poseidonSchedule := by
  rfl

@[simp] theorem circuitPackageOf_permutation
    (batches : List WitnessBatch) (instructions : List WitnessInstruction)
    (assertions : List SparseRow) :
    (circuitPackageOf batches instructions assertions).permutation =
      permutationTemplate () := by
  rfl

@[simp] theorem circuitPackageOf_hashChains
    (batches : List WitnessBatch) (instructions : List WitnessInstruction)
    (assertions : List SparseRow) :
    (circuitPackageOf batches instructions assertions).hashChains =
      [priorChain, outputChain] := by
  rfl

@[simp] theorem circuitPackageOf_assertionRows
    (batches : List WitnessBatch) (instructions : List WitnessInstruction)
    (assertions : List SparseRow) :
    (circuitPackageOf batches instructions assertions).assertionRows =
      assertions := by
  rfl

def relationIdentifier (_unit : Unit) : List F :=
  Package.relationIdentifier (circuitPackage ())

def artifact (_unit : Unit) : Artifact :=
  Package.sealPackage (circuitPackage ())

end NightstreamFPrime.Export.PilotData
