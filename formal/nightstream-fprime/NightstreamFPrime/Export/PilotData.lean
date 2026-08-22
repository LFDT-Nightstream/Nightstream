import NightstreamFPrime.Export.Package
import NightstreamFPrime.Gadgets.Poseidon2.Permutation
import NightstreamFPrime.Layout.R1CS

/-!
Owns the executable data of the canonical Stage 1 pilot package. This module
has no lifecycle or production-key import, so the emitter initializes only
the compiler and layout data it emits. `Export.Pilot` owns the proofs that
connect this data to the full production lifecycle layout.
-/

namespace NightstreamFPrime.Export.PilotData

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Layout
open NightstreamFPrime.Export.Package

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

def priorHashRowStart : Nat := 0
def priorHashRowCount : Nat := 6031300
def priorBindingRowStart : Nat := 6031300
def outputHashRowStart : Nat := 6031350

def witnessPrivateLength : Nat := 12062592
def priorWitnessStart : Nat := 81490
def outputWitnessStart : Nat := 6112786

def priorChain : HashChain where
  phase := 1
  rowStart := priorHashRowStart
  rowCount := priorHashRowCount
  inputStart := 0
  inputLength := 40745
  witnessStart := priorWitnessStart
  witnessLength := 6031296
  absorbCount := 10187
  digestStart := 12144084

def outputChain : HashChain where
  phase := 2
  rowStart := outputHashRowStart
  rowCount := 6031300
  inputStart := 40745
  inputLength := 40745
  witnessStart := outputWitnessStart
  witnessLength := 6031296
  absorbCount := 10187
  digestStart := 12144137

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

def markerBindingRow : SparseRow :=
  ⟨priorBindingRowStart,
    ⟨(-1 : F).val, [⟨12144083, 1⟩]⟩,
    oneCombination, zeroCombination⟩

def tailBindingRows : List SparseRow :=
  List.ofFn fun lane : Fin 49 =>
    ⟨priorBindingRowStart + 1 + lane.val,
      ⟨0, [⟨12144088 + lane.val, 1⟩]⟩,
      oneCombination, zeroCombination⟩

def bindingRows (_unit : Unit) : List SparseRow :=
  markerBindingRow :: tailBindingRows

def assertionRows (_unit : Unit) : List SparseRow :=
  digestRows priorChain ++ bindingRows () ++ digestRows outputChain

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
  cubeVariables := 24

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
  [⟨Role.priorPreimage, 0, 40745⟩,
   ⟨Role.outputPreimage, 40745, 40745⟩,
   ⟨Role.witness, priorWitnessStart, witnessPrivateLength⟩]

def publicSegments : List Segment :=
  [⟨Role.priorPublicInput, 12144083, 54⟩,
   ⟨Role.outputDigest, 12144137, 4⟩]

def physicalLayout : PhysicalLayout where
  rowCount := 12062650
  privateColumnCount := 12144082
  constantColumn := 12144082
  publicColumnCount := 58
  totalColumnCount := 12144141
  privateSegments := privateSegments
  publicSegments := publicSegments

def circuitPackage (_unit : Unit) : CircuitPackage where
  schemaVersion := 1
  profile := profile
  poseidon := poseidonSchedule
  layout := physicalLayout
  permutation := permutationTemplate ()
  hashChains := [priorChain, outputChain]
  assertionRows := assertionRows ()
  terminal := none

def relationIdentifier (_unit : Unit) : List F :=
  Package.relationIdentifier (circuitPackage ())

def artifact (_unit : Unit) : Artifact :=
  Package.sealPackage (circuitPackage ())

end NightstreamFPrime.Export.PilotData
