import NightstreamFPrime.Export.RowSemantics
import NightstreamFPrime.Export.Stage1.ApplicationPackage
import NightstreamFPrime.Export.Stage1.CompactRows
import NightstreamFPrime.Export.Stage1.Data
import NightstreamFPrime.Export.Stage1.NextPreimagePackage
import NightstreamFPrime.Export.Stage1.TerminalPackage
import NightstreamFPrime.Lifecycle.Stage1.VerificationKey

/-!
Owns the generic final-package constructor for one Lean-authored application.

The current validated package is an unchanged prefix. For a selected
application, this module appends its Lean-compiled rows in a new private
suffix, shifts the prior constant/public suffix, installs outer-terminal
metadata, and recomputes all binding inputs. A concrete production application
must still supply the final `Program` and prove the `2^28` fit.
-/

namespace NightstreamFPrime.Export.Stage1.PerApplicationPackage

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec

/- Schema roles owned by the per-application extension. -/
namespace Role

def applicationWitness : Nat := 17
def applicationLocal : Nat := 18

end Role

def basePackage (_delay : Unit := ()) : CircuitPackage := Data.circuitPackage ()

/-- Exact application plan whose rows start after the validated prefix. -/
def applicationPlan (program : Lifecycle.Stage1.Application.Program) :
    ApplicationPackage.Plan :=
  ApplicationPackage.productionPlan program basePackage.layout.rowCount

/-- Closed-form executable plan. The reference definition above remains the
package authority. -/
def directApplicationPlan (program : Lifecycle.Stage1.Application.Program) :
    ApplicationPackage.Plan :=
  ApplicationPackage.productionPlan program 29218024

theorem directApplicationPlan_eq_applicationPlan
    (program : Lifecycle.Stage1.Application.Program) :
    directApplicationPlan program = applicationPlan program := by
  unfold directApplicationPlan applicationPlan basePackage
  rw [Data.circuitPackage_layout]
  rfl

@[csimp] theorem applicationPlan_eq_directApplicationPlan :
    @applicationPlan = @directApplicationPlan := by
  funext program
  exact (directApplicationPlan_eq_applicationPlan program).symm

/-- Count only the physical application rows. This avoids constructing,
lowering, and classifying the complete row list when geometry needs only its
length. -/
def directApplicationRowCount
    (program : Lifecycle.Stage1.Application.Program) : Nat :=
  R1CS.totalRowCount
    (ApplicationPackage.constraints program
      (ApplicationPackage.productionColumns program)
      (Layout.Stage1.ApplicationInputs.localStart program))

theorem directApplicationRowCount_eq_plan
    (program : Lifecycle.Stage1.Application.Program) :
    directApplicationRowCount program =
      (directApplicationPlan program).rowCount := by
  unfold directApplicationRowCount directApplicationPlan
  rw [ApplicationPackage.productionPlan_rowCount]
  unfold ApplicationPackage.compiledRows
  rw [Rows.compileRowsTR_length, Rows.lowerConstraintsTR_eq,
    R1CS.lowerConstraints_rows_length]

/-- Count only application-owned private columns. Geometry does not need the
compiled rows, witness instructions, or assertion partition. -/
def directApplicationPrivateCount
    (program : Lifecycle.Stage1.Application.Program) : Nat :=
  let operations := ApplicationPackage.operations program
    (ApplicationPackage.productionColumns program)
    (Layout.Stage1.ApplicationInputs.localStart program)
  localLength operations +
    R1CS.totalFreshCount (flatConstraints operations)

theorem directApplicationPrivateCount_eq_plan
    (program : Lifecycle.Stage1.Application.Program) :
    directApplicationPrivateCount program =
      (directApplicationPlan program).privateCount := by
  unfold directApplicationPrivateCount directApplicationPlan
  rw [ApplicationPackage.productionPlan_privateCount]
  rfl

def nextPreimageRowStart
    (program : Lifecycle.Stage1.Application.Program) : Nat :=
  basePackage.layout.rowCount + (applicationPlan program).rowCount

def directNextPreimageRowStart
    (program : Lifecycle.Stage1.Application.Program) : Nat :=
  29218024 + directApplicationRowCount program

theorem directNextPreimageRowStart_eq_nextPreimageRowStart
    (program : Lifecycle.Stage1.Application.Program) :
    directNextPreimageRowStart program = nextPreimageRowStart program := by
  unfold directNextPreimageRowStart nextPreimageRowStart basePackage
  rw [directApplicationRowCount_eq_plan,
    directApplicationPlan_eq_applicationPlan, Data.circuitPackage_layout]
  rfl

@[csimp] theorem nextPreimageRowStart_eq_directNextPreimageRowStart :
    @nextPreimageRowStart = @directNextPreimageRowStart := by
  funext program
  exact (directNextPreimageRowStart_eq_nextPreimageRowStart program).symm

/-- New caller-owned witness words plus every application-generated private
column. -/
def addedPrivateColumnCount
    (program : Lifecycle.Stage1.Application.Program) : Nat :=
  program.witnessWordCount + (applicationPlan program).privateCount

def directAddedPrivateColumnCount
    (program : Lifecycle.Stage1.Application.Program) : Nat :=
  program.witnessWordCount + directApplicationPrivateCount program

theorem directAddedPrivateColumnCount_eq_addedPrivateColumnCount
    (program : Lifecycle.Stage1.Application.Program) :
    directAddedPrivateColumnCount program = addedPrivateColumnCount program := by
  unfold directAddedPrivateColumnCount addedPrivateColumnCount
  rw [directApplicationPrivateCount_eq_plan,
    directApplicationPlan_eq_applicationPlan]

@[csimp] theorem addedPrivateColumnCount_eq_directAddedPrivateColumnCount :
    @addedPrivateColumnCount = @directAddedPrivateColumnCount := by
  funext program
  exact (directAddedPrivateColumnCount_eq_addedPrivateColumnCount program).symm

/-- Existing private columns stay fixed. The former constant and every public
column move after the application-private suffix. -/
def shiftColumn (program : Lifecycle.Stage1.Application.Program)
    (column : Nat) : Nat :=
  if column < basePackage.layout.constantColumn then column
  else column + addedPrivateColumnCount program

def directShiftColumn (program : Lifecycle.Stage1.Application.Program)
    (column : Nat) : Nat :=
  if column < Data.physicalLayout.constantColumn then column
  else column + directAddedPrivateColumnCount program

theorem directShiftColumn_eq_shiftColumn
    (program : Lifecycle.Stage1.Application.Program) (column : Nat) :
    directShiftColumn program column = shiftColumn program column := by
  unfold directShiftColumn shiftColumn basePackage
  rw [directAddedPrivateColumnCount_eq_addedPrivateColumnCount,
    Data.circuitPackage_layout]

@[csimp] theorem shiftColumn_eq_directShiftColumn :
    @shiftColumn = @directShiftColumn := by
  funext program column
  exact (directShiftColumn_eq_shiftColumn program column).symm

def baseEnv (program : Lifecycle.Stage1.Application.Program)
    (env : Env) : Env :=
  fun column => env (shiftColumn program column)

def shiftExpr (program : Lifecycle.Stage1.Application.Program) (value : Expr) :
    Expr :=
  CompactRows.renameExpr (shiftColumn program) value

def shiftHint (program : Lifecycle.Stage1.Application.Program) : Hint → Hint
  | .bit source index => .bit (shiftExpr program source) index
  | .inverseOrZero source => .inverseOrZero (shiftExpr program source)
  | .quotientFive source => .quotientFive (shiftExpr program source)
  | .remainderFive source => .remainderFive (shiftExpr program source)

def shiftBatch (program : Lifecycle.Stage1.Application.Program)
    (batch : WitnessBatch) : WitnessBatch where
  start := shiftColumn program batch.start
  recipes := batch.recipes.map (shiftExpr program)
  hints := batch.hints.map (shiftHint program)

def shiftSparseTerm (program : Lifecycle.Stage1.Application.Program)
    (term : SparseTerm) : SparseTerm :=
  ⟨shiftColumn program term.column, term.coefficient⟩

def shiftSparseCombination (program : Lifecycle.Stage1.Application.Program)
    (combination : SparseCombination) : SparseCombination :=
  ⟨combination.constant,
    combination.terms.map (shiftSparseTerm program)⟩

def shiftWitnessInstruction
    (program : Lifecycle.Stage1.Application.Program)
    (instruction : WitnessInstruction) : WitnessInstruction where
  rowIndex := instruction.rowIndex
  target := shiftColumn program instruction.target
  a := shiftSparseCombination program instruction.a
  b := shiftSparseCombination program instruction.b

def shiftSparseRow (program : Lifecycle.Stage1.Application.Program)
    (row : SparseRow) : SparseRow where
  rowIndex := row.rowIndex
  a := shiftSparseCombination program row.a
  b := shiftSparseCombination program row.b
  c := shiftSparseCombination program row.c

theorem shiftSparseCombination_toR1CS
    (program : Lifecycle.Stage1.Application.Program)
    (combination : SparseCombination) :
    (shiftSparseCombination program combination).toR1CS =
      CompactRows.renameCombination (shiftColumn program)
        combination.toR1CS := by
  cases combination
  simp [shiftSparseCombination, shiftSparseTerm,
    SparseCombination.toR1CS, CompactRows.renameCombination,
    List.map_map, Function.comp_def]

theorem shiftSparseCombination_eval
    (program : Lifecycle.Stage1.Application.Program)
    (combination : SparseCombination) (env : Env) :
    (shiftSparseCombination program combination).eval env =
      combination.eval (fun column => env (shiftColumn program column)) := by
  cases combination
  simp [shiftSparseCombination, shiftSparseTerm, SparseCombination.eval,
    List.map_map, Function.comp_def]

theorem shiftWitnessInstruction_holds
    (program : Lifecycle.Stage1.Application.Program)
    (instruction : WitnessInstruction) (env : Env) :
    (shiftWitnessInstruction program instruction).Holds env ↔
      instruction.Holds (fun column => env (shiftColumn program column)) := by
  simp [shiftWitnessInstruction, WitnessInstruction.Holds,
    shiftSparseCombination_eval]

theorem shiftSparseRow_holds
    (program : Lifecycle.Stage1.Application.Program)
    (row : SparseRow) (env : Env) :
    (shiftSparseRow program row).Holds env ↔
      row.Holds (fun column => env (shiftColumn program column)) := by
  simp [shiftSparseRow, SparseRow.Holds, shiftSparseCombination_eval]

def shiftHashChain (program : Lifecycle.Stage1.Application.Program)
    (chain : HashChain) : HashChain where
  phase := chain.phase
  rowStart := chain.rowStart
  rowCount := chain.rowCount
  inputStart := shiftColumn program chain.inputStart
  inputLength := chain.inputLength
  witnessStart := shiftColumn program chain.witnessStart
  witnessLength := chain.witnessLength
  absorbCount := chain.absorbCount
  digestLength := chain.digestLength
  digestStart := shiftColumn program chain.digestStart

def shiftPermutationInvocation
    (program : Lifecycle.Stage1.Application.Program)
    (invocation : PermutationInvocation) : PermutationInvocation where
  phase := invocation.phase
  rowStart := invocation.rowStart
  witnessStart := shiftColumn program invocation.witnessStart
  inputs := invocation.inputs.map (shiftSparseCombination program)

def shiftCompactInputRange
    (program : Lifecycle.Stage1.Application.Program)
    (range : CompactInputRange) : CompactInputRange where
  inputStart := range.inputStart
  inputCount := range.inputCount
  columnStart := shiftColumn program range.columnStart
  columnStride := range.columnStride

def shiftCompactRowInvocation
    (program : Lifecycle.Stage1.Application.Program)
    (invocation : CompactRowInvocation) : CompactRowInvocation where
  phase := invocation.phase
  templateIndex := invocation.templateIndex
  rowStart := invocation.rowStart
  localStart := shiftColumn program invocation.localStart
  inputRanges := invocation.inputRanges.map (shiftCompactInputRange program)

def shiftSegment (program : Lifecycle.Stage1.Application.Program)
    (segment : Segment) : Segment :=
  ⟨segment.role, shiftColumn program segment.start, segment.length⟩

def applicationWitnessSegment
    (program : Lifecycle.Stage1.Application.Program) : Segment :=
  ⟨Role.applicationWitness, basePackage.layout.constantColumn,
    program.witnessWordCount⟩

def applicationLocalSegment
    (program : Lifecycle.Stage1.Application.Program) : Segment :=
  ⟨Role.applicationLocal, (applicationPlan program).privateStart,
    (applicationPlan program).privateCount⟩

def finalLayout (program : Lifecycle.Stage1.Application.Program) :
    PhysicalLayout where
  rowCount := nextPreimageRowStart program + 5
  privateColumnCount :=
    basePackage.layout.privateColumnCount + addedPrivateColumnCount program
  constantColumn :=
    basePackage.layout.constantColumn + addedPrivateColumnCount program
  publicColumnCount := basePackage.layout.publicColumnCount
  totalColumnCount :=
    basePackage.layout.totalColumnCount + addedPrivateColumnCount program
  privateSegments := basePackage.layout.privateSegments ++
    [applicationWitnessSegment program, applicationLocalSegment program]
  publicSegments := basePackage.layout.publicSegments.map (shiftSegment program)

def directApplicationWitnessSegment
    (program : Lifecycle.Stage1.Application.Program) : Segment :=
  ⟨Role.applicationWitness, Data.physicalLayout.constantColumn,
    program.witnessWordCount⟩

def directApplicationLocalSegment
    (program : Lifecycle.Stage1.Application.Program) : Segment :=
  ⟨Role.applicationLocal, (directApplicationPlan program).privateStart,
    (directApplicationPlan program).privateCount⟩

def directShiftSegment (program : Lifecycle.Stage1.Application.Program)
    (segment : Segment) : Segment :=
  ⟨segment.role, directShiftColumn program segment.start, segment.length⟩

def directFinalLayout (program : Lifecycle.Stage1.Application.Program) :
    PhysicalLayout where
  rowCount := directNextPreimageRowStart program + 5
  privateColumnCount := Data.physicalLayout.privateColumnCount +
    directAddedPrivateColumnCount program
  constantColumn := Data.physicalLayout.constantColumn +
    directAddedPrivateColumnCount program
  publicColumnCount := Data.physicalLayout.publicColumnCount
  totalColumnCount := Data.physicalLayout.totalColumnCount +
    directAddedPrivateColumnCount program
  privateSegments := Data.physicalLayout.privateSegments ++
    [directApplicationWitnessSegment program,
      directApplicationLocalSegment program]
  publicSegments := Data.physicalLayout.publicSegments.map
    (directShiftSegment program)

theorem directApplicationWitnessSegment_eq_applicationWitnessSegment
    (program : Lifecycle.Stage1.Application.Program) :
    directApplicationWitnessSegment program = applicationWitnessSegment program := by
  unfold directApplicationWitnessSegment applicationWitnessSegment basePackage
  rw [Data.circuitPackage_layout]

theorem directApplicationLocalSegment_eq_applicationLocalSegment
    (program : Lifecycle.Stage1.Application.Program) :
    directApplicationLocalSegment program = applicationLocalSegment program := by
  unfold directApplicationLocalSegment applicationLocalSegment
  rw [directApplicationPlan_eq_applicationPlan]

theorem directShiftSegment_eq_shiftSegment
    (program : Lifecycle.Stage1.Application.Program) (segment : Segment) :
    directShiftSegment program segment = shiftSegment program segment := by
  unfold directShiftSegment shiftSegment
  rw [directShiftColumn_eq_shiftColumn]

theorem directFinalLayout_eq_finalLayout
    (program : Lifecycle.Stage1.Application.Program) :
    directFinalLayout program = finalLayout program := by
  have shiftSegments :
      directShiftSegment program = shiftSegment program := by
    funext segment
    exact directShiftSegment_eq_shiftSegment program segment
  unfold directFinalLayout finalLayout
  rw [directNextPreimageRowStart_eq_nextPreimageRowStart,
    directAddedPrivateColumnCount_eq_addedPrivateColumnCount,
    directApplicationWitnessSegment_eq_applicationWitnessSegment,
    directApplicationLocalSegment_eq_applicationLocalSegment,
    shiftSegments, basePackage, Data.circuitPackage_layout]

/-- One complete package for one selected application. Application rows are
ordinary rows because `Program.circuit` is the semantic authority; Rust does
not select an application-specific template. -/
def package (program : Lifecycle.Stage1.Application.Program) : CircuitPackage :=
  let plan := applicationPlan program
  TerminalPackage.install {
    schemaVersion := 8
    profile := basePackage.profile
    poseidon := basePackage.poseidon
    layout := finalLayout program
    relation := productionCcsRelation
      (finalLayout program).rowCount (finalLayout program).totalColumnCount
      Lifecycle.cubeVariables
    permutation := basePackage.permutation
    hashChains := basePackage.hashChains.map (shiftHashChain program)
    permutationInvocations := basePackage.permutationInvocations.map
      (shiftPermutationInvocation program)
    compactRowTemplates := basePackage.compactRowTemplates
    compactRowInvocations := basePackage.compactRowInvocations.map
      (shiftCompactRowInvocation program)
    witnessBatches := basePackage.witnessBatches.map (shiftBatch program) ++
      plan.witnessBatches
    witnessInstructions := basePackage.witnessInstructions.map
      (shiftWitnessInstruction program) ++ plan.witnessInstructions
    assertionRows := (basePackage.assertionRows.map (shiftSparseRow program) ++
      plan.assertionRows) ++
        NextPreimagePackage.assertionRows (nextPreimageRowStart program)
    terminal := none }

def structuralPackageIdentity
    (program : Lifecycle.Stage1.Application.Program) :
    VerifierContext.Digest4 :=
  VerifierContext.Digest4.ofList
    (Package.relationIdentifier (package program))

def applicationAuthorityWords
    (program : Lifecycle.Stage1.Application.Program) : List F :=
  ApplicationPackage.authorityWords (applicationPlan program)

/-- Raw verifier authority for one exact application package. The caller
supplies full static key words, not their digests. -/
def authority (program : Lifecycle.Stage1.Application.Program)
    (nifsKeyWords commitmentKeyWords : List F) : VerifierContext.Authority where
  relationWords := (structuralPackageIdentity program).toList
  applicationWords := applicationAuthorityWords program
  nifsKeyWords := nifsKeyWords
  commitmentKeyWords := commitmentKeyWords

def verifierContextDescriptor (program : Lifecycle.Stage1.Application.Program)
    (nifsKeyWords commitmentKeyWords : List F) : VerifierContext.Descriptor :=
  VerifierContext.descriptor
    (authority program nifsKeyWords commitmentKeyWords)

def verifierContext (program : Lifecycle.Stage1.Application.Program)
    (nifsKeyWords commitmentKeyWords : List F) : VerifierContext.Digest4 :=
  (verifierContextDescriptor program nifsKeyWords commitmentKeyWords).digest4

def packageIdentityDomain : List F :=
  ([78, 105, 103, 104, 116, 115, 116, 114, 101, 97, 109, 47,
    70, 80, 114, 105, 109, 101, 47, 115, 101, 97, 108, 101, 100,
    45, 112, 97, 99, 107, 97, 103, 101, 47, 118, 49] : List Nat).map
      Poseidon2.ofNat

/-- Canonical preimage that joins the exact package to the verifier-owned
static authority. The structural identity is recomputed from the package, and
the descriptor is recomputed from the full application and key words. -/
def packageIdentityPreimage (program : Lifecycle.Stage1.Application.Program)
    (nifsKeyWords commitmentKeyWords : List F) : List F :=
  packageIdentityDomain ++
    VerifierContext.framed (structuralPackageIdentity program).toList ++
    VerifierContext.framed
      (verifierContextDescriptor program nifsKeyWords
        commitmentKeyWords).serialize

/-- Final verifier-owned identity for one exact application package and one
exact static authority. It is acyclic because the context contains only the
structural package identity, not this final identity. -/
def packageIdentity (program : Lifecycle.Stage1.Application.Program)
    (nifsKeyWords commitmentKeyWords : List F) : VerifierContext.Digest4 :=
  VerifierContext.Digest4.ofList
    (Poseidon2.hash
      (packageIdentityPreimage program nifsKeyWords commitmentKeyWords))

def verificationKeyBinding (program : Lifecycle.Stage1.Application.Program)
    (nifsKeyWords commitmentKeyWords : List F) :
    Lifecycle.Stage1.VerificationKey.Binding :=
  Lifecycle.Stage1.VerificationKey.ofAuthority
    (packageIdentity program nifsKeyWords commitmentKeyWords)
    (authority program nifsKeyWords commitmentKeyWords)

theorem structuralPackageIdentity_recomputed
    (program : Lifecycle.Stage1.Application.Program) :
    (structuralPackageIdentity program).toList =
      (VerifierContext.Digest4.ofList
        (Package.relationIdentifier (package program))).toList := by
  rfl

theorem packageIdentity_recomputed
    (program : Lifecycle.Stage1.Application.Program)
    (nifsKeyWords commitmentKeyWords : List F) :
    (packageIdentity program nifsKeyWords commitmentKeyWords).toList =
      (VerifierContext.Digest4.ofList
        (Poseidon2.hash
          (packageIdentityPreimage program nifsKeyWords
            commitmentKeyWords))).toList := by
  rfl

theorem packageIdentityPreimage_length
    (program : Lifecycle.Stage1.Application.Program)
    (nifsKeyWords commitmentKeyWords : List F) :
    (packageIdentityPreimage program nifsKeyWords
      commitmentKeyWords).length = 128 := by
  simp [packageIdentityPreimage, packageIdentityDomain,
    VerifierContext.framed,
    VerifierContext.Descriptor.serialize_length,
    VerifierContext.Digest4.toList_length]

/-- Equality of final binding preimages identifies both acyclic components.
The fixed-length framing prevents one component from consuming words from the
other component. -/
theorem packageIdentityPreimage_components
    (leftProgram rightProgram : Lifecycle.Stage1.Application.Program)
    (leftNifs leftCommitment rightNifs rightCommitment : List F)
    (same : packageIdentityPreimage leftProgram leftNifs leftCommitment =
      packageIdentityPreimage rightProgram rightNifs rightCommitment) :
    structuralPackageIdentity leftProgram =
        structuralPackageIdentity rightProgram ∧
      verifierContextDescriptor leftProgram leftNifs leftCommitment =
        verifierContextDescriptor rightProgram rightNifs rightCommitment := by
  have body :
      VerifierContext.framed
          (structuralPackageIdentity leftProgram).toList ++
          VerifierContext.framed
            (verifierContextDescriptor leftProgram leftNifs
              leftCommitment).serialize =
        VerifierContext.framed
          (structuralPackageIdentity rightProgram).toList ++
          VerifierContext.framed
            (verifierContextDescriptor rightProgram rightNifs
              rightCommitment).serialize := by
    apply List.append_cancel_left (as := packageIdentityDomain)
    simpa [packageIdentityPreimage, List.append_assoc] using same
  have structuralWords := congrArg (List.take 5) body
  have descriptorWords := congrArg (List.drop 5) body
  constructor
  · apply VerifierContext.Digest4.toList_injective
    simpa [VerifierContext.framed, VerifierContext.Digest4.toList] using
      structuralWords
  · apply VerifierContext.Descriptor.serialize_injective
    simpa [VerifierContext.framed, VerifierContext.Digest4.toList,
      VerifierContext.Descriptor.serialize_length] using descriptorWords

@[simp] theorem shiftColumn_private
    (program : Lifecycle.Stage1.Application.Program) (column : Nat)
    (bound : column < basePackage.layout.constantColumn) :
    shiftColumn program column = column := by
  simp [shiftColumn, bound]

@[simp] theorem shiftColumn_constantOrPublic
    (program : Lifecycle.Stage1.Application.Program) (column : Nat)
    (bound : basePackage.layout.constantColumn ≤ column) :
    shiftColumn program column = column + addedPrivateColumnCount program := by
  simp [shiftColumn, Nat.not_lt.mpr bound]

theorem applicationPlan_wellFormed
    (program : Lifecycle.Stage1.Application.Program) :
    (applicationPlan program).WellFormed := by
  exact ApplicationPackage.productionPlan_wellFormed program
    basePackage.layout.rowCount

@[simp] theorem package_schemaVersion
    (program : Lifecycle.Stage1.Application.Program) :
    (package program).schemaVersion = 8 := by
  rfl

@[simp] theorem package_layout
    (program : Lifecycle.Stage1.Application.Program) :
    (package program).layout = finalLayout program := by
  rfl

@[simp] theorem package_terminal
    (program : Lifecycle.Stage1.Application.Program) :
    (package program).terminal =
      some (TerminalPackage.layoutFor (package program)) := by
  rfl

theorem package_rowCount
    (program : Lifecycle.Stage1.Application.Program) :
    (package program).layout.rowCount =
      basePackage.layout.rowCount + (applicationPlan program).rowCount + 5 := by
  rfl

theorem package_privateColumnCount
    (program : Lifecycle.Stage1.Application.Program) :
    (package program).layout.privateColumnCount =
      basePackage.layout.privateColumnCount + addedPrivateColumnCount program := by
  rfl

theorem package_constantColumn
    (program : Lifecycle.Stage1.Application.Program) :
    (package program).layout.constantColumn =
      basePackage.layout.constantColumn + addedPrivateColumnCount program := by
  rfl

theorem package_publicColumnCount
    (program : Lifecycle.Stage1.Application.Program) :
    (package program).layout.publicColumnCount =
      basePackage.layout.publicColumnCount := by
  rfl

theorem package_totalColumnCount
    (program : Lifecycle.Stage1.Application.Program) :
    (package program).layout.totalColumnCount =
      basePackage.layout.totalColumnCount + addedPrivateColumnCount program := by
  rfl

theorem package_relation
    (program : Lifecycle.Stage1.Application.Program) :
    (package program).relation = productionCcsRelation
      (package program).layout.rowCount
      (package program).layout.totalColumnCount Lifecycle.cubeVariables := by
  rfl

theorem package_permutation
    (program : Lifecycle.Stage1.Application.Program) :
    (package program).permutation = basePackage.permutation := by
  rfl

theorem package_hashChains
    (program : Lifecycle.Stage1.Application.Program) :
    (package program).hashChains =
      basePackage.hashChains.map (shiftHashChain program) := by
  rfl

theorem package_permutationInvocations
    (program : Lifecycle.Stage1.Application.Program) :
    (package program).permutationInvocations =
      basePackage.permutationInvocations.map
        (shiftPermutationInvocation program) := by
  rfl

theorem package_compactRowTemplates
    (program : Lifecycle.Stage1.Application.Program) :
    (package program).compactRowTemplates =
      basePackage.compactRowTemplates := by
  rfl

theorem package_compactRowInvocations
    (program : Lifecycle.Stage1.Application.Program) :
    (package program).compactRowInvocations =
      basePackage.compactRowInvocations.map
        (shiftCompactRowInvocation program) := by
  rfl

theorem package_witnessBatches
    (program : Lifecycle.Stage1.Application.Program) :
    (package program).witnessBatches =
      basePackage.witnessBatches.map (shiftBatch program) ++
        (applicationPlan program).witnessBatches := by
  rfl

theorem package_witnessInstructions
    (program : Lifecycle.Stage1.Application.Program) :
    (package program).witnessInstructions =
      basePackage.witnessInstructions.map
          (shiftWitnessInstruction program) ++
        (applicationPlan program).witnessInstructions := by
  rfl

theorem package_assertionRows
    (program : Lifecycle.Stage1.Application.Program) :
    (package program).assertionRows =
      (basePackage.assertionRows.map (shiftSparseRow program) ++
        (applicationPlan program).assertionRows) ++
          NextPreimagePackage.assertionRows
            (nextPreimageRowStart program) := by
  rfl

/-- Any satisfying final-package assignment obeys the exact selected
application transition. This theorem uses only the application-owned suffix;
base-package preservation is a separate Stage 1 theorem. -/
theorem packageRows_imply_applicationHolds
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (holds : (package program).RowsHold env) :
    Lifecycle.Stage1.Application.Holds program.step
      (Layout.Stage1.ApplicationInputs.interface program)
      (Layout.Stage1.ApplicationInputs.localStart program) env := by
  have instructions : ∀ instruction ∈
      (applicationPlan program).witnessInstructions,
      instruction.Holds env := by
    intro instruction member
    exact holds.2.2.2.1 instruction (by
      rw [package_witnessInstructions]
      exact List.mem_append_right _ member)
  have assertions : ∀ assertion ∈
      (applicationPlan program).assertionRows, assertion.Holds env := by
    intro assertion member
    exact holds.2.2.2.2 assertion (by
      rw [package_assertionRows]
      apply List.mem_append_left
      exact List.mem_append_right _ member)
  have assumptions := program.assumptions
    (Layout.Stage1.ApplicationInputs.interface program)
    (Layout.Stage1.ApplicationInputs.localStart program) env
    (Layout.Stage1.ApplicationInputs.externalBelow program)
  exact ApplicationPackage.rows_imply_programHolds program
    (ApplicationPackage.productionColumns program)
    (Layout.Stage1.ApplicationInputs.localStart program)
    basePackage.layout.rowCount env (by simpa using assumptions)
    (by simpa [applicationPlan, ApplicationPackage.productionPlan] using
      instructions)
    (by simpa [applicationPlan, ApplicationPackage.productionPlan] using
      assertions)

/-- Shifted ordinary prefix rows have exactly the original meaning under the
column pullback. -/
theorem packageRows_imply_baseOrdinaryRows
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (holds : (package program).RowsHold env) :
    (∀ instruction ∈ basePackage.witnessInstructions,
        instruction.Holds (baseEnv program env)) ∧
      ∀ assertion ∈ basePackage.assertionRows,
        assertion.Holds (baseEnv program env) := by
  constructor
  · intro instruction member
    have shifted : (shiftWitnessInstruction program instruction).Holds env :=
      holds.2.2.2.1 _ (by
        rw [package_witnessInstructions]
        apply List.mem_append_left
        exact List.mem_map_of_mem member)
    exact (shiftWitnessInstruction_holds program instruction env).mp shifted
  · intro assertion member
    have shifted : (shiftSparseRow program assertion).Holds env :=
      holds.2.2.2.2 _ (by
        rw [package_assertionRows]
        apply List.mem_append_left
        apply List.mem_append_left
        exact List.mem_map_of_mem member)
    exact (shiftSparseRow_holds program assertion env).mp shifted

/-- The five per-application parent-wiring rows force the exact next-preimage
counter and initial-state equations. -/
theorem packageRows_imply_nextPreimageSpec
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (holds : (package program).RowsHold env) :
    Lifecycle.Stage1.NextPreimage.SpecHolds
      Layout.Stage1.NextPreimageInputs.sourceInterface
      Layout.Stage1.RunningTransitionInputs.phaseOffset
      (Layout.Stage1.Spartan.pullback env) := by
  let rowStart := nextPreimageRowStart program
  let rows := NextPreimagePackage.compiledRows rowStart
  have instructions : ∀ instruction ∈
      Rows.witnessInstructionsTR rows, instruction.Holds env := by
    intro instruction member
    have empty : Rows.witnessInstructionsTR rows = [] := by
      simpa [rows, rowStart, NextPreimagePackage.witnessInstructions] using
        NextPreimagePackage.witnessInstructions_eq_nil rowStart
    rw [empty] at member
    contradiction
  have assertions : ∀ assertion ∈ Rows.assertionRowsTR rows,
      assertion.Holds env := by
    intro assertion member
    exact holds.2.2.2.2 assertion (by
      rw [package_assertionRows]
      apply List.mem_append_right
      simpa [rows, rowStart, NextPreimagePackage.assertionRows] using member)
  have compiled : R1CS.RowsHold env
      (rows.map Rows.CompiledRow.toR1CS) :=
    (Rows.compiledRows_hold_iff rows env).mpr ⟨instructions, assertions⟩
  have sourceRows : R1CS.RowsHold env NextPreimagePackage.sourceRows := by
    rw [NextPreimagePackage.sourceRows_eq,
      ← NextPreimagePackage.compiledRows_toR1CS rowStart]
    exact compiled
  have spartanSpec := NextPreimagePackage.sourceRows_imply_spec env sourceRows
  have sourceSpec :=
    (Layout.Stage1.NextPreimageInputs.spartanSpec_iff_sourceSpec
      NextPreimagePackage.privateStart env).mp spartanSpec
  refine {
    iteration := ?_
    initialState := fun index => ?_ }
  · simpa [Layout.Stage1.NextPreimageInputs.sourceInterface] using
      sourceSpec.iteration
  · simpa [Layout.Stage1.NextPreimageInputs.sourceInterface] using
      sourceSpec.initialState index

@[simp] theorem authority_relationWords
    (program : Lifecycle.Stage1.Application.Program)
    (nifsKeyWords commitmentKeyWords : List F) :
    (authority program nifsKeyWords commitmentKeyWords).relationWords =
      (structuralPackageIdentity program).toList := by
  rfl

@[simp] theorem authority_applicationWords
    (program : Lifecycle.Stage1.Application.Program)
    (nifsKeyWords commitmentKeyWords : List F) :
    (authority program nifsKeyWords commitmentKeyWords).applicationWords =
      applicationAuthorityWords program := by
  rfl

@[simp] theorem authority_nifsKeyWords
    (program : Lifecycle.Stage1.Application.Program)
    (nifsKeyWords commitmentKeyWords : List F) :
    (authority program nifsKeyWords commitmentKeyWords).nifsKeyWords =
      nifsKeyWords := by
  rfl

@[simp] theorem authority_commitmentKeyWords
    (program : Lifecycle.Stage1.Application.Program)
    (nifsKeyWords commitmentKeyWords : List F) :
    (authority program nifsKeyWords commitmentKeyWords).commitmentKeyWords =
      commitmentKeyWords := by
  rfl

theorem verificationKeyBinding_packageIdentity
    (program : Lifecycle.Stage1.Application.Program)
    (nifsKeyWords commitmentKeyWords : List F) :
    (verificationKeyBinding program nifsKeyWords commitmentKeyWords
      ).packageIdentity =
        packageIdentity program nifsKeyWords commitmentKeyWords := by
  rfl

theorem verificationKeyBinding_context
    (program : Lifecycle.Stage1.Application.Program)
    (nifsKeyWords commitmentKeyWords : List F) :
    (verificationKeyBinding program nifsKeyWords commitmentKeyWords).context =
      verifierContextDescriptor program nifsKeyWords commitmentKeyWords := by
  rfl

@[simp] theorem basePackage_rowCount_eq :
    basePackage.layout.rowCount = 29218024 := by
  rw [basePackage, Data.circuitPackage_layout]
  rfl

@[simp] theorem basePackage_totalColumnCount_eq :
    basePackage.layout.totalColumnCount = 29336725 := by
  rw [basePackage, Data.circuitPackage_layout]
  rfl

/-- Exact application-only physical row budget in the `2^28` package. -/
theorem package_rowCount_le_twoPow28_iff
    (program : Lifecycle.Stage1.Application.Program) :
    (package program).layout.rowCount ≤ 2 ^ Lifecycle.cubeVariables ↔
      (applicationPlan program).rowCount ≤ 239217427 := by
  rw [package_rowCount, basePackage_rowCount_eq]
  norm_num [Lifecycle.cubeVariables]
  omega

/-- Exact application-only private-column budget in the `2^28` package. -/
theorem package_totalColumnCount_le_twoPow28_iff
    (program : Lifecycle.Stage1.Application.Program) :
    (package program).layout.totalColumnCount ≤ 2 ^ Lifecycle.cubeVariables ↔
      addedPrivateColumnCount program ≤ 239098731 := by
  rw [package_totalColumnCount, basePackage_totalColumnCount_eq]
  norm_num [Lifecycle.cubeVariables]
  omega

/-- Exact finite bounds that one concrete application must discharge. -/
structure FitsTwoPow28 (program : Lifecycle.Stage1.Application.Program) : Prop where
  rows : (package program).layout.rowCount ≤ 2 ^ Lifecycle.cubeVariables
  columns : (package program).layout.totalColumnCount ≤
    2 ^ Lifecycle.cubeVariables

/-- Construct the physical package fit from application-only row and private
column bounds. -/
def fitsTwoPow28OfApplicationBounds
    (program : Lifecycle.Stage1.Application.Program)
    (rows : (applicationPlan program).rowCount ≤ 239217427)
    (columns : addedPrivateColumnCount program ≤ 239098731) :
    FitsTwoPow28 program where
  rows := (package_rowCount_le_twoPow28_iff program).2 rows
  columns := (package_totalColumnCount_le_twoPow28_iff program).2 columns

def jointDomain (program : Lifecycle.Stage1.Application.Program) : Nat :=
  max (package program).layout.rowCount
    (package program).layout.totalColumnCount

theorem jointDomain_le_twoPow28
    (program : Lifecycle.Stage1.Application.Program)
    (fits : FitsTwoPow28 program) :
    jointDomain program ≤ 2 ^ Lifecycle.cubeVariables := by
  exact max_le fits.rows fits.columns

/-- The recursive public width is fixed before the application is selected;
only private columns and rows depend on its proved plan. -/
theorem recursivePublicWidth_fixed
    (program : Lifecycle.Stage1.Application.Program) :
    (package program).layout.publicColumnCount = 278 := by
  rw [package_publicColumnCount]
  rfl

end NightstreamFPrime.Export.Stage1.PerApplicationPackage
