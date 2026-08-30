import NightstreamFPrime.Export.RowSemantics
import NightstreamFPrime.Export.Stage1.ApplicationPackage
import NightstreamFPrime.Export.Stage1.CompactRows
import NightstreamFPrime.Export.Stage1.Data
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
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec

/- Schema roles owned by the per-application extension. -/
namespace Role

def applicationWitness : Nat := 17
def applicationLocal : Nat := 18

end Role

def basePackage : CircuitPackage := Data.circuitPackage ()

/-- Exact application plan whose rows start after the validated prefix. -/
def applicationPlan (program : Lifecycle.Stage1.Application.Program) :
    ApplicationPackage.Plan :=
  ApplicationPackage.productionPlan program basePackage.layout.rowCount

/-- New caller-owned witness words plus every application-generated private
column. -/
def addedPrivateColumnCount
    (program : Lifecycle.Stage1.Application.Program) : Nat :=
  program.witnessWordCount + (applicationPlan program).privateCount

/-- Existing private columns stay fixed. The former constant and every public
column move after the application-private suffix. -/
def shiftColumn (program : Lifecycle.Stage1.Application.Program)
    (column : Nat) : Nat :=
  if column < basePackage.layout.constantColumn then column
  else column + addedPrivateColumnCount program

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
  rowCount := basePackage.layout.rowCount + (applicationPlan program).rowCount
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
    assertionRows := basePackage.assertionRows.map (shiftSparseRow program) ++
      plan.assertionRows
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
      basePackage.layout.rowCount + (applicationPlan program).rowCount := by
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
      basePackage.assertionRows.map (shiftSparseRow program) ++
        (applicationPlan program).assertionRows := by
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
        exact List.mem_map_of_mem member)
    exact (shiftSparseRow_holds program assertion env).mp shifted

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

/-- Exact finite bounds that one concrete application must discharge. -/
structure FitsTwoPow28 (program : Lifecycle.Stage1.Application.Program) : Prop where
  rows : (package program).layout.rowCount ≤ 2 ^ Lifecycle.cubeVariables
  columns : (package program).layout.totalColumnCount ≤
    2 ^ Lifecycle.cubeVariables

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
