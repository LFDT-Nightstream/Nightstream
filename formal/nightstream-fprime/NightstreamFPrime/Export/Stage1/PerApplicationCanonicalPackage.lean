import NightstreamFPrime.Export.Stage1.PerApplicationFixedPoint
import NightstreamFPrime.Export.SharedFormulas
import NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransport
import NightstreamFPrime.Export.Stage1.PerApplicationPackageSourceCustody
import NightstreamFPrime.Export.StreamingIdentity
import NightstreamFPrime.Lifecycle.Stage1.VerificationKey
import NightstreamFPrime.Spec.AjtaiSetupV1
import NightstreamFPrime.Export.MatrixProgram.Program

/-! Rust must still implement the proved direct-plan expansion and pin or
allowlist the final identity. -/

namespace NightstreamFPrime.Export.Stage1.PerApplicationCanonicalPackage

open NightstreamFPrime.Export
open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

abbrev Program := Lifecycle.Stage1.Application.Program

abbrev FitsTwoPow28 (program : Program) :=
  PerApplicationFixedPoint.FitsTwoPow28 program

abbrev CommitmentSetup (program : Program) :=
  AjtaiSetupV1.Setup productionProfile.commitmentWidth
    (Phi81ColumnLayout.blockCount
      (Phi81CarrierLayout.carrierWidth
        (PerApplicationFixedPoint.logicalWidth program)))

def commitmentKey {program : Program} (setup : CommitmentSetup program) :
    AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program) :=
  setup.verifierKey

def commitmentKeyWords {program : Program}
    (setup : CommitmentSetup program) : List F :=
  setup.authorityWords

def recursiveRelation (program : Program) (fits : FitsTwoPow28 program) :
    CcsRelation :=
  productionCcsRelation
    (PerApplicationFixedPoint.structuralPlan program fits).rowCount
    (PerApplicationFixedPoint.logicalWidth program)
    Lifecycle.cubeVariables

def directStructuralRowCount (program : Program) : Nat :=
  998638 + (PerApplicationPackage.directApplicationPlan program).rowCount + 9

theorem directStructuralRowCount_eq
    (program : Program) (fits : FitsTwoPow28 program) :
    directStructuralRowCount program =
      (PerApplicationFixedPoint.structuralPlan program fits).rowCount := by
  rw [directStructuralRowCount,
    PerApplicationPackage.directApplicationPlan_eq_applicationPlan,
    PerApplicationFixedPoint.structuralPlan_rowCount]

def directLogicalWidth (program : Program) : Nat :=
  43738906 +
    (program.witnessWordCount + ApplicationRetainedBlocks.localCount program) * 41

theorem directLogicalWidth_eq (program : Program) :
    directLogicalWidth program = PerApplicationFixedPoint.logicalWidth program := by
  unfold directLogicalWidth PerApplicationFixedPoint.logicalWidth
  exact (ApplicationRetainedGeometry.completeLogicalWidth_eq_applicationCounts
    program).symm

def directRecursiveRelation (program : Program) : CcsRelation :=
  productionCcsRelation (directStructuralRowCount program)
    (directLogicalWidth program) Lifecycle.cubeVariables

theorem directRecursiveRelation_eq_recursiveRelation
    (program : Program) (fits : FitsTwoPow28 program) :
    directRecursiveRelation program = recursiveRelation program fits := by
  unfold directRecursiveRelation recursiveRelation
  rw [directStructuralRowCount_eq program fits, directLogicalWidth_eq]

private def recursiveRelationFast (program : Program)
    (_fits : FitsTwoPow28 program) : CcsRelation :=
  directRecursiveRelation program

@[csimp] theorem recursiveRelation_eq_recursiveRelationFast :
    @recursiveRelation = @recursiveRelationFast := by
  funext program fits
  exact (directRecursiveRelation_eq_recursiveRelation program fits).symm

def replaceRelation (source : CircuitPackage) (relation : CcsRelation) :
    CircuitPackage :=
  { source with relation := relation, terminal := none }

@[simp] theorem replaceRelation_witnessInstructions (source : CircuitPackage)
    (relation : CcsRelation) :
    (replaceRelation source relation).witnessInstructions =
      source.witnessInstructions := by
  rfl

@[simp] theorem replaceRelation_assertionRows (source : CircuitPackage)
    (relation : CcsRelation) :
    (replaceRelation source relation).assertionRows = source.assertionRows := by
  rfl

def package (program : Program) (fits : FitsTwoPow28 program) :
    CircuitPackage :=
  TerminalPackage.install (replaceRelation
    (PerApplicationPackage.package program) (recursiveRelation program fits))

def directTerminalLayout (program : Program) : TerminalLayout where
  rowStart := 0
  rowCount := (directRecursiveRelation program).rowCount
  runningClaims := productionShape.runningCount
  freshClaims := productionShape.freshCount

theorem directTerminalLayout_eq_layoutFor_package
    (program : Program) (fits : FitsTwoPow28 program) :
    directTerminalLayout program = TerminalPackage.layoutFor
      (package program fits) := by
  unfold directTerminalLayout TerminalPackage.layoutFor
  rw [directRecursiveRelation_eq_recursiveRelation]
  rfl

private theorem rowsHold_replaceRelation (source : CircuitPackage)
    (relation : CcsRelation) (env : Circuit.Env) :
    (TerminalPackage.install
        { source with relation := relation, terminal := none }).RowsHold env ↔
      source.RowsHold env := by
  rfl

theorem rowsHold_iff_sourcePackage (program : Program)
    (fits : FitsTwoPow28 program) (env : Circuit.Env) :
    (package program fits).RowsHold env ↔
      (PerApplicationPackage.package program).RowsHold env := by
  exact rowsHold_replaceRelation _ _ env

@[simp] theorem package_relation (program : Program)
    (fits : FitsTwoPow28 program) :
    (package program fits).relation = recursiveRelation program fits := by
  rfl

@[simp] theorem package_terminal (program : Program)
    (fits : FitsTwoPow28 program) :
    (package program fits).terminal =
      some (TerminalPackage.layoutFor (package program fits)) := by
  simp only [package, TerminalPackage.install, TerminalPackage.layoutFor]

@[simp] theorem terminal_rowCount (program : Program)
    (fits : FitsTwoPow28 program) :
    ((package program fits).terminal.getD
      (TerminalPackage.layoutFor (package program fits))).rowCount =
        (PerApplicationFixedPoint.structuralPlan program fits).rowCount := by
  rfl

@[simp] theorem recursiveRelation_rowCount (program : Program)
    (fits : FitsTwoPow28 program) :
    (recursiveRelation program fits).rowCount =
      (PerApplicationFixedPoint.structuralPlan program fits).rowCount := by
  rfl

@[simp] theorem recursiveRelation_columnCount (program : Program)
    (fits : FitsTwoPow28 program) :
    (recursiveRelation program fits).columnCount =
      PerApplicationFixedPoint.logicalWidth program := by
  rfl

@[simp] theorem recursiveRelation_cubeVariables (program : Program)
    (fits : FitsTwoPow28 program) :
    (recursiveRelation program fits).cubeVariables =
      Lifecycle.cubeVariables := by
  rfl

theorem logicalRelation_matrices (program : Program)
    (fits : FitsTwoPow28 program) :
    (PerApplicationFixedPoint.relation program fits).matrices =
      (PerApplicationFixedPoint.structuralPlan program fits).matrix := by
  exact PerApplicationFixedPoint.relation_matrices program fits

def sourceRow (program : Program) (fits : FitsTwoPow28 program) :
    Nat → Option R1CS.Row :=
  PackageSourceRows.packageSourceRow? (package program fits)

theorem sourceRow_eq_packageSource (program : Program)
    (fits : FitsTwoPow28 program) :
    sourceRow program fits =
      PerApplicationPackageSourceCustody.sourceRow program := by
  have witnesses : (package program fits).witnessInstructions =
      (PerApplicationPackage.package program).witnessInstructions := by
    rw [package, TerminalPackage.install_witnessInstructions,
      replaceRelation_witnessInstructions]
  have assertions : (package program fits).assertionRows =
      (PerApplicationPackage.package program).assertionRows := by
    rw [package, TerminalPackage.install_assertionRows,
      replaceRelation_assertionRows]
  funext rowIndex
  unfold sourceRow PerApplicationPackageSourceCustody.sourceRow
    PackageSourceRows.packageSourceRow?
  rw [witnesses, assertions]

theorem matrixProgram_exact (program : Program)
    (fits : FitsTwoPow28 program) :
    PerApplicationMatrixProgramSemantics.Exact
      (PerApplicationMatrixProgram.matrixProgram program)
      (PerApplicationFixedPoint.structuralPlan program fits)
      (sourceRow program fits) := by
  rw [sourceRow_eq_packageSource program fits]
  exact PerApplicationMatrixProgramSemantics.matrixProgramExact program fits
    (PerApplicationPackageSourceCustody.sourceRow program)
    (PerApplicationPackageSourceCustody.custody program fits)

@[simp] theorem matrixProgram_rowCount (program : Program)
    (fits : FitsTwoPow28 program) :
    (PerApplicationMatrixProgram.matrixProgram program).rowCount =
      (recursiveRelation program fits).rowCount := by
  exact PerApplicationMatrixProgram.matrixProgram_rowCount_eq_structuralPlan
    program fits

theorem matrixProgram_row? (program : Program)
    (fits : FitsTwoPow28 program)
    (row : Fin (PerApplicationFixedPoint.structuralPlan program fits).rowCount) :
    (PerApplicationMatrixProgram.matrixProgram program).row?
        (PerApplicationFixedPoint.logicalWidth program)
        (sourceRow program fits) row.val =
      some ((PerApplicationFixedPoint.structuralPlan program fits).forms row) :=
  (matrixProgram_exact program fits).row? row

def nextPreimageRange (program : Program) : MatrixProgram.IndexRange :=
  ⟨PerApplicationPackage.nextPreimageRowStart program, 5⟩

@[simp] theorem nextPreimageRange_count (program : Program) :
    (nextPreimageRange program).count = 5 := by
  rfl

theorem nextPreimageRange_startsAtApplicationEnd (program : Program) :
    (nextPreimageRange program).start =
      (PerApplicationPackage.applicationPlan program).rowStart +
        (PerApplicationPackage.applicationPlan program).rowCount := by
  rfl

theorem nextPreimageRange_endsAtPackageEnd (program : Program)
    (fits : FitsTwoPow28 program) :
    (nextPreimageRange program).endExclusive =
      (package program fits).layout.rowCount := by
  rfl

def logicalPublicInputCount : Nat := ProductionAssignment.publicWidth

@[simp] theorem logicalPublicInputCount_eq : logicalPublicInputCount = 270 :=
  ProductionAssignment.publicWidth_eq

def sealedPackageSchema : Nat := 6

/-- Rust must decode these children; it must not reconstruct relation,
assignment, or application layout. -/
def sealedPackageValue (program : Program)
    (fits : FitsTwoPow28 program) : Value :=
  .array [
    .atom sealedPackageSchema,
    CircuitPackage.format.encode (package program fits),
    MatrixProgram.Program.format.encode
      (PerApplicationMatrixProgram.matrixProgram program),
    ApplicationPackage.Plan.format.encode
      (PerApplicationPackage.applicationPlan program),
    PerApplicationAssignmentTransport.Plan.format.encode
      (PerApplicationAssignmentTransport.canonical program),
    MatrixProgram.IndexRange.format.encode (nextPreimageRange program),
    .atom logicalPublicInputCount]

theorem sealedPackageValue_exact (program : Program)
    (fits : FitsTwoPow28 program) :
    sealedPackageValue program fits = .array [
      .atom sealedPackageSchema,
      CircuitPackage.format.encode (package program fits),
      MatrixProgram.Program.format.encode
        (PerApplicationMatrixProgram.matrixProgram program),
      ApplicationPackage.Plan.format.encode
        (PerApplicationPackage.applicationPlan program),
      PerApplicationAssignmentTransport.Plan.format.encode
        (PerApplicationAssignmentTransport.canonical program),
      MatrixProgram.IndexRange.format.encode
        (nextPreimageRange program),
      .atom logicalPublicInputCount] := by
  rfl

theorem sealedPackageValue_components
    (leftProgram rightProgram : Program)
    (leftFits : FitsTwoPow28 leftProgram)
    (rightFits : FitsTwoPow28 rightProgram)
    (same : sealedPackageValue leftProgram leftFits =
      sealedPackageValue rightProgram rightFits) :
    package leftProgram leftFits = package rightProgram rightFits ∧
      PerApplicationMatrixProgram.matrixProgram leftProgram =
        PerApplicationMatrixProgram.matrixProgram rightProgram ∧
      PerApplicationPackage.applicationPlan leftProgram =
        PerApplicationPackage.applicationPlan rightProgram ∧
      PerApplicationAssignmentTransport.canonical leftProgram =
        PerApplicationAssignmentTransport.canonical rightProgram ∧
      nextPreimageRange leftProgram = nextPreimageRange rightProgram := by
  have packageEncoded := congrArg (fun value =>
    match value with
    | .array (_schema :: encodedPackage :: _matrix :: _application ::
        _assignment :: _nextPreimage :: _publicInputCount :: []) =>
        encodedPackage
    | _ => .array []) same
  have matrixEncoded := congrArg (fun value =>
    match value with
    | .array (_schema :: _package :: encodedMatrix :: _application ::
        _assignment :: _nextPreimage :: _publicInputCount :: []) =>
        encodedMatrix
    | _ => .array []) same
  have applicationEncoded := congrArg (fun value =>
    match value with
    | .array (_schema :: _package :: _matrix :: encodedApplication ::
        _assignment :: _nextPreimage :: _publicInputCount :: []) =>
        encodedApplication
    | _ => .array []) same
  have assignmentEncoded := congrArg (fun value =>
    match value with
    | .array (_schema :: _package :: _matrix :: _application ::
        encodedAssignment :: _nextPreimage :: _publicInputCount :: []) =>
        encodedAssignment
    | _ => .array []) same
  have nextPreimageEncoded := congrArg (fun value =>
    match value with
    | .array (_schema :: _package :: _matrix :: _application ::
        _assignment :: encodedNextPreimage :: _publicInputCount :: []) =>
        encodedNextPreimage
    | _ => .array []) same
  constructor
  · have decoded := congrArg CircuitPackage.format.decode packageEncoded
    simpa [sealedPackageValue, CircuitPackage.format.decode_encode] using
      decoded
  · constructor
    · have decoded :=
        congrArg MatrixProgram.Program.format.decode matrixEncoded
      simpa [sealedPackageValue,
        MatrixProgram.Program.format.decode_encode] using decoded
    · constructor
      · have decoded :=
          congrArg ApplicationPackage.Plan.format.decode applicationEncoded
        simpa [sealedPackageValue,
          ApplicationPackage.Plan.format.decode_encode] using decoded
      · constructor
        · have decoded := congrArg
              PerApplicationAssignmentTransport.Plan.format.decode
              assignmentEncoded
          simpa [sealedPackageValue,
            PerApplicationAssignmentTransport.Plan.format.decode_encode] using
              decoded
        · have decoded :=
            congrArg MatrixProgram.IndexRange.format.decode nextPreimageEncoded
          simpa [sealedPackageValue,
            MatrixProgram.IndexRange.format.decode_encode] using decoded

def applicationAuthorityWords (program : Program) : List F :=
  ApplicationPackage.authorityWords
    (PerApplicationPackage.applicationPlan program)

/-- The application component of the verifier context. It is the only hash
of the application plan: the structural identity takes it in place of the
plan. -/
def applicationDigest (program : Program) : VerifierContext.Digest4 :=
  VerifierContext.componentDigest 2 (applicationAuthorityWords program)

def digestValue (digest : VerifierContext.Digest4) : Value :=
  .array (digest.toList.map fun word => .atom word.val)

theorem digestValue_injective : Function.Injective digestValue := by
  intro left right equal
  rcases left with ⟨left0, left1, left2, left3⟩
  rcases right with ⟨right0, right1, right2, right3⟩
  simp only [digestValue, VerifierContext.Digest4.toList, List.map_cons,
    List.map_nil, Value.array.injEq, List.cons.injEq, Value.atom.injEq,
    Fin.val_inj, and_true] at equal
  obtain ⟨same0, same1, same2, same3⟩ := equal
  rw [same0, same1, same2, same3]

/-- The package without the application witness batches, witness
instructions and assertion rows. -/
def fixedPackage (program : Program) (fits : FitsTwoPow28 program) :
    CircuitPackage :=
  { package program fits with
    witnessBatches := PerApplicationPackage.basePackage.witnessBatches.map
      (PerApplicationPackage.shiftBatch program)
    witnessInstructions :=
      PerApplicationPackage.basePackage.witnessInstructions.map
        (PerApplicationPackage.shiftWitnessInstruction program)
    assertionRows :=
      PerApplicationPackage.basePackage.assertionRows.map
          (PerApplicationPackage.shiftSparseRow program) ++
        NextPreimagePackage.assertionRows
          (PerApplicationPackage.nextPreimageRowStart program) }

private theorem encode_withLists (source : CircuitPackage)
    (batches : List Circuit.WitnessBatch)
    (instructions : List WitnessInstruction) (rows : List SparseRow) :
    CircuitPackage.format.encode { source with
        witnessBatches := batches
        witnessInstructions := instructions
        assertionRows := rows } =
      .array [
        .atom source.schemaVersion,
        Profile.format.encode source.profile,
        PoseidonSchedule.format.encode source.poseidon,
        PhysicalLayout.format.encode source.layout,
        CcsRelation.format.encode source.relation,
        PermutationTemplate.format.encode source.permutation,
        (list HashChain.format).encode source.hashChains,
        (list PermutationInvocation.format).encode source.permutationInvocations,
        (list CompactRowTemplate.format).encode source.compactRowTemplates,
        (list CompactRowInvocation.format).encode source.compactRowInvocations,
        (list WitnessBatch.format).encode batches,
        (list WitnessInstruction.format).encode instructions,
        (list SparseRow.format).encode rows,
        (option TerminalLayout.format).encode source.terminal] := by
  rfl

/-- The encoding of `fixedPackage`, field by field. -/
theorem fixedPackage_encode (program : Program) (fits : FitsTwoPow28 program) :
    CircuitPackage.format.encode (fixedPackage program fits) =
      .array [
        .atom (package program fits).schemaVersion,
        Profile.format.encode (package program fits).profile,
        PoseidonSchedule.format.encode (package program fits).poseidon,
        PhysicalLayout.format.encode (package program fits).layout,
        CcsRelation.format.encode (package program fits).relation,
        PermutationTemplate.format.encode (package program fits).permutation,
        (list HashChain.format).encode (package program fits).hashChains,
        (list PermutationInvocation.format).encode
          (package program fits).permutationInvocations,
        (list CompactRowTemplate.format).encode
          (package program fits).compactRowTemplates,
        (list CompactRowInvocation.format).encode
          (package program fits).compactRowInvocations,
        (list WitnessBatch.format).encode
          (PerApplicationPackage.basePackage.witnessBatches.map
            (PerApplicationPackage.shiftBatch program)),
        (list WitnessInstruction.format).encode
          (PerApplicationPackage.basePackage.witnessInstructions.map
            (PerApplicationPackage.shiftWitnessInstruction program)),
        (list SparseRow.format).encode
          (PerApplicationPackage.basePackage.assertionRows.map
              (PerApplicationPackage.shiftSparseRow program) ++
            NextPreimagePackage.assertionRows
              (PerApplicationPackage.nextPreimageRowStart program)),
        (option TerminalLayout.format).encode (package program fits).terminal] := by
  unfold fixedPackage
  exact encode_withLists (package program fits) _ _ _

/-- The rows of `NextPreimagePackage.assertionRows`, the last package rows. -/
def nextPreimageRowCount : Nat := 5

/-- The package of a fixed package and an application plan. The plan batches
and witness instructions go last. The plan rows go before the last
`nextPreimageRowCount` rows. -/
def reassemble (fixed : CircuitPackage) (plan : ApplicationPackage.Plan) :
    CircuitPackage :=
  { fixed with
    witnessBatches := fixed.witnessBatches ++ plan.witnessBatches
    witnessInstructions :=
      fixed.witnessInstructions ++ plan.witnessInstructions
    assertionRows :=
      (fixed.assertionRows.take
          (fixed.assertionRows.length - nextPreimageRowCount) ++
        plan.assertionRows) ++
      fixed.assertionRows.drop
        (fixed.assertionRows.length - nextPreimageRowCount) }

private theorem reassemble_parts (source : CircuitPackage)
    (plan : ApplicationPackage.Plan) (batches : List Circuit.WitnessBatch)
    (instructions : List WitnessInstruction) (rows next : List SparseRow)
    (batchesSplit : source.witnessBatches = batches ++ plan.witnessBatches)
    (instructionsSplit :
      source.witnessInstructions = instructions ++ plan.witnessInstructions)
    (rowsSplit : source.assertionRows = (rows ++ plan.assertionRows) ++ next)
    (nextLength : next.length = nextPreimageRowCount) :
    reassemble { source with
        witnessBatches := batches
        witnessInstructions := instructions
        assertionRows := rows ++ next } plan = source := by
  have split : (rows ++ next).length - nextPreimageRowCount = rows.length := by
    rw [List.length_append, nextLength, Nat.add_sub_cancel]
  cases source
  dsimp only at batchesSplit instructionsSplit rowsSplit
  subst batchesSplit instructionsSplit rowsSplit
  unfold reassemble
  simp only [split, List.take_left, List.drop_left]

@[simp] theorem replaceRelation_witnessBatches (source : CircuitPackage)
    (relation : CcsRelation) :
    (replaceRelation source relation).witnessBatches =
      source.witnessBatches := by
  rfl

private theorem install_witnessBatches (source : CircuitPackage) :
    (TerminalPackage.install source).witnessBatches =
      source.witnessBatches := by
  rfl

theorem reassemble_fixedPackage (program : Program)
    (fits : FitsTwoPow28 program) :
    reassemble (fixedPackage program fits)
        (PerApplicationPackage.applicationPlan program) =
      package program fits := by
  unfold fixedPackage
  refine reassemble_parts (package program fits)
    (PerApplicationPackage.applicationPlan program) _ _ _ _ ?_ ?_ ?_ ?_
  · rw [package, install_witnessBatches, replaceRelation_witnessBatches]
    exact PerApplicationPackage.package_witnessBatches program
  · rw [package, TerminalPackage.install_witnessInstructions,
      replaceRelation_witnessInstructions]
    exact PerApplicationPackage.package_witnessInstructions program
  · rw [package, TerminalPackage.install_assertionRows,
      replaceRelation_assertionRows]
    exact PerApplicationPackage.package_assertionRows program
  · exact NextPreimagePackage.assertionRows_length _

theorem package_eq_of_fixedPackage_eq {leftProgram rightProgram : Program}
    (leftFits : FitsTwoPow28 leftProgram)
    (rightFits : FitsTwoPow28 rightProgram)
    (fixedSame : fixedPackage leftProgram leftFits =
      fixedPackage rightProgram rightFits)
    (planSame : PerApplicationPackage.applicationPlan leftProgram =
      PerApplicationPackage.applicationPlan rightProgram) :
    package leftProgram leftFits = package rightProgram rightFits := by
  rw [← reassemble_fixedPackage leftProgram leftFits,
    ← reassemble_fixedPackage rightProgram rightFits, fixedSame, planSame]

/-- The value that the structural identity hashes: `sealedPackageValue` with
the fixed package in place of the package and the application digest in
place of the application plan. -/
def structuralValue (program : Program) (fits : FitsTwoPow28 program) :
    Value :=
  .array [
    .atom sealedPackageSchema,
    CircuitPackage.format.encode (fixedPackage program fits),
    MatrixProgram.Program.format.encode
      (PerApplicationMatrixProgram.matrixProgram program),
    digestValue (applicationDigest program),
    PerApplicationAssignmentTransport.Plan.format.encode
      (PerApplicationAssignmentTransport.canonical program),
    MatrixProgram.IndexRange.format.encode (nextPreimageRange program),
    .atom logicalPublicInputCount]

private theorem encode_injective {Alpha : Type} (format : Format Alpha) :
    Function.Injective format.encode := by
  intro left right equal
  have decoded := congrArg format.decode equal
  rw [format.decode_encode, format.decode_encode] at decoded
  exact Except.ok.inj decoded

theorem structuralValue_components
    (leftProgram rightProgram : Program)
    (leftFits : FitsTwoPow28 leftProgram)
    (rightFits : FitsTwoPow28 rightProgram)
    (same : structuralValue leftProgram leftFits =
      structuralValue rightProgram rightFits) :
    fixedPackage leftProgram leftFits = fixedPackage rightProgram rightFits ∧
      PerApplicationMatrixProgram.matrixProgram leftProgram =
        PerApplicationMatrixProgram.matrixProgram rightProgram ∧
      applicationDigest leftProgram = applicationDigest rightProgram ∧
      PerApplicationAssignmentTransport.canonical leftProgram =
        PerApplicationAssignmentTransport.canonical rightProgram ∧
      nextPreimageRange leftProgram = nextPreimageRange rightProgram := by
  have values := Value.array.inj same
  simp only [List.cons.injEq] at values
  obtain ⟨_, fixed, matrix, digest, assignment, nextPreimage, _⟩ := values
  exact ⟨encode_injective _ fixed, encode_injective _ matrix,
    digestValue_injective digest, encode_injective _ assignment,
    encode_injective _ nextPreimage⟩

/-- The structural value and the application plan fix the sealed package
value. -/
theorem sealedPackageValue_eq_of_structuralValue_eq
    {leftProgram rightProgram : Program}
    (leftFits : FitsTwoPow28 leftProgram)
    (rightFits : FitsTwoPow28 rightProgram)
    (structuralSame : structuralValue leftProgram leftFits =
      structuralValue rightProgram rightFits)
    (planSame : PerApplicationPackage.applicationPlan leftProgram =
      PerApplicationPackage.applicationPlan rightProgram) :
    sealedPackageValue leftProgram leftFits =
      sealedPackageValue rightProgram rightFits := by
  obtain ⟨fixed, matrix, _, assignment, nextPreimage⟩ :=
    structuralValue_components leftProgram rightProgram leftFits rightFits
      structuralSame
  unfold sealedPackageValue
  rw [package_eq_of_fixedPackage_eq leftFits rightFits fixed planSame,
    matrix, planSame, assignment, nextPreimage]

def structuralPackageIdentity (program : Program)
    (fits : FitsTwoPow28 program) : VerifierContext.Digest4 :=
  VerifierContext.Digest4.ofList
    (Package.relationIdentifierValue (structuralValue program fits))

def structuralPackageIdentityFast (program : Program)
    (fits : FitsTwoPow28 program) : VerifierContext.Digest4 :=
  VerifierContext.Digest4.ofList
    (StreamingIdentity.relationIdentifierValueFast
      (structuralValue program fits))

theorem structuralPackageIdentityFast_eq (program : Program)
    (fits : FitsTwoPow28 program) :
    structuralPackageIdentityFast program fits =
      structuralPackageIdentity program fits := by
  unfold structuralPackageIdentityFast structuralPackageIdentity
  rw [StreamingIdentity.relationIdentifierValueFast_eq]

theorem structuralPackageIdentity_recomputed (program : Program)
    (fits : FitsTwoPow28 program) :
    (structuralPackageIdentity program fits).toList =
      (VerifierContext.Digest4.ofList
        (Package.relationIdentifierValue
          (structuralValue program fits))).toList := by
  rfl

def relationShapeValue (program : Program)
    (fits : FitsTwoPow28 program) : Value :=
  CcsRelation.format.encode (recursiveRelation program fits)

def directRelationShapeValue (program : Program) : Value :=
  CcsRelation.format.encode (directRecursiveRelation program)

theorem directRelationShapeValue_eq (program : Program)
    (fits : FitsTwoPow28 program) :
    directRelationShapeValue program = relationShapeValue program fits := by
  unfold directRelationShapeValue relationShapeValue
  rw [directRecursiveRelation_eq_recursiveRelation]

def relationAuthorityWords (program : Program)
    (fits : FitsTwoPow28 program) : List F :=
  Package.valuePreimage (relationShapeValue program fits) ++
    (structuralPackageIdentity program fits).toList

def relationAuthorityWordsFromStructural (program : Program)
    (fits : FitsTwoPow28 program)
    (structural : VerifierContext.Digest4) : List F :=
  Package.valuePreimage (relationShapeValue program fits) ++ structural.toList

@[simp] theorem relationAuthorityWordsFromStructural_canonical
    (program : Program) (fits : FitsTwoPow28 program) :
    relationAuthorityWordsFromStructural program fits
        (structuralPackageIdentity program fits) =
      relationAuthorityWords program fits := by
  rfl

def nifsKeyDomain : List F :=
  ([78, 105, 103, 104, 116, 115, 116, 114, 101, 97, 109, 47,
    70, 80, 114, 105, 109, 101, 47, 110, 105, 102, 115, 45,
    107, 101, 121, 47, 118, 49, 95, 49] : List Nat).map Poseidon2.ofNat

def nifsKeyWords {program : Program} (fits : FitsTwoPow28 program)
    (setup : CommitmentSetup program) : List F :=
  nifsKeyDomain ++
    VerifierContext.framed (relationAuthorityWords program fits) ++
    VerifierContext.framed VerifierContext.profileWords ++
    VerifierContext.framed VerifierContext.scheduleWords ++
    VerifierContext.framed
      (VerifierContext.componentDigest 4 (commitmentKeyWords setup)).toList

def nifsKeyWordsFromStructural {program : Program}
    (fits : FitsTwoPow28 program) (setup : CommitmentSetup program)
    (structural : VerifierContext.Digest4) : List F :=
  nifsKeyDomain ++
    VerifierContext.framed
      (relationAuthorityWordsFromStructural program fits structural) ++
    VerifierContext.framed VerifierContext.profileWords ++
    VerifierContext.framed VerifierContext.scheduleWords ++
    VerifierContext.framed
      (VerifierContext.componentDigest 4 (commitmentKeyWords setup)).toList

@[simp] theorem nifsKeyWordsFromStructural_canonical {program : Program}
    (fits : FitsTwoPow28 program) (setup : CommitmentSetup program) :
    nifsKeyWordsFromStructural fits setup
        (structuralPackageIdentity program fits) =
      nifsKeyWords fits setup := by
  rfl

def authority {program : Program} (fits : FitsTwoPow28 program)
    (setup : CommitmentSetup program) :
    VerifierContext.Authority where
  relationWords := relationAuthorityWords program fits
  applicationWords := applicationAuthorityWords program
  nifsKeyWords := nifsKeyWords fits setup
  commitmentKeyWords := commitmentKeyWords setup

def authorityFromStructural {program : Program}
    (fits : FitsTwoPow28 program) (setup : CommitmentSetup program)
    (structural : VerifierContext.Digest4) :
    VerifierContext.Authority where
  relationWords := relationAuthorityWordsFromStructural program fits structural
  applicationWords := applicationAuthorityWords program
  nifsKeyWords := nifsKeyWordsFromStructural fits setup structural
  commitmentKeyWords := commitmentKeyWords setup

@[simp] theorem authorityFromStructural_canonical {program : Program}
    (fits : FitsTwoPow28 program) (setup : CommitmentSetup program) :
    authorityFromStructural fits setup
        (structuralPackageIdentity program fits) =
      authority fits setup := by
  rfl

def verifierContextDescriptor {program : Program}
    (fits : FitsTwoPow28 program)
    (setup : CommitmentSetup program) :
    VerifierContext.Descriptor :=
  VerifierContext.descriptor (authority fits setup)

def verifierContextDigest {program : Program}
    (fits : FitsTwoPow28 program)
    (setup : CommitmentSetup program) : KeyDigest :=
  (verifierContextDescriptor fits setup).digest4.toList

def verifierContextDescriptorFromStructural {program : Program}
    (fits : FitsTwoPow28 program) (setup : CommitmentSetup program)
    (structural : VerifierContext.Digest4) :
    VerifierContext.Descriptor :=
  VerifierContext.descriptor
    (authorityFromStructural fits setup structural)

def verifierContextDescriptorFromStructuralAndApplicationDigest
    {program : Program} (fits : FitsTwoPow28 program)
    (setup : CommitmentSetup program)
    (structural applicationDigest : VerifierContext.Digest4) :
    VerifierContext.Descriptor where
  relation := VerifierContext.componentDigest 1
    (relationAuthorityWordsFromStructural program fits structural)
  application := applicationDigest
  nifsKey := VerifierContext.componentDigest 3
    (nifsKeyWordsFromStructural fits setup structural)
  commitmentKey := VerifierContext.componentDigest 4
    (commitmentKeyWords setup)

@[simp] theorem
    verifierContextDescriptorFromStructuralAndApplicationDigest_canonical
    {program : Program} (fits : FitsTwoPow28 program)
    (setup : CommitmentSetup program)
    (structural : VerifierContext.Digest4) :
    verifierContextDescriptorFromStructuralAndApplicationDigest fits setup
        structural
        (VerifierContext.componentDigest 2
          (applicationAuthorityWords program)) =
      verifierContextDescriptorFromStructural fits setup structural := by
  rfl

@[simp] theorem verifierContextDescriptorFromStructural_canonical
    {program : Program} (fits : FitsTwoPow28 program)
    (setup : CommitmentSetup program) :
    verifierContextDescriptorFromStructural fits setup
        (structuralPackageIdentity program fits) =
      verifierContextDescriptor fits setup := by
  rfl

def packageIdentityDomain : List F :=
  ([78, 105, 103, 104, 116, 115, 116, 114, 101, 97, 109, 47,
    70, 80, 114, 105, 109, 101, 47, 115, 101, 97, 108, 101, 100,
    45, 112, 97, 99, 107, 97, 103, 101, 47, 118, 50] : List Nat).map
      Poseidon2.ofNat

def packageIdentityPreimage {program : Program}
    (fits : FitsTwoPow28 program)
    (setup : CommitmentSetup program) : List F :=
  packageIdentityDomain ++
    VerifierContext.framed (structuralPackageIdentity program fits).toList ++
    VerifierContext.framed (verifierContextDescriptor fits setup).serialize

theorem packageIdentityPreimage_components
    {leftProgram rightProgram : Program}
    (leftFits : FitsTwoPow28 leftProgram)
    (rightFits : FitsTwoPow28 rightProgram)
    (leftSetup : CommitmentSetup leftProgram)
    (rightSetup : CommitmentSetup rightProgram)
    (same : packageIdentityPreimage leftFits leftSetup =
      packageIdentityPreimage rightFits rightSetup) :
    structuralPackageIdentity leftProgram leftFits =
        structuralPackageIdentity rightProgram rightFits ∧
      verifierContextDescriptor leftFits leftSetup =
        verifierContextDescriptor rightFits rightSetup := by
  have body :
      VerifierContext.framed
          (structuralPackageIdentity leftProgram leftFits).toList ++
          VerifierContext.framed
            (verifierContextDescriptor leftFits leftSetup).serialize =
        VerifierContext.framed
          (structuralPackageIdentity rightProgram rightFits).toList ++
          VerifierContext.framed
            (verifierContextDescriptor rightFits rightSetup).serialize := by
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

def packageIdentity {program : Program} (fits : FitsTwoPow28 program)
    (setup : CommitmentSetup program) :
    VerifierContext.Digest4 :=
  VerifierContext.Digest4.ofList
    (Poseidon2.hash (packageIdentityPreimage fits setup))

private def packageIdentityFromParts
    (structural : VerifierContext.Digest4)
    (context : VerifierContext.Descriptor) :
    VerifierContext.Digest4 :=
  VerifierContext.Digest4.ofList
    (Poseidon2.hash
      (packageIdentityDomain ++
        VerifierContext.framed structural.toList ++
        VerifierContext.framed context.serialize))

def packageIdentityFromStructural {program : Program}
    (fits : FitsTwoPow28 program) (setup : CommitmentSetup program)
    (structural : VerifierContext.Digest4) :
    VerifierContext.Digest4 :=
  packageIdentityFromParts structural
    (verifierContextDescriptorFromStructural fits setup structural)

@[simp] theorem packageIdentityFromStructural_canonical {program : Program}
    (fits : FitsTwoPow28 program) (setup : CommitmentSetup program) :
    packageIdentityFromStructural fits setup
        (structuralPackageIdentity program fits) =
      packageIdentity fits setup := by
  rfl

def verificationKeyBinding {program : Program}
    (fits : FitsTwoPow28 program)
    (setup : CommitmentSetup program) :
    Lifecycle.Stage1.VerificationKey.Binding :=
  Lifecycle.Stage1.VerificationKey.ofAuthority
    (packageIdentity fits setup) (authority fits setup)
    (SharedFormulas.libraryDigest ())

def verificationKeyBindingFromStructural {program : Program}
    (fits : FitsTwoPow28 program) (setup : CommitmentSetup program)
    (structural : VerifierContext.Digest4) :
    Lifecycle.Stage1.VerificationKey.Binding :=
  let authority := authorityFromStructural fits setup structural
  let context := VerifierContext.descriptor authority
  {
    packageIdentity := packageIdentityFromParts structural context
    context := context
    formulas := SharedFormulas.libraryDigest ()
  }

def verificationKeyBindingFromStructuralAndApplicationDigest
    {program : Program} (fits : FitsTwoPow28 program)
    (setup : CommitmentSetup program)
    (structural applicationDigest : VerifierContext.Digest4) :
    Lifecycle.Stage1.VerificationKey.Binding :=
  let context :=
    verifierContextDescriptorFromStructuralAndApplicationDigest fits setup
      structural applicationDigest
  {
    packageIdentity := packageIdentityFromParts structural context
    context := context
    formulas := SharedFormulas.libraryDigest ()
  }

@[simp] theorem
    verificationKeyBindingFromStructuralAndApplicationDigest_canonical
    {program : Program} (fits : FitsTwoPow28 program)
    (setup : CommitmentSetup program)
    (structural : VerifierContext.Digest4) :
    verificationKeyBindingFromStructuralAndApplicationDigest fits setup
        structural
        (VerifierContext.componentDigest 2
          (applicationAuthorityWords program)) =
      verificationKeyBindingFromStructural fits setup structural := by
  rfl

@[simp] theorem verificationKeyBindingFromStructural_canonical
    {program : Program} (fits : FitsTwoPow28 program)
    (setup : CommitmentSetup program) :
    verificationKeyBindingFromStructural fits setup
        (structuralPackageIdentity program fits) =
      verificationKeyBinding fits setup := by
  rfl

@[simp] theorem authority_relationWords {program : Program}
    (fits : FitsTwoPow28 program)
    (setup : CommitmentSetup program) :
    (authority fits setup).relationWords =
      relationAuthorityWords program fits := by
  rfl

@[simp] theorem authority_applicationWords {program : Program}
    (fits : FitsTwoPow28 program)
    (setup : CommitmentSetup program) :
    (authority fits setup).applicationWords =
      applicationAuthorityWords program := by
  rfl

@[simp] theorem authority_nifsKeyWords {program : Program}
    (fits : FitsTwoPow28 program)
    (setup : CommitmentSetup program) :
    (authority fits setup).nifsKeyWords = nifsKeyWords fits setup := by
  rfl

@[simp] theorem authority_commitmentKeyWords {program : Program}
    (fits : FitsTwoPow28 program)
    (setup : CommitmentSetup program) :
    (authority fits setup).commitmentKeyWords = commitmentKeyWords setup := by
  rfl

theorem packageIdentity_recomputed {program : Program}
    (fits : FitsTwoPow28 program)
    (setup : CommitmentSetup program) :
    (packageIdentity fits setup).toList =
      (VerifierContext.Digest4.ofList
        (Poseidon2.hash (packageIdentityPreimage fits setup))).toList := by
  rfl

@[simp] theorem verificationKeyBinding_packageIdentity {program : Program}
    (fits : FitsTwoPow28 program)
    (setup : CommitmentSetup program) :
    (verificationKeyBinding fits setup).packageIdentity =
      packageIdentity fits setup := by
  rfl

@[simp] theorem verificationKeyBinding_context {program : Program}
    (fits : FitsTwoPow28 program)
    (setup : CommitmentSetup program) :
    (verificationKeyBinding fits setup).context =
      verifierContextDescriptor fits setup := by
  rfl

theorem verificationKeyDigest_recomputed {program : Program}
    (fits : FitsTwoPow28 program)
    (setup : CommitmentSetup program) :
    (verificationKeyBinding fits setup).digest =
      (VerifierContext.Digest4.ofList
        (Poseidon2.hash (verificationKeyBinding fits setup).serialize)).toList := by
  exact Lifecycle.Stage1.VerificationKey.Binding.digest_recomputed _

end NightstreamFPrime.Export.Stage1.PerApplicationCanonicalPackage
