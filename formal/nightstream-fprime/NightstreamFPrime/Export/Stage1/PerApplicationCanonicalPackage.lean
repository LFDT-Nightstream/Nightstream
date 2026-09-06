import NightstreamFPrime.Export.Stage1.PerApplicationFixedPoint
import NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransport
import NightstreamFPrime.Export.Stage1.PerApplicationPackageSourceCustody
import NightstreamFPrime.Export.StreamingIdentity
import NightstreamFPrime.Lifecycle.Stage1.VerificationKey
import NightstreamFPrime.Spec.AjtaiSetupV1

/-!
Owns the verifier-owned canonical package for one Lean-authored application.

The physical row program comes from `PerApplicationPackage`. Its `relation`
field is replaced with the exact dimensions of the self-derived 14-matrix
plan. The package identity and verifier context are then recomputed from this
canonical package, the exact application plan, and the actual Ajtai key.

This module does not authorize a Rust loader or proof backend. Rust must still
implement the proved direct-plan expansion and pin or allowlist the final
identity.
-/

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

/-- One verifier-owned indexed setup with dimensions derived from the exact
recursive fixed point. The setup seed is the only stored key material. -/
abbrev CommitmentSetup (program : Program) :=
  AjtaiSetupV1.Setup productionProfile.commitmentWidth
    (Phi81ColumnLayout.blockCount
      (Phi81CarrierLayout.carrierWidth
        (PerApplicationFixedPoint.logicalWidth program)))

/-- The semantic Ajtai key and the compact authority words are projections of
the same indexed setup. -/
def commitmentKey {program : Program} (setup : CommitmentSetup program) :
    AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program) :=
  setup.verifierKey

def commitmentKeyWords {program : Program}
    (setup : CommitmentSetup program) : List F :=
  setup.authorityWords

/-- Exact key-facing relation metadata for the self-derived matrix plan. -/
def recursiveRelation (program : Program) (fits : FitsTwoPow28 program) :
    CcsRelation :=
  productionCcsRelation
    (PerApplicationFixedPoint.structuralPlan program fits).rowCount
    (PerApplicationFixedPoint.logicalWidth program)
    Lifecycle.cubeVariables

def directStructuralRowCount (program : Program) : Nat :=
  6369850 + (PerApplicationPackage.directApplicationPlan program).rowCount + 9

theorem directStructuralRowCount_eq
    (program : Program) (fits : FitsTwoPow28 program) :
    directStructuralRowCount program =
      (PerApplicationFixedPoint.structuralPlan program fits).rowCount := by
  rw [directStructuralRowCount,
    PerApplicationPackage.directApplicationPlan_eq_applicationPlan,
    PerApplicationFixedPoint.structuralPlan_rowCount]

def directLogicalWidth (program : Program) : Nat :=
  256216447 +
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

/-- Native metadata evaluation uses the proved direct counts without building
the complete structural row plan. -/
@[csimp] theorem recursiveRelation_eq_recursiveRelationFast :
    @recursiveRelation = @recursiveRelationFast := by
  funext program fits
  exact (directRecursiveRelation_eq_recursiveRelation program fits).symm

/-- One canonical physical row program and one exact recursive relation. -/
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

/-- Changing only the key-facing relation metadata does not change physical
row satisfaction. -/
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
  rfl

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

/-- The semantic matrices named by the package are exactly the matrices of
the self-derived plan. -/
theorem logicalRelation_matrices (program : Program)
    (fits : FitsTwoPow28 program) :
    (PerApplicationFixedPoint.relation program fits).matrices =
      (PerApplicationFixedPoint.structuralPlan program fits).matrix := by
  exact PerApplicationFixedPoint.relation_matrices program fits

/-- The only source-row accessor for canonical matrix interpretation. -/
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

/-- The compact matrix interpreter selected by this application is exactly
the key-facing structural plan. Every ordinary row comes from this package. -/
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

/-- Exact physical owner of the five NextPreimage rows after the selected
application. The range is carried in the sealed authority value so Rust does
not infer this boundary. -/
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

/-- Exact recursive public prefix of the production low-norm relation. This
value is carried in the sealed envelope so Rust does not infer it from the
physical package's different 278-column public layout. -/
def logicalPublicInputCount : Nat := ProductionAssignment.publicWidth

@[simp] theorem logicalPublicInputCount_eq : logicalPublicInputCount = 270 :=
  ProductionAssignment.publicWidth_eq

/-- Schema of the canonical per-application package envelope. The inner
`CircuitPackage` and application plan keep their own schema versions. -/
def sealedPackageSchema : Nat := 6

/-- One prefix-free authority value that carries the physical circuit package,
the exact compact 14-matrix program, the exact Lean-authored application plan,
the retained-assignment transport plan, the exact NextPreimage row owner, and
the recursive public prefix length. Rust must decode these children; it must
not reconstruct relation, assignment, or application layout. -/
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

def structuralPackageIdentity (program : Program)
    (fits : FitsTwoPow28 program) : VerifierContext.Digest4 :=
  VerifierContext.Digest4.ofList
    (Package.relationIdentifierValue (sealedPackageValue program fits))

/-- Bounded-memory executable identity. The theorem below keeps
`structuralPackageIdentity` as the semantic authority. -/
def structuralPackageIdentityFast (program : Program)
    (fits : FitsTwoPow28 program) : VerifierContext.Digest4 :=
  VerifierContext.Digest4.ofList
    (StreamingIdentity.relationIdentifierValueFast
      (sealedPackageValue program fits))

theorem structuralPackageIdentityFast_eq (program : Program)
    (fits : FitsTwoPow28 program) :
    structuralPackageIdentityFast program fits =
      structuralPackageIdentity program fits := by
  unfold structuralPackageIdentityFast structuralPackageIdentity
  rw [StreamingIdentity.relationIdentifierValueFast_eq]

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

/-- Equality of sealed values identifies all authoritative children. The
codec round trips discharge injectivity without making either digest an
authority. -/
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

theorem structuralPackageIdentity_recomputed (program : Program)
    (fits : FitsTwoPow28 program) :
    (structuralPackageIdentity program fits).toList =
      (VerifierContext.Digest4.ofList
        (Package.relationIdentifierValue
          (sealedPackageValue program fits))).toList := by
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

/-- Raw relation authority is the canonical relation metadata followed by a
recomputed digest of the complete circuit-and-matrix envelope. The envelope
remains authoritative; the digest is never accepted without verifier-side
recomputation or pinning. -/
def relationAuthorityWords (program : Program)
    (fits : FitsTwoPow28 program) : List F :=
  Package.valuePreimage (relationShapeValue program fits) ++
    (structuralPackageIdentity program fits).toList

/-- Relation authority with an explicitly supplied structural digest. The
caller must recompute this digest from `sealedPackageValue`; the theorem below
shows the canonical specialization. -/
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

def applicationAuthorityWords (program : Program) : List F :=
  ApplicationPackage.authorityWords
    (PerApplicationPackage.applicationPlan program)

def nifsKeyDomain : List F :=
  ([78, 105, 103, 104, 116, 115, 116, 114, 101, 97, 109, 47,
    70, 80, 114, 105, 109, 101, 47, 110, 105, 102, 115, 45,
    107, 101, 121, 47, 118, 49, 95, 49] : List Nat).map Poseidon2.ofNat

/-- Canonical constructor inputs for `ProductionKey.key`. The NIFS key is a
fixed Lean definition, so these words bind its relation, profile, schedule,
and actual commitment key without serializing erased proof fields. -/
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

/-- Raw verifier authority computed only from the exact package, application,
and indexed Ajtai setup. There are no caller-supplied authority words. -/
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

/-- Canonical four-word verifier-context digest carried by recursive and
terminal state preimages. The full verification-key digest remains the
separate package-and-context binding defined below. -/
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

/-- Construct the verifier context from a separately recomputed application
component digest. The caller must prove that digest against the complete
application authority stream. -/
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

/-- Equality of final binding preimages identifies both acyclic digest
components. Fixed-length framing prevents either component from consuming
words from the other. -/
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

/-- Final verifier-owned identity for one exact application, recursive
relation, physical package, and indexed commitment setup. -/
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

def verificationKeyBindingFromStructural {program : Program}
    (fits : FitsTwoPow28 program) (setup : CommitmentSetup program)
    (structural : VerifierContext.Digest4) :
    Lifecycle.Stage1.VerificationKey.Binding :=
  let authority := authorityFromStructural fits setup structural
  let context := VerifierContext.descriptor authority
  {
    packageIdentity := packageIdentityFromParts structural context
    context := context
  }

/-- Construct the complete binding from independently recomputed structural
and application identities. Both remain evidence only after their raw
authority streams are checked. -/
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
