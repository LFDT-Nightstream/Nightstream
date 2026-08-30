import NightstreamFPrime.Export.Stage1.PerApplicationFixedPoint
import NightstreamFPrime.Export.Stage1.PerApplicationPackageSourceCustody
import NightstreamFPrime.Lifecycle.Stage1.VerificationKey

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

/-- Exact key-facing relation metadata for the self-derived matrix plan. -/
def recursiveRelation (program : Program) (fits : FitsTwoPow28 program) :
    CcsRelation :=
  productionCcsRelation
    (PerApplicationFixedPoint.structuralPlan program fits).rowCount
    (PerApplicationFixedPoint.logicalWidth program)
    Lifecycle.cubeVariables

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

/-- Prefix-free canonical value for one Goldilocks ring element. -/
def ringValue (value : RingF) : Value :=
  .array ((List.ofFn value).map fun coefficient => .atom coefficient.val)

/-- Prefix-free canonical value for the actual semantic Ajtai key. -/
def commitmentKeyValue {program : Program}
    (key : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program)) : Value :=
  .array ((List.ofFn key).map fun row =>
    .array ((List.ofFn row).map ringValue))

/-- Complete raw canonical token stream of the actual Ajtai key. This is the
authority input. Its component digest is only a compression of these words. -/
def commitmentKeyWords {program : Program}
    (key : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program)) : List F :=
  Package.valuePreimage (commitmentKeyValue key)

/-- Schema of the canonical per-application package envelope. The inner
`CircuitPackage` and application plan keep their own schema versions. -/
def sealedPackageSchema : Nat := 2

/-- One prefix-free authority value that carries the physical circuit package,
the exact compact 14-matrix program, and the exact Lean-authored application
plan. Rust must decode these children; it must not reconstruct application
layout or rows. -/
def sealedPackageValue (program : Program)
    (fits : FitsTwoPow28 program) : Value :=
  .array [
    .atom sealedPackageSchema,
    CircuitPackage.format.encode (package program fits),
    MatrixProgram.Program.format.encode
      (PerApplicationMatrixProgram.matrixProgram program),
    ApplicationPackage.Plan.format.encode
      (PerApplicationPackage.applicationPlan program)]

def structuralPackageIdentity (program : Program)
    (fits : FitsTwoPow28 program) : VerifierContext.Digest4 :=
  VerifierContext.Digest4.ofList
    (Package.relationIdentifierValue (sealedPackageValue program fits))

theorem sealedPackageValue_exact (program : Program)
    (fits : FitsTwoPow28 program) :
    sealedPackageValue program fits = .array [
      .atom sealedPackageSchema,
      CircuitPackage.format.encode (package program fits),
      MatrixProgram.Program.format.encode
        (PerApplicationMatrixProgram.matrixProgram program),
      ApplicationPackage.Plan.format.encode
        (PerApplicationPackage.applicationPlan program)] := by
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
        PerApplicationPackage.applicationPlan rightProgram := by
  have packageEncoded := congrArg (fun value =>
    match value with
    | .array (_schema :: encodedPackage :: _matrix :: _application :: []) =>
        encodedPackage
    | _ => .array []) same
  have matrixEncoded := congrArg (fun value =>
    match value with
    | .array (_schema :: _package :: encodedMatrix :: _application :: []) =>
        encodedMatrix
    | _ => .array []) same
  have applicationEncoded := congrArg (fun value =>
    match value with
    | .array (_schema :: _package :: _matrix :: encodedApplication :: []) =>
        encodedApplication
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
    · have decoded :=
        congrArg ApplicationPackage.Plan.format.decode applicationEncoded
      simpa [sealedPackageValue,
        ApplicationPackage.Plan.format.decode_encode] using decoded

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

/-- Raw relation authority is the canonical relation metadata followed by a
recomputed digest of the complete circuit-and-matrix envelope. The envelope
remains authoritative; the digest is never accepted without verifier-side
recomputation or pinning. -/
def relationAuthorityWords (program : Program)
    (fits : FitsTwoPow28 program) : List F :=
  Package.valuePreimage (relationShapeValue program fits) ++
    (structuralPackageIdentity program fits).toList

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
    (key : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program)) : List F :=
  nifsKeyDomain ++
    VerifierContext.framed (relationAuthorityWords program fits) ++
    VerifierContext.framed VerifierContext.profileWords ++
    VerifierContext.framed VerifierContext.scheduleWords ++
    VerifierContext.framed
      (VerifierContext.componentDigest 4 (commitmentKeyWords key)).toList

/-- Raw verifier authority computed only from the exact package, application,
and semantic Ajtai key. There are no caller-supplied authority words. -/
def authority {program : Program} (fits : FitsTwoPow28 program)
    (key : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program)) :
    VerifierContext.Authority where
  relationWords := relationAuthorityWords program fits
  applicationWords := applicationAuthorityWords program
  nifsKeyWords := nifsKeyWords fits key
  commitmentKeyWords := commitmentKeyWords key

def verifierContextDescriptor {program : Program}
    (fits : FitsTwoPow28 program)
    (key : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program)) :
    VerifierContext.Descriptor :=
  VerifierContext.descriptor (authority fits key)

def packageIdentityDomain : List F :=
  ([78, 105, 103, 104, 116, 115, 116, 114, 101, 97, 109, 47,
    70, 80, 114, 105, 109, 101, 47, 115, 101, 97, 108, 101, 100,
    45, 112, 97, 99, 107, 97, 103, 101, 47, 118, 50] : List Nat).map
      Poseidon2.ofNat

def packageIdentityPreimage {program : Program}
    (fits : FitsTwoPow28 program)
    (key : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program)) : List F :=
  packageIdentityDomain ++
    VerifierContext.framed (structuralPackageIdentity program fits).toList ++
    VerifierContext.framed (verifierContextDescriptor fits key).serialize

/-- Equality of final binding preimages identifies both acyclic digest
components. Fixed-length framing prevents either component from consuming
words from the other. -/
theorem packageIdentityPreimage_components
    {leftProgram rightProgram : Program}
    (leftFits : FitsTwoPow28 leftProgram)
    (rightFits : FitsTwoPow28 rightProgram)
    (leftKey : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth leftProgram)
      (publicFits := PerApplicationFixedPoint.publicFits leftProgram))
    (rightKey : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth rightProgram)
      (publicFits := PerApplicationFixedPoint.publicFits rightProgram))
    (same : packageIdentityPreimage leftFits leftKey =
      packageIdentityPreimage rightFits rightKey) :
    structuralPackageIdentity leftProgram leftFits =
        structuralPackageIdentity rightProgram rightFits ∧
      verifierContextDescriptor leftFits leftKey =
        verifierContextDescriptor rightFits rightKey := by
  have body :
      VerifierContext.framed
          (structuralPackageIdentity leftProgram leftFits).toList ++
          VerifierContext.framed
            (verifierContextDescriptor leftFits leftKey).serialize =
        VerifierContext.framed
          (structuralPackageIdentity rightProgram rightFits).toList ++
          VerifierContext.framed
            (verifierContextDescriptor rightFits rightKey).serialize := by
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
relation, physical package, and semantic commitment key. -/
def packageIdentity {program : Program} (fits : FitsTwoPow28 program)
    (key : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program)) :
    VerifierContext.Digest4 :=
  VerifierContext.Digest4.ofList
    (Poseidon2.hash (packageIdentityPreimage fits key))

def verificationKeyBinding {program : Program}
    (fits : FitsTwoPow28 program)
    (key : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program)) :
    Lifecycle.Stage1.VerificationKey.Binding :=
  Lifecycle.Stage1.VerificationKey.ofAuthority
    (packageIdentity fits key) (authority fits key)

@[simp] theorem authority_relationWords {program : Program}
    (fits : FitsTwoPow28 program)
    (key : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program)) :
    (authority fits key).relationWords =
      relationAuthorityWords program fits := by
  rfl

@[simp] theorem authority_applicationWords {program : Program}
    (fits : FitsTwoPow28 program)
    (key : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program)) :
    (authority fits key).applicationWords =
      applicationAuthorityWords program := by
  rfl

@[simp] theorem authority_nifsKeyWords {program : Program}
    (fits : FitsTwoPow28 program)
    (key : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program)) :
    (authority fits key).nifsKeyWords = nifsKeyWords fits key := by
  rfl

@[simp] theorem authority_commitmentKeyWords {program : Program}
    (fits : FitsTwoPow28 program)
    (key : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program)) :
    (authority fits key).commitmentKeyWords = commitmentKeyWords key := by
  rfl

theorem packageIdentity_recomputed {program : Program}
    (fits : FitsTwoPow28 program)
    (key : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program)) :
    (packageIdentity fits key).toList =
      (VerifierContext.Digest4.ofList
        (Poseidon2.hash (packageIdentityPreimage fits key))).toList := by
  rfl

@[simp] theorem verificationKeyBinding_packageIdentity {program : Program}
    (fits : FitsTwoPow28 program)
    (key : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program)) :
    (verificationKeyBinding fits key).packageIdentity =
      packageIdentity fits key := by
  rfl

@[simp] theorem verificationKeyBinding_context {program : Program}
    (fits : FitsTwoPow28 program)
    (key : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program)) :
    (verificationKeyBinding fits key).context =
      verifierContextDescriptor fits key := by
  rfl

theorem verificationKeyDigest_recomputed {program : Program}
    (fits : FitsTwoPow28 program)
    (key : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program)) :
    (verificationKeyBinding fits key).digest =
      (VerifierContext.Digest4.ofList
        (Poseidon2.hash (verificationKeyBinding fits key).serialize)).toList := by
  exact Lifecycle.Stage1.VerificationKey.Binding.digest_recomputed _

end NightstreamFPrime.Export.Stage1.PerApplicationCanonicalPackage
