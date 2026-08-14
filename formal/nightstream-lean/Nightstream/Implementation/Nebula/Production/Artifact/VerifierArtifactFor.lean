import Nightstream.Implementation.Nebula.FPrime.Manifest.BaseSchema
import Nightstream.Implementation.Nebula.Commitment.Core.ConfigAuthority
import Nightstream.Implementation.Nebula.Production.FPrime.Base.ChallengeAuthorityRowsFor
import Nightstream.Implementation.Nebula.Production.FPrime.Base.CurrentMemoryRowsFor
import Nightstream.Implementation.Nebula.Production.FPrime.Fresh.ClaimProducerFor
import Nightstream.Implementation.Nebula.Production.FPrime.Terminal.CompleteProgramFor
import Nightstream.Implementation.Nebula.Production.Artifact.RelationDimensions
import Nightstream.Implementation.Lowering.Goldilocks.TypedNumericRowBridge

/-!
Contract: one verifier-owned artifact for the complete generated Nebula V2
fresh relation and terminal relation.

The base arm, recursive arm, selector program, source compiler, CCS relation,
NIFS exponent, common fold, and terminal program are one dependency graph.
The important row identities are definitions. A proof caller cannot supply
different row lists and repair the mismatch with a semantic assumption.

The remaining equality fields compare independently generated identifiers,
columns, and manifests. They contain no row-satisfaction, verifier-acceptance,
extraction, or execution conclusion.

Assurance tier: generated-artifact schema.

Emits constraints: no. It owns the exact programs that emit constraints.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.ProductionVerifierArtifactFor

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation

/-- Structural inputs for compiling one exact numeric row list. These facts
bound columns and select the minimum Boolean-cube exponent. They do not state
that any assignment satisfies the rows. -/
structure SourceCompilerEvidence
    (privateWidth rowVariables : Nat) (rows : List Row) where
  sourceColumns : ProductionFreshRelationCompilerFor.NumericBridge.RowsBelow
    (ProductionFreshLowNormEncoding.sourceWidth privateWidth) rows
  loweredColumns : ProductionFreshRelationCompilerFor.NumericBridge.RowsBelow
    (ProductionFreshLowNormEncoding.logicalWidth privateWidth)
    (Nightstream.Implementation.R1CS.CenteredTernaryLinearCompiler.loweredRows
      (ProductionFreshLinearSubstitution.layout privateWidth) rows)
  rowDomain :
    Nightstream.Implementation.R1CS.SelectiveCcs.RelationProfile.ExactRowDomain
      rows.length rowVariables
  carrierFits : ProductionFreshLowNormEncoding.logicalWidth privateWidth <=
    2 ^ rowVariables

namespace SourceCompilerEvidence

/-- The compiler program is constructed from the exact row-list parameter.
Its `rows` field cannot be replaced by a caller-selected list. -/
def program
    {privateWidth rowVariables : Nat} {rows : List Row}
    (evidence : SourceCompilerEvidence privateWidth rowVariables rows) :
    ProductionFreshRelationCompilerFor.SourceProgram privateWidth
      rowVariables where
  rows := rows
  sourceColumns := evidence.sourceColumns
  loweredColumns := evidence.loweredColumns
  rowDomain := evidence.rowDomain
  carrierFits := evidence.carrierFits

@[simp] theorem program_rows
    {privateWidth rowVariables : Nat} {rows : List Row}
    (evidence : SourceCompilerEvidence privateWidth rowVariables rows) :
    evidence.program.rows = rows := rfl

end SourceCompilerEvidence

/-- One complete verifier artifact. The relation arms occur in the branch
layout's type, and the source compiler consumes the branch rows by definition.
The resulting CCS relation artifact is also derived by definition below. -/
structure Artifact (candidate : Id)
    (baseWidths : FullClaimEnvelope.CompilerWidths) where
  dimensions : ProductionRelationDimensions.Artifact candidate
  base : BaseManifestSchema.Artifact baseWidths
  baseMemory : ProductionBaseCurrentMemoryRowsFor.Authority candidate base
  privateWidth : Nat
  branchLayout : ProductionFreshFPrimeBranchRows.Layout base.programRows
    dimensions.recursiveRows
  compiler : SourceCompilerEvidence privateWidth
    dimensions.relationRowVariables
    (ProductionFreshFPrimeBranchRows.rows branchLayout)
  operationsShape : Phi81Relation.Shape
  snapshotShape : Phi81Relation.Shape
  configAuthority : ProductCommitmentConfigAuthority.Authority base.seedManifest
    (ProductPaperAlgebraFor.FullShape dimensions.relationRowVariables
      (ProductionFreshLowNormEncoding.logicalWidth privateWidth)
      (ProductionFreshLowNormEncoding.publicFits privateWidth))
    operationsShape snapshotShape
  terminalTypedProgram : ProductionPaperTerminalCompleteProgramFor.Program
    candidate configAuthority.config dimensions.terminalProgram
  terminalColumnIndex :
    Nightstream.Implementation.Lowering.Goldilocks.ColumnId -> Nat
  terminalTypedEmbedding :
    Nightstream.Implementation.Lowering.Goldilocks.TypedNumericRowBridge.SplitEmbedding
      terminalColumnIndex terminalTypedProgram.foldFrame.rows
      (terminalTypedProgram.childrenRows compiler.program.relationArtifact.system)
      dimensions.terminalRows
  terminalTypedColumnsBelow :
    ProductionRelationDimensions.TerminalRowsBelow
      dimensions.terminalAssignmentWidth
      (Nightstream.Implementation.Lowering.Goldilocks.TypedNumericRowBridge.rows
        terminalColumnIndex
        (terminalTypedProgram.foldFrame.rows ++
          terminalTypedProgram.childrenRows compiler.program.relationArtifact.system))
  statementId : ProductConcreteNifsFor.StatementId
  baseChallengeProgram : ProductionBaseChallengeAuthorityRowsFor.Program
    candidate dimensions.relationRowVariables
  baseChallengeRowsMatched : baseChallengeProgram.MatchesArtifact base
  baseChallengeStatementIdExact :
    baseChallengeProgram.statementId = statementId
  baseChallengeStatementIdentityExact :
    baseChallengeProgram.statementIdentity =
      dimensions.coreProgram.statementIdentity
  verifierKeyIdentity :
    Nightstream.Protocol.Nebula.Soundness.VerifierKeyIdentity Digest.Value
  /-- The verifier key used by the recursive memory-challenge authority is
  the complete verifier key owned by this artifact. -/
  recursiveStatementVerifierKeyExact :
    dimensions.coreProgram.statementIdentity.verifierKey = verifierKeyIdentity
  verifierKeyDigestExact :
    verifierKeyIdentity.digest = dimensions.verifierKeyDigest
  relationManifestDigestExact :
    verifierKeyIdentity.relationManifestDigest =
      dimensions.relationManifestDigest
  terminalManifestDigestExact :
    verifierKeyIdentity.terminalManifestDigest =
      dimensions.terminalManifestDigest
  baseProfileExact : base.profile = dimensions.profile
  baseVerifierKeyExact :
    base.verifierKeyDigest = dimensions.verifierKeyDigest
  baseRelationManifestExact :
    base.relationManifestDigest = dimensions.relationManifestDigest
  baseRowVariablesExact :
    base.rowVariableCount = dimensions.relationRowVariables
  seedManifestExact :
    dimensions.coreProgram.fold.seedManifest = base.seedManifest
  baseIterationColumnExact :
    branchLayout.iterationZero.iterationColumn =
      base.layouts.baseIteration.iterationColumn
  recursiveIterationColumnExact :
    branchLayout.iterationZero.iterationColumn =
      dimensions.coreProgram.fold.priorLayout.state.invocationColumn
  recursiveStatementIdExact :
    dimensions.coreProgram.fold.statementId = statementId

namespace Artifact

abbrev LogicalWidth
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : Artifact candidate baseWidths) :=
  ProductionFreshLowNormEncoding.logicalWidth artifact.privateWidth

abbrev PublicFits
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : Artifact candidate baseWidths) :=
  ProductionFreshLowNormEncoding.publicFits artifact.privateWidth

abbrev Assignment
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : Artifact candidate baseWidths) :=
  ProductPaperAlgebraFor.Assignment artifact.dimensions.relationRowVariables
    artifact.LogicalWidth artifact.PublicFits

/-- The commitment map is derived from the verifier-owned seed manifest and
whole-ring lane layout. It is not an independent artifact field. -/
def config
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : Artifact candidate baseWidths) :
    ProductPaperAlgebraFor.Config artifact.dimensions.relationRowVariables
      artifact.LogicalWidth artifact.PublicFits artifact.operationsShape
      artifact.snapshotShape :=
  artifact.configAuthority.config

@[simp] theorem config_lanes
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : Artifact candidate baseWidths) :
    artifact.config.lanes = artifact.configAuthority.lanes := rfl

@[simp] theorem config_fullKey
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : Artifact candidate baseWidths) :
    artifact.config.fullKey = artifact.configAuthority.agreement.fullKey := rfl

@[simp] theorem config_operationsKey
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : Artifact candidate baseWidths) :
    artifact.config.operationsKey =
      artifact.configAuthority.agreement.operationsKey := rfl

@[simp] theorem config_snapshotKey
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : Artifact candidate baseWidths) :
    artifact.config.snapshotKey =
      artifact.configAuthority.agreement.snapshotKey := rfl

/-- Exact selector-gated augmented program. Its arms are definitionally the
generated base rows and generated recursive rows. -/
def fPrimeProgram
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : Artifact candidate baseWidths) :
    ProductionFreshFPrimeBranchRows.Program where
  baseRows := artifact.base.programRows
  recursiveRows := artifact.dimensions.recursiveRows
  layout := artifact.branchLayout

@[simp] theorem fPrimeProgram_baseRows
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : Artifact candidate baseWidths) :
    artifact.fPrimeProgram.baseRows = artifact.base.programRows := rfl

@[simp] theorem fPrimeProgram_recursiveRows
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : Artifact candidate baseWidths) :
    artifact.fPrimeProgram.recursiveRows =
      artifact.dimensions.recursiveRows := rfl

@[simp] theorem fPrimeProgram_rows
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : Artifact candidate baseWidths) :
    artifact.fPrimeProgram.rows =
      ProductionFreshFPrimeBranchRows.rows artifact.branchLayout := rfl

/-- Exact source compiler for the complete selector-gated F-prime program. -/
def sourceProgram
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : Artifact candidate baseWidths) :
    ProductionFreshRelationCompilerFor.SourceProgram artifact.privateWidth
      artifact.dimensions.relationRowVariables :=
  artifact.compiler.program

@[simp] theorem sourceProgram_rows
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : Artifact candidate baseWidths) :
    artifact.sourceProgram.rows = artifact.fPrimeProgram.rows := rfl

/-- The relation artifact is generated from the exact source compiler. There
is no separately supplied matrix family. -/
def relationArtifact
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : Artifact candidate baseWidths) :=
  artifact.sourceProgram.relationArtifact

/-- Exact numeric lowering of the complete verifier-owned typed terminal
program. This list includes the common terminal fold and all fourteen
same-witness opening and CE programs. -/
noncomputable def terminalTypedNumericRows
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : Artifact candidate baseWidths) : List Row :=
  Nightstream.Implementation.Lowering.Goldilocks.TypedNumericRowBridge.rows
    artifact.terminalColumnIndex
    (artifact.terminalTypedProgram.foldFrame.rows ++
      artifact.terminalTypedProgram.childrenRows artifact.relationArtifact.system)

/-- Canonical typed view of one generated numeric terminal assignment. -/
def terminalTypedAssignment
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : Artifact candidate baseWidths)
    (assignment : Nat -> Nat) :
    Nightstream.Implementation.Lowering.Goldilocks.ColumnId -> F :=
  Nightstream.Implementation.Lowering.Goldilocks.TypedNumericRowBridge.typedAssignment
    artifact.terminalColumnIndex assignment

/-- The generated numeric terminal relation contains the complete lowered
typed terminal program in exact source order. -/
theorem terminalTypedProgramIncluded
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : Artifact candidate baseWidths) :
    artifact.terminalTypedNumericRows.Sublist
      artifact.dimensions.terminalRows := by
  exact artifact.terminalTypedEmbedding.included

/-- All columns used by the complete lowered typed terminal program belong to
the finite terminal assignment. -/
theorem terminalTypedProgramColumnsScoped
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : Artifact candidate baseWidths) :
    ProductionRelationDimensions.TerminalRowsBelow
      artifact.dimensions.terminalAssignmentWidth
      artifact.terminalTypedNumericRows := by
  exact artifact.terminalTypedColumnsBelow

/-- Exact relation authority derived from this artifact. All program and
matrix equalities reduce by definition. -/
def relationAuthority
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : Artifact candidate baseWidths) :
    ProductionFreshClaimProducerFor.RelationAuthority artifact.PublicFits
      artifact.relationArtifact where
  privateWidth := artifact.privateWidth
  widthExact := rfl
  program := artifact.sourceProgram
  fPrimeProgram := artifact.fPrimeProgram
  programRowsExact := rfl
  artifactExact := HEq.rfl

/-- The selected exponent is exact for the complete gated F-prime source
program, not only for the recursive core or for a row-count lower bound. -/
theorem exactAugmentedRowDomain
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : Artifact candidate baseWidths) :
    Nightstream.Implementation.R1CS.SelectiveCcs.RelationProfile.ExactRowDomain
      artifact.fPrimeProgram.rows.length
      artifact.dimensions.relationRowVariables :=
  artifact.compiler.rowDomain

theorem augmentedRowsFit
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : Artifact candidate baseWidths) :
    artifact.fPrimeProgram.rows.length <=
      2 ^ artifact.dimensions.relationRowVariables :=
  artifact.exactAugmentedRowDomain.1

/-- Profile identity shared by the base and dimension artifacts. -/
theorem baseProfileSelected
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : Artifact candidate baseWidths) :
    artifact.base.profile = identity candidate :=
  artifact.baseProfileExact.trans artifact.dimensions.profileExact

/-- The base seed schedule and recursive fold use the same selected profile. -/
theorem recursiveSeedProfileSelected
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : Artifact candidate baseWidths) :
    artifact.dimensions.coreProgram.fold.seedManifest.profile =
      identity candidate := by
  rw [artifact.seedManifestExact, artifact.base.seedManifestProfile,
    artifact.baseProfileSelected]

/-- The full public verifier-key identity names the same aggregate key digest
as the generated relation artifact. This is an exact static link, not a
collision-resistance claim. -/
theorem selectedVerifierKeyDigest
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : Artifact candidate baseWidths) :
    artifact.verifierKeyIdentity.digest =
      artifact.dimensions.verifierKeyDigest :=
  artifact.verifierKeyDigestExact

/-- The public key identity names the exact generated relation manifest. -/
theorem selectedRelationManifestDigest
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : Artifact candidate baseWidths) :
    artifact.verifierKeyIdentity.relationManifestDigest =
      artifact.dimensions.relationManifestDigest :=
  artifact.relationManifestDigestExact

/-- The public key identity names the exact generated terminal manifest. -/
theorem selectedTerminalManifestDigest
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : Artifact candidate baseWidths) :
    artifact.verifierKeyIdentity.terminalManifestDigest =
      artifact.dimensions.terminalManifestDigest :=
  artifact.terminalManifestDigestExact

/-- Exact decoded branch of one arbitrary accepted low-norm carrier. This is
not restricted to the honest encoder image. -/
def ExactGeneratedBranch
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : Artifact candidate baseWidths)
    (assignment : artifact.Assignment) : Prop :=
  exists exactAssignment : artifact.Assignment,
    HEq assignment exactAssignment /\
      let decoded := artifact.sourceProgram.decodedSourceAssignment
        exactAssignment
      (decoded artifact.branchLayout.iterationZero.iterationColumn = 0 /\
          Satisfies artifact.base.programRows decoded) \/
        (0 < decoded artifact.branchLayout.iterationZero.iterationColumn /\
          Satisfies artifact.dimensions.recursiveRows decoded)

/-- The generic reverse compiler result and the generated-artifact branch
result are the same proposition because all row identities are definitions. -/
theorem exactDecodedBranch_iff_generated
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : Artifact candidate baseWidths)
    (assignment : artifact.Assignment) :
    ProductionFreshClaimProducerFor.RelationAuthority.ExactDecodedBranch
        artifact.relationAuthority assignment <->
      ExactGeneratedBranch artifact assignment := by
  rfl

/-- Arbitrary CCS satisfaction selects the exact generated base or recursive
arm. The constant-one coordinate remains an explicit verifier premise. -/
theorem selectedGeneratedBranchOfCcsPublic
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : Artifact candidate baseWidths)
    (assignment : artifact.Assignment)
    (publicInput : ProductPaperAlgebraFor.PublicInput
      artifact.dimensions.relationRowVariables artifact.LogicalWidth
      artifact.PublicFits)
    (publicExact : Phi81Relation.projectPublicInput assignment = publicInput)
    (publicZero : publicInput
      ⟨0, by
        change 0 < 540
        decide⟩ = 1)
    (relation :
      (ProductPaperAlgebraFor.semantics artifact.config).ccsSatisfied
        (ProductPaperAlgebraFor.matrixSource
          artifact.relationArtifact.system) assignment) :
    ExactGeneratedBranch artifact assignment := by
  rw [← artifact.exactDecodedBranch_iff_generated assignment]
  exact
    ProductionFreshClaimProducerFor.RelationAuthority.selectedBranchOfCcsPublic
      artifact.config artifact.relationArtifact artifact.relationAuthority
      assignment publicInput publicExact publicZero relation

/-- Stronger release result: a recursive branch also satisfies the exact
mandatory core included in the generated recursive rows. -/
def ExactGeneratedCoreBranch
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : Artifact candidate baseWidths)
    (assignment : artifact.Assignment) : Prop :=
  exists exactAssignment : artifact.Assignment,
    HEq assignment exactAssignment /\
      let decoded := artifact.sourceProgram.decodedSourceAssignment
        exactAssignment
      (decoded artifact.branchLayout.iterationZero.iterationColumn = 0 /\
          Satisfies artifact.base.programRows decoded) \/
        (0 < decoded artifact.branchLayout.iterationZero.iterationColumn /\
          Satisfies artifact.dimensions.coreProgram.rows decoded)

theorem generatedBranch_implies_coreBranch
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    {artifact : Artifact candidate baseWidths}
    {assignment : artifact.Assignment}
    (branch : ExactGeneratedBranch artifact assignment) :
    ExactGeneratedCoreBranch artifact assignment := by
  rcases branch with ⟨exactAssignment, assignmentExact, branch⟩
  refine ⟨exactAssignment, assignmentExact, ?_⟩
  rcases branch with base | recursive
  . exact Or.inl base
  . exact Or.inr ⟨recursive.1,
      artifact.dimensions.selected_exponent_core_satisfied recursive.2⟩

/-- Satisfaction of the generated terminal rows implies the exact terminal
program. That terminal program contains the same common fold as the recursive
core by construction in `ProductionRelationDimensions.Artifact`. -/
theorem terminalProgramSatisfied
    {candidate : Id} {baseWidths : FullClaimEnvelope.CompilerWidths}
    (artifact : Artifact candidate baseWidths)
    {assignment : Nat -> Nat}
    (satisfied : Satisfies artifact.dimensions.terminalRows assignment) :
    Satisfies artifact.dimensions.terminalProgram.rows assignment :=
  artifact.dimensions.terminal_program_satisfied satisfied

end Artifact

/-! ## Necessity countermodel -/

/-- A count-only link does not identify an authority-bearing row list. -/
def SameRowCountOnly (left right : List Row) : Prop :=
  left.length = right.length

/-- Equal row counts accept two different one-row relations. The first rejects
the all-zero assignment; the second accepts it. -/
theorem sameRowCountOnly_accepts_different_relations :
    SameRowCountOnly
      [ProductionRecursiveCoreManifestFor.rejectingConstantRow]
      [ProductionRecursiveCoreManifestFor.zeroRow] /\
    [ProductionRecursiveCoreManifestFor.rejectingConstantRow] ≠
      [ProductionRecursiveCoreManifestFor.zeroRow] := by
  simp [SameRowCountOnly,
    ProductionRecursiveCoreManifestFor.rejectingConstantRow,
    ProductionRecursiveCoreManifestFor.zeroRow]

end Nightstream.Implementation.Nebula.ProductionVerifierArtifactFor
