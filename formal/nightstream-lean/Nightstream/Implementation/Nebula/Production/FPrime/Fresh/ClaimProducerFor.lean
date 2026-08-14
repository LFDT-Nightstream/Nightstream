import Nightstream.Implementation.Nebula.Production.Application.ApplicationBatchBridge
import Nightstream.Implementation.Nebula.Production.FPrime.Fresh.FPrimeBranchRows
import Nightstream.Implementation.Nebula.Production.FPrime.Fresh.RelationCompilerFor
import Nightstream.Implementation.Nebula.Production.FPrime.Recursive.RecursiveSuccessorFor

/-!
Contract: exact producer-side construction of one delayed fresh claim at the
generated augmented-relation exponent.

The final low-norm assignment exists before its product commitment is put in
the claim. One `rowVariables` value selects the assignment, public projection,
compiled relation, and CCS statement.

`RelationAuthority` fixes one selector-gated F-prime source program and its
compiled relation artifact once for the verifier. `FreshRelationWitness`
retains only one assignment for that fixed program. Public projection, norm,
CCS satisfaction, and the base-or-recursive branch are derived. A per-claim
caller cannot select a different relation or branch relation.

Assurance tier: exponent-indexed producer model.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Nebula.ProductionFreshClaimProducerFor

open Nightstream.Implementation.Nebula
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.ApplicationBatch
open Nightstream.Protocol.Nebula.ProductionProfileCandidates
open Nightstream.Protocol.Nebula.WasmPublicStatementEncoding
open Nightstream.Protocol.Nebula.WasmStatement
open Nightstream.Protocol.Nebula.WasmState
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev FullShape
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits

abbrev FreshAssignment
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  ProductPaperAlgebraFor.Assignment rowVariables logicalWidth publicFits

noncomputable def ccsPublic
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (statementId : ProductPoseidon2.StatementId)
    (successor : ProductionSuccessorStateBinding.Value candidate fullShape)
    (memory : ProductionMemoryBatchPoseidonBinding.Batch candidate) :
    ProductionFieldNativeFullClaim.CcsPublic :=
  ProductionMemoryBoundCcsPublic.word
    (ProductionSuccessorStateBinding.outputDigest statementId successor) memory

noncomputable def value
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductPoseidon2.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (successor : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (memory : ProductionMemoryBatchPoseidonBinding.Batch candidate)
    (assignment : FreshAssignment rowVariables logicalWidth publicFits) :
    ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits) where
  ccsPublic := ccsPublic statementId successor memory
  commitmentBundle := ProductNifsCodec.protocolBundleOf
    (ProductCommitmentAlgebra.commit config assignment)
  recursiveState := successor.running
  memory := memory

@[simp] theorem value_running
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductPoseidon2.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (successor : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (memory : ProductionMemoryBatchPoseidonBinding.Batch candidate)
    (assignment : FreshAssignment rowVariables logicalWidth publicFits) :
    (value candidate statementId config successor memory
      assignment).recursiveState = successor.running := rfl

@[simp] theorem value_memory
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductPoseidon2.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (successor : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (memory : ProductionMemoryBatchPoseidonBinding.Batch candidate)
    (assignment : FreshAssignment rowVariables logicalWidth publicFits) :
    (value candidate statementId config successor memory
      assignment).memory = memory := rfl

theorem value_bundle_opens
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductPoseidon2.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (successor : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (memory : ProductionMemoryBatchPoseidonBinding.Batch candidate)
    (assignment : FreshAssignment rowVariables logicalWidth publicFits) :
    ProductNifsCodec.codecBundle
        (value candidate statementId config successor memory
          assignment).commitmentBundle =
      ProductCommitmentAlgebra.commit config assignment :=
  ProductNifsCodec.codecBundle_protocolBundleOf _

theorem value_ccs_fullMatches
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductPoseidon2.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (successor : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (memory : ProductionMemoryBatchPoseidonBinding.Batch candidate)
    (assignment : FreshAssignment rowVariables logicalWidth publicFits) :
    ProductionMemoryBoundCcsPublic.FullMatches
      (value candidate statementId config successor memory
        assignment).ccsPublic
      (ProductionSuccessorStateBinding.outputDigest statementId successor)
      memory :=
  ProductionMemoryBoundCcsPublic.word_fullMatches _ _

theorem value_memoryBound
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductPoseidon2.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (successor : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (memory : ProductionMemoryBatchPoseidonBinding.Batch candidate)
    (assignment : FreshAssignment rowVariables logicalWidth publicFits) :
    (value candidate statementId config successor memory
      assignment).MemoryBound :=
  ProductionMemoryBoundCcsPublic.word_memoryMatches _ _

theorem checkedBatch_memoryCanonical
    {candidate : Id}
    {layout : ProductionMemoryCheckedBatchRows.Layout candidate}
    {sourceAssignment : Nat -> Nat}
    {headers : FPrime.ChainHeaders Digest.Value}
    (result : ProductionMemoryCheckedBatchRows.Result layout sourceAssignment
      headers) :
    forall claim, claim ∈ result.suffixBatch.suffixes ->
      MemoryClaimCodec.Claim.Canonical claim := by
  intro claim member
  change claim ∈ List.ofFn result.claim at member
  rcases List.mem_ofFn.mp member with ⟨index, rfl⟩
  exact (result.claimParsed index).canonical

theorem value_canonical
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductPoseidon2.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (successor : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    {layout : ProductionMemoryCheckedBatchRows.Layout candidate}
    {sourceAssignment : Nat -> Nat}
    {headers : FPrime.ChainHeaders Digest.Value}
    (result : ProductionMemoryCheckedBatchRows.Result layout sourceAssignment
      headers)
    (assignment : FreshAssignment rowVariables logicalWidth publicFits) :
    (value candidate statementId config successor
      result.suffixBatch assignment).Canonical where
  memoryCanonical := checkedBatch_memoryCanonical result

/-! The verifier owns this package. Its dependent equality binds one exact
source program to the relation artifact used by the NIFS verifier. -/
structure RelationAuthority
    {rowVariables logicalWidth : Nat}
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits) where
  privateWidth : Nat
  widthExact : logicalWidth =
    ProductionFreshLowNormEncoding.logicalWidth privateWidth
  program : ProductionFreshRelationCompilerFor.SourceProgram
    privateWidth rowVariables
  fPrimeProgram : ProductionFreshFPrimeBranchRows.Program
  programRowsExact : program.rows = fPrimeProgram.rows
  artifactExact : HEq artifact program.relationArtifact

def FreshRelationWitness
    {candidate : Id} {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductPoseidon2.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (authority : RelationAuthority publicFits artifact)
    (successor : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (memory : ProductionMemoryBatchPoseidonBinding.Batch candidate)
    (assignment : FreshAssignment rowVariables logicalWidth publicFits) : Prop :=
  ∃ (source : ProductionFreshLowNormEncoding.SourceAssignment
      authority.privateWidth),
    HEq assignment
      (ProductionFreshLowNormEncoding.encodeCarrier source :
        ProductPaperAlgebraFor.Assignment rowVariables
          (ProductionFreshLowNormEncoding.logicalWidth authority.privateWidth)
          (ProductionFreshLowNormEncoding.publicFits authority.privateWidth)) ∧
    ProductionFreshRelationCompilerFor.SourceProgram.PublicMatches source
      (ProductNifsCodec.publicInputOfFor
        (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
          publicFits)
        (ccsPublic statementId successor memory)) ∧
    ProductionFreshLowNormEncoding.DirectBinary source ∧
    R1CS.Satisfies authority.program.rows
      (ProductionFreshLinearSubstitution.sourceNat source)

/-- Honest-prover witness for generated rows. This predicate records that the
carrier is the output of the deterministic shifted-ternary encoder. It is a
completeness premise only. Soundness must use `FreshRelationWitnessForRows`,
which decodes an arbitrary accepted bounded carrier. -/
def EncodedFreshRelationWitnessForRows
    {candidate : Id} {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductPoseidon2.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (authority : RelationAuthority publicFits artifact)
    (successor : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (memory : ProductionMemoryBatchPoseidonBinding.Batch candidate)
    (assignment : FreshAssignment rowVariables logicalWidth publicFits)
    (rowAssignment : Nat -> Nat) : Prop :=
  exists source : ProductionFreshLowNormEncoding.SourceAssignment
      authority.privateWidth,
    HEq assignment
      (ProductionFreshLowNormEncoding.encodeCarrier source :
        ProductPaperAlgebraFor.Assignment rowVariables
          (ProductionFreshLowNormEncoding.logicalWidth authority.privateWidth)
          (ProductionFreshLowNormEncoding.publicFits authority.privateWidth)) /\
    ProductionFreshRelationCompilerFor.SourceProgram.PublicMatches source
      (ProductNifsCodec.publicInputOfFor
        (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
          publicFits)
        (ccsPublic statementId successor memory)) /\
    ProductionFreshLowNormEncoding.DirectBinary source /\
    rowAssignment = ProductionFreshLinearSubstitution.sourceNat source /\
    R1CS.Satisfies authority.program.rows rowAssignment

namespace EncodedFreshRelationWitnessForRows

private theorem sourceNat_canonical
    {privateWidth : Nat}
    (source : ProductionFreshLowNormEncoding.SourceAssignment privateWidth) :
    forall column,
      ProductionFreshLinearSubstitution.sourceNat source column <
        R1CS.goldilocksP := by
  intro column
  by_cases within :
      column < ProductionFreshLowNormEncoding.sourceWidth privateWidth
  · simp only [ProductionFreshLinearSubstitution.sourceNat, dif_pos within]
    simpa only [R1CS.goldilocksP, goldilocksModulus] using
      (source ⟨column, within⟩).isLt
  · simp [ProductionFreshLinearSubstitution.sourceNat, within,
      R1CS.goldilocksP]

private theorem targetPublic_zero
    {candidate : Id} {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (statementId : ProductPoseidon2.StatementId)
    (successor : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (memory : ProductionMemoryBatchPoseidonBinding.Batch candidate) :
    ProductNifsCodec.publicInputOfFor
        (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
          publicFits)
        (ccsPublic statementId successor memory)
        ⟨0, by
          change 0 < 540
          decide⟩ = 1 := by
  simp [ProductNifsCodec.publicInputOfFor, ccsPublic,
    ProductionMemoryBoundCcsPublic.word,
    ProductionMemoryBoundCcsPublic.encode, ProductNifsCodec.fieldOfBit]

/-- Forget only the explicit numeric-assignment identity. All remaining facts
use the same source witness. -/
theorem toFreshRelationWitness
    {candidate : Id} {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductPoseidon2.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {authority : RelationAuthority publicFits artifact}
    {successor : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {memory : ProductionMemoryBatchPoseidonBinding.Batch candidate}
    {assignment : FreshAssignment rowVariables logicalWidth publicFits}
    {rowAssignment : Nat -> Nat}
    (witness : EncodedFreshRelationWitnessForRows statementId config artifact authority
      successor memory assignment rowAssignment) :
    FreshRelationWitness statementId config artifact authority successor memory
      assignment := by
  rcases witness with
    ⟨source, assignmentExact, publicExact, directBinary, rowsExact,
      sourceRows⟩
  refine ⟨source, assignmentExact, publicExact, directBinary, ?_⟩
  rw [← rowsExact]
  exact sourceRows

/-- The semantic rows satisfy the verifier-owned relation on the exact
numeric view of the committed witness. -/
theorem authorityRows
    {candidate : Id} {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductPoseidon2.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {authority : RelationAuthority publicFits artifact}
    {successor : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {memory : ProductionMemoryBatchPoseidonBinding.Batch candidate}
    {assignment : FreshAssignment rowVariables logicalWidth publicFits}
    {rowAssignment : Nat -> Nat}
    (witness : EncodedFreshRelationWitnessForRows statementId config artifact authority
      successor memory assignment rowAssignment) :
    R1CS.Satisfies authority.program.rows rowAssignment := by
  rcases witness with ⟨_, _, _, _, _, rows⟩
  exact rows

/-- A satisfying fresh claim selects the base arm exactly at iteration zero
and the recursive arm at every nonzero iteration. The relation rows, branch
selector, and semantic rows all use the source view of the same committed
witness. -/
theorem selectedBranch
    {candidate : Id} {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductPoseidon2.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {authority : RelationAuthority publicFits artifact}
    {successor : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {memory : ProductionMemoryBatchPoseidonBinding.Batch candidate}
    {assignment : FreshAssignment rowVariables logicalWidth publicFits}
    {rowAssignment : Nat -> Nat}
    (witness : EncodedFreshRelationWitnessForRows statementId config artifact authority
      successor memory assignment rowAssignment) :
    (rowAssignment
          authority.fPrimeProgram.layout.iterationZero.iterationColumn = 0 /\
        R1CS.Satisfies authority.fPrimeProgram.baseRows rowAssignment) \/
      (0 < rowAssignment
          authority.fPrimeProgram.layout.iterationZero.iterationColumn /\
        R1CS.Satisfies authority.fPrimeProgram.recursiveRows rowAssignment) := by
  rcases witness with
    ⟨source, _assignmentExact, publicExact, _directBinary, rowAssignmentExact,
      sourceRows⟩
  have canonical : forall column, rowAssignment column < R1CS.goldilocksP := by
    rw [rowAssignmentExact]
    exact sourceNat_canonical source
  have sourceOne :=
    ProductionFreshRelationCompilerFor.SourceProgram.sourceOne_of_publicMatches
      source
      (ProductNifsCodec.publicInputOfFor
        (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
          publicFits)
        (ccsPublic statementId successor memory))
      publicExact (targetPublic_zero statementId successor memory)
  have one : rowAssignment 0 = 1 := by
    rw [rowAssignmentExact]
    rw [show 0 =
        (ProductionFreshLowNormEncoding.publicSourceColumn
          (privateWidth := authority.privateWidth) ⟨0, by decide⟩).val by rfl,
      ProductionFreshLinearSubstitution.sourceNat_sourceColumn]
    simpa using congrArg Fin.val sourceOne
  have branchRows :
      R1CS.Satisfies authority.fPrimeProgram.rows rowAssignment := by
    rw [← authority.programRowsExact]
    exact sourceRows
  exact authority.fPrimeProgram.sound canonical one branchRows

end EncodedFreshRelationWitnessForRows

/-- Soundness witness for one arbitrary accepted bounded carrier. The row
assignment is the exact source assignment decoded from that carrier. This
predicate does not require the carrier to be in the honest encoder image.

The public projection, norm bound, and source-row satisfaction are separate
facts because they come from different verifier checks. None of them assumes
CCS satisfaction or an F-prime branch conclusion. -/
def FreshRelationWitnessForRows
    {candidate : Id} {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductPoseidon2.StatementId)
    (_config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (_artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (authority : RelationAuthority publicFits _artifact)
    (successor : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (memory : ProductionMemoryBatchPoseidonBinding.Batch candidate)
    (assignment : FreshAssignment rowVariables logicalWidth publicFits)
    (rowAssignment : Nat -> Nat) : Prop :=
  exists assignmentAtAuthorityWidth :
      ProductPaperAlgebraFor.Assignment rowVariables
        (ProductionFreshLowNormEncoding.logicalWidth authority.privateWidth)
        (ProductionFreshLowNormEncoding.publicFits authority.privateWidth),
    HEq assignment assignmentAtAuthorityWidth /\
    Phi81Relation.projectPublicInput assignmentAtAuthorityWidth =
      ProductNifsCodec.publicInputOfFor
        (ProductPaperAlgebraFor.fullShapeContract rowVariables
          (ProductionFreshLowNormEncoding.logicalWidth authority.privateWidth)
          (ProductionFreshLowNormEncoding.publicFits authority.privateWidth))
        (ccsPublic statementId successor memory) /\
    Phi81Relation.assignmentNormBounded productionGlobalParams.b
      assignmentAtAuthorityWidth /\
    rowAssignment = authority.program.decodedSourceAssignment
      assignmentAtAuthorityWidth /\
    R1CS.Satisfies authority.program.rows rowAssignment

namespace RelationAuthority

/-- The source assignment decoded from an accepted carrier remains tied to
that exact carrier by `HEq`. This witness does not require the carrier to be
in the image of the honest deterministic encoder. -/
def ExactDecodedBranch
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    (authority : RelationAuthority publicFits artifact)
    (assignment : FreshAssignment rowVariables logicalWidth publicFits) : Prop :=
  exists assignmentAtAuthorityWidth :
      ProductPaperAlgebraFor.Assignment rowVariables
        (ProductionFreshLowNormEncoding.logicalWidth authority.privateWidth)
        (ProductionFreshLowNormEncoding.publicFits authority.privateWidth),
    HEq assignment assignmentAtAuthorityWidth /\
      let decoded := authority.program.decodedSourceAssignment
        assignmentAtAuthorityWidth
      (decoded
            authority.fPrimeProgram.layout.iterationZero.iterationColumn = 0 /\
          R1CS.Satisfies authority.fPrimeProgram.baseRows decoded) \/
        (0 < decoded
            authority.fPrimeProgram.layout.iterationZero.iterationColumn /\
          R1CS.Satisfies authority.fPrimeProgram.recursiveRows decoded)

/-- Core reverse compiler theorem for an arbitrary accepted public input and
carrier assignment. The only public-input premise used by the compiler is
the verifier-required constant-one coordinate. -/
theorem selectedBranchOfCcsPublic
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (authority : RelationAuthority publicFits artifact)
    (assignment : FreshAssignment rowVariables logicalWidth publicFits)
    (publicInput : ProductPaperAlgebraFor.PublicInput rowVariables logicalWidth
      publicFits)
    (publicExact :
      Phi81Relation.projectPublicInput assignment = publicInput)
    (publicZero :
      publicInput
        ⟨0, by
          change 0 < 540
          decide⟩ = 1)
    (relation :
      (ProductPaperAlgebraFor.semantics config).ccsSatisfied
        (ProductPaperAlgebraFor.matrixSource artifact.system) assignment) :
    ExactDecodedBranch authority assignment := by
  have projectedZero :
      Phi81Relation.projectPublicInput assignment
          ⟨0, by
            change 0 < 540
            decide⟩ = 1 := by
    rw [publicExact]
    exact publicZero
  rcases authority with
    ⟨privateWidth, widthExact, program, fPrimeProgram, programRowsExact,
      artifactExact⟩
  subst logicalWidth
  have artifactEq : artifact = program.relationArtifact :=
    eq_of_heq artifactExact
  subst artifact
  refine ⟨assignment, HEq.rfl, ?_⟩
  let decoded := program.decodedSourceAssignment assignment
  have logicalOne :
      program.logicalAssignment assignment
          (ProductionFreshRelationCompilerFor.NumericBridge.finiteColumnIndex
            program.logicalWidth_positive
            (ProductionFreshRelationCompilerFor.NumericBridge.sourceColumn 0)) =
        1 := by
    simpa [ProductionFreshRelationCompilerFor.SourceProgram.logicalAssignment,
      ProductionFreshRelationCompilerFor.NumericBridge.finiteColumnIndex,
      ProductionFreshRelationCompilerFor.NumericBridge.sourceColumn,
      Phi81Relation.projectPublicInput, Phi81Relation.Shape.publicColumn,
      Phi81CarrierLayout.embedLogical] using projectedZero
  have sourceRows : R1CS.Satisfies program.rows decoded := by
    apply (program.ccsSatisfied_iff_decodedSourceRows config assignment
      logicalOne).mp
    exact relation
  have branchRows : R1CS.Satisfies fPrimeProgram.rows decoded := by
    rw [← programRowsExact]
    exact sourceRows
  have canonical : forall column, decoded column < R1CS.goldilocksP := by
    intro column
    exact program.decodedSourceAssignment_canonical assignment column
  have decodedZero : decoded 0 = 1 := by
    let zero : Fin ProductionFreshLowNormEncoding.directWidth :=
      ⟨0, by decide⟩
    have direct := program.decodedSourceAssignment_direct assignment zero
    have sameLogical :
        (⟨zero.val, Nat.lt_of_lt_of_le zero.isLt
          (Nat.le_trans
            (ProductionFreshLowNormEncoding.directWidth_le_payloadWidth
              privateWidth)
            (ProductionFreshLowNormEncoding.payloadWidth_le_logicalWidth
              privateWidth))⟩ :
          Fin (ProductionFreshLowNormEncoding.logicalWidth privateWidth)) =
          ProductionFreshRelationCompilerFor.NumericBridge.finiteColumnIndex
          program.logicalWidth_positive
          (ProductionFreshRelationCompilerFor.NumericBridge.sourceColumn 0) := by
      apply Fin.ext
      rfl
    rw [sameLogical] at direct
    have logicalOneNat :
        (program.logicalAssignment assignment
          (ProductionFreshRelationCompilerFor.NumericBridge.finiteColumnIndex
            program.logicalWidth_positive
            (ProductionFreshRelationCompilerFor.NumericBridge.sourceColumn 0))).val =
          1 := by
      simpa using congrArg Fin.val logicalOne
    exact direct.trans logicalOneNat
  exact fPrimeProgram.sound canonical decodedZero branchRows

/-- Reverse compiler theorem for one typed V2 fresh-claim public image. Unlike
`FreshRelationWitness`, this theorem does not assume that the committed
private words came from the honest deterministic encoder. -/
theorem selectedBranchOfCcsAssignment
    {candidate : Id} {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductPoseidon2.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (authority : RelationAuthority publicFits artifact)
    (successor : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (memory : ProductionMemoryBatchPoseidonBinding.Batch candidate)
    (assignment : FreshAssignment rowVariables logicalWidth publicFits)
    (publicExact :
      Phi81Relation.projectPublicInput assignment =
        ProductNifsCodec.publicInputOfFor
          (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
            publicFits)
          (ccsPublic statementId successor memory))
    (relation :
      (ProductPaperAlgebraFor.semantics config).ccsSatisfied
        (ProductPaperAlgebraFor.matrixSource artifact.system) assignment) :
    ExactDecodedBranch authority assignment := by
  apply selectedBranchOfCcsPublic config artifact authority assignment
    (ProductNifsCodec.publicInputOfFor
      (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
        publicFits)
      (ccsPublic statementId successor memory)) publicExact
  · simp [ProductNifsCodec.publicInputOfFor, ccsPublic,
      ProductionMemoryBoundCcsPublic.word,
      ProductionMemoryBoundCcsPublic.encode, ProductNifsCodec.fieldOfBit]
  · exact relation

end RelationAuthority

namespace FreshRelationWitnessForRows

private theorem targetPublic_zero
    {candidate : Id} {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (statementId : ProductPoseidon2.StatementId)
    (successor : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (memory : ProductionMemoryBatchPoseidonBinding.Batch candidate) :
    ProductNifsCodec.publicInputOfFor
        (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
          publicFits)
        (ccsPublic statementId successor memory)
        ⟨0, by
          change 0 < 540
          decide⟩ = 1 := by
  simp [ProductNifsCodec.publicInputOfFor, ccsPublic,
    ProductionMemoryBoundCcsPublic.word,
    ProductionMemoryBoundCcsPublic.encode, ProductNifsCodec.fieldOfBit]

/-- The accepted carrier has the exact public image in the claim. -/
theorem publicOutput
    {candidate : Id} {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductPoseidon2.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {authority : RelationAuthority publicFits artifact}
    {successor : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {memory : ProductionMemoryBatchPoseidonBinding.Batch candidate}
    {assignment : FreshAssignment rowVariables logicalWidth publicFits}
    {rowAssignment : Nat -> Nat}
    (witness : FreshRelationWitnessForRows statementId config artifact authority
      successor memory assignment rowAssignment) :
    Phi81Relation.projectPublicInput assignment =
      ProductNifsCodec.publicInputOfFor
        (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
          publicFits)
        (ccsPublic statementId successor memory) := by
  rcases authority with
    ⟨privateWidth, widthExact, program, fPrimeProgram, programRowsExact,
      artifactExact⟩
  subst logicalWidth
  rcases witness with
    ⟨assignmentAtAuthorityWidth, assignmentExact, publicExact, _norm,
      _rowExact, _rows⟩
  have assignmentEq : assignment = assignmentAtAuthorityWidth :=
    eq_of_heq assignmentExact
  subst assignment
  exact publicExact

/-- The accepted carrier meets the fresh-claim norm bound. -/
theorem norm
    {candidate : Id} {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductPoseidon2.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {authority : RelationAuthority publicFits artifact}
    {successor : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {memory : ProductionMemoryBatchPoseidonBinding.Batch candidate}
    {assignment : FreshAssignment rowVariables logicalWidth publicFits}
    {rowAssignment : Nat -> Nat}
    (witness : FreshRelationWitnessForRows statementId config artifact authority
      successor memory assignment rowAssignment) :
    Phi81Relation.assignmentNormBounded productionGlobalParams.b assignment := by
  rcases authority with
    ⟨privateWidth, widthExact, program, fPrimeProgram, programRowsExact,
      artifactExact⟩
  subst logicalWidth
  rcases witness with
    ⟨assignmentAtAuthorityWidth, assignmentExact, _publicExact, normExact,
      _rowExact, _rows⟩
  have assignmentEq : assignment = assignmentAtAuthorityWidth :=
    eq_of_heq assignmentExact
  subst assignment
  exact normExact

/-- The semantic rows use the exact source view decoded from the accepted
carrier. -/
theorem authorityRows
    {candidate : Id} {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductPoseidon2.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {authority : RelationAuthority publicFits artifact}
    {successor : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {memory : ProductionMemoryBatchPoseidonBinding.Batch candidate}
    {assignment : FreshAssignment rowVariables logicalWidth publicFits}
    {rowAssignment : Nat -> Nat}
    (witness : FreshRelationWitnessForRows statementId config artifact authority
      successor memory assignment rowAssignment) :
    R1CS.Satisfies authority.program.rows rowAssignment := by
  rcases witness with ⟨_, _, _, _, _, rows⟩
  exact rows

/-- The source rows imply CCS satisfaction for the exact accepted carrier.
The proof derives the constant-one coordinate from the verifier public image. -/
theorem relation
    {candidate : Id} {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductPoseidon2.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {authority : RelationAuthority publicFits artifact}
    {successor : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {memory : ProductionMemoryBatchPoseidonBinding.Batch candidate}
    {assignment : FreshAssignment rowVariables logicalWidth publicFits}
    {rowAssignment : Nat -> Nat}
    (witness : FreshRelationWitnessForRows statementId config artifact authority
      successor memory assignment rowAssignment) :
    (ProductPaperAlgebraFor.semantics config).ccsSatisfied
      (ProductPaperAlgebraFor.matrixSource artifact.system) assignment := by
  rcases authority with
    ⟨privateWidth, widthExact, program, fPrimeProgram, programRowsExact,
      artifactExact⟩
  subst logicalWidth
  rcases witness with
    ⟨assignmentAtAuthorityWidth, assignmentExact, publicExact, _normExact,
      rowExact, rows⟩
  have assignmentEq : assignment = assignmentAtAuthorityWidth :=
    eq_of_heq assignmentExact
  subst assignment
  have artifactEq : artifact = program.relationArtifact :=
    eq_of_heq artifactExact
  subst artifact
  have projectedZero :
      Phi81Relation.projectPublicInput assignmentAtAuthorityWidth
          ⟨0, by
            change 0 < 540
            decide⟩ = 1 := by
    rw [publicExact]
    exact targetPublic_zero statementId successor memory
  have logicalOne :
      program.logicalAssignment assignmentAtAuthorityWidth
          (ProductionFreshRelationCompilerFor.NumericBridge.finiteColumnIndex
            program.logicalWidth_positive
            (ProductionFreshRelationCompilerFor.NumericBridge.sourceColumn 0)) =
        1 := by
    simpa [ProductionFreshRelationCompilerFor.SourceProgram.logicalAssignment,
      ProductionFreshRelationCompilerFor.NumericBridge.finiteColumnIndex,
      ProductionFreshRelationCompilerFor.NumericBridge.sourceColumn,
      Phi81Relation.projectPublicInput, Phi81Relation.Shape.publicColumn,
      Phi81CarrierLayout.embedLogical] using projectedZero
  apply (program.ccsSatisfied_iff_decodedSourceRows config
    assignmentAtAuthorityWidth logicalOne).mpr
  rw [← rowExact]
  exact rows

/-- A satisfying accepted carrier selects the base arm at iteration zero and
the recursive arm at every nonzero iteration. No honest encoding premise is
used. -/
theorem selectedBranch
    {candidate : Id} {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductPoseidon2.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {authority : RelationAuthority publicFits artifact}
    {successor : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {memory : ProductionMemoryBatchPoseidonBinding.Batch candidate}
    {assignment : FreshAssignment rowVariables logicalWidth publicFits}
    {rowAssignment : Nat -> Nat}
    (witness : FreshRelationWitnessForRows statementId config artifact authority
      successor memory assignment rowAssignment) :
    (rowAssignment
          authority.fPrimeProgram.layout.iterationZero.iterationColumn = 0 /\
        R1CS.Satisfies authority.fPrimeProgram.baseRows rowAssignment) \/
      (0 < rowAssignment
          authority.fPrimeProgram.layout.iterationZero.iterationColumn /\
        R1CS.Satisfies authority.fPrimeProgram.recursiveRows rowAssignment) := by
  rcases witness with
    ⟨assignmentAtAuthorityWidth, _assignmentExact, _publicExact, _normExact,
      rowExact, rows⟩
  have canonical : forall column, rowAssignment column < R1CS.goldilocksP := by
    rw [rowExact]
    exact authority.program.decodedSourceAssignment_canonical
      assignmentAtAuthorityWidth
  have one : rowAssignment 0 = 1 := by
    rw [rowExact]
    let zero : Fin ProductionFreshLowNormEncoding.directWidth :=
      ⟨0, by decide⟩
    have direct := authority.program.decodedSourceAssignment_direct
      assignmentAtAuthorityWidth zero
    have publicZero :
        Phi81Relation.projectPublicInput assignmentAtAuthorityWidth
            ⟨0, by
              change 0 < 540
              decide⟩ = 1 := by
      rw [_publicExact]
      exact targetPublic_zero statementId successor memory
    have logicalOne :
        authority.program.logicalAssignment assignmentAtAuthorityWidth
            (ProductionFreshRelationCompilerFor.NumericBridge.finiteColumnIndex
              authority.program.logicalWidth_positive
              (ProductionFreshRelationCompilerFor.NumericBridge.sourceColumn 0)) =
          1 := by
      simpa [ProductionFreshRelationCompilerFor.SourceProgram.logicalAssignment,
        ProductionFreshRelationCompilerFor.NumericBridge.finiteColumnIndex,
        ProductionFreshRelationCompilerFor.NumericBridge.sourceColumn,
        Phi81Relation.projectPublicInput, Phi81Relation.Shape.publicColumn,
        Phi81CarrierLayout.embedLogical] using publicZero
    have sameLogical :
        (⟨zero.val, Nat.lt_of_lt_of_le zero.isLt
          (Nat.le_trans
            (ProductionFreshLowNormEncoding.directWidth_le_payloadWidth
              authority.privateWidth)
            (ProductionFreshLowNormEncoding.payloadWidth_le_logicalWidth
              authority.privateWidth))⟩ :
          Fin (ProductionFreshLowNormEncoding.logicalWidth
            authority.privateWidth)) =
          ProductionFreshRelationCompilerFor.NumericBridge.finiteColumnIndex
            authority.program.logicalWidth_positive
            (ProductionFreshRelationCompilerFor.NumericBridge.sourceColumn 0) := by
      apply Fin.ext
      rfl
    rw [sameLogical] at direct
    have logicalOneNat :
        (authority.program.logicalAssignment assignmentAtAuthorityWidth
          (ProductionFreshRelationCompilerFor.NumericBridge.finiteColumnIndex
            authority.program.logicalWidth_positive
            (ProductionFreshRelationCompilerFor.NumericBridge.sourceColumn 0))).val =
          1 := by
      simpa using congrArg Fin.val logicalOne
    exact direct.trans logicalOneNat
  have branchRows :
      R1CS.Satisfies authority.fPrimeProgram.rows rowAssignment := by
    rw [← authority.programRowsExact]
    exact rows
  exact authority.fPrimeProgram.sound canonical one branchRows

end FreshRelationWitnessForRows

namespace FreshRelationWitness

private theorem targetPublic_zero
    {candidate : Id} {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (statementId : ProductPoseidon2.StatementId)
    (successor : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (memory : ProductionMemoryBatchPoseidonBinding.Batch candidate) :
    ProductNifsCodec.publicInputOfFor
        (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
          publicFits)
        (ccsPublic statementId successor memory)
        ⟨0, by
          change 0 < 540
          decide⟩ = 1 := by
  simp [ProductNifsCodec.publicInputOfFor, ccsPublic,
    ProductionMemoryBoundCcsPublic.word,
    ProductionMemoryBoundCcsPublic.encode, ProductNifsCodec.fieldOfBit]

theorem publicOutput
    {candidate : Id} {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductPoseidon2.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {authority : RelationAuthority publicFits artifact}
    {successor : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {memory : ProductionMemoryBatchPoseidonBinding.Batch candidate}
    {assignment : FreshAssignment rowVariables logicalWidth publicFits}
    (witness : FreshRelationWitness statementId config artifact authority successor
      memory assignment) :
    Phi81Relation.projectPublicInput assignment =
      ProductNifsCodec.publicInputOfFor
        (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
          publicFits)
        (ccsPublic statementId successor memory) := by
  rcases authority with
    ⟨privateWidth, widthExact, program, _fPrimeProgram, _programRowsExact,
      artifactExact⟩
  rcases witness with
    ⟨source, assignmentExact, publicExact, _directBinary, _sourceRows⟩
  subst logicalWidth
  have assignmentEq : assignment =
      (ProductionFreshLowNormEncoding.encodeCarrier source :
        ProductPaperAlgebraFor.Assignment rowVariables
          (ProductionFreshLowNormEncoding.logicalWidth privateWidth)
          (ProductionFreshLowNormEncoding.publicFits privateWidth)) :=
    eq_of_heq assignmentExact
  subst assignment
  rw [ProductionFreshRelationCompilerFor.SourceProgram.projectPublicInput_encodeCarrier_for
    source]
  funext column
  exact publicExact column

theorem norm
    {candidate : Id} {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductPoseidon2.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {authority : RelationAuthority publicFits artifact}
    {successor : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {memory : ProductionMemoryBatchPoseidonBinding.Batch candidate}
    {assignment : FreshAssignment rowVariables logicalWidth publicFits}
    (witness : FreshRelationWitness statementId config artifact authority successor
      memory assignment) :
    Phi81Relation.assignmentNormBounded productionGlobalParams.b assignment := by
  rcases authority with
    ⟨privateWidth, widthExact, program, _fPrimeProgram, _programRowsExact,
      artifactExact⟩
  rcases witness with
    ⟨source, assignmentExact, _publicExact, directBinary, _sourceRows⟩
  subst logicalWidth
  have assignmentEq : assignment =
      (ProductionFreshLowNormEncoding.encodeCarrier source :
        ProductPaperAlgebraFor.Assignment rowVariables
          (ProductionFreshLowNormEncoding.logicalWidth privateWidth)
          (ProductionFreshLowNormEncoding.publicFits privateWidth)) :=
    eq_of_heq assignmentExact
  subst assignment
  simpa [productionGlobalParams] using
    ProductionFreshLowNormEncoding.encodeCarrier_norm source directBinary

theorem relation
    {candidate : Id} {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductPoseidon2.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {authority : RelationAuthority publicFits artifact}
    {successor : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {memory : ProductionMemoryBatchPoseidonBinding.Batch candidate}
    {assignment : FreshAssignment rowVariables logicalWidth publicFits}
    (witness : FreshRelationWitness statementId config artifact authority successor
      memory assignment) :
    (ProductPaperAlgebraFor.semantics config).ccsSatisfied
      (ProductPaperAlgebraFor.matrixSource artifact.system) assignment := by
  rcases authority with
    ⟨privateWidth, widthExact, program, _fPrimeProgram, _programRowsExact,
      artifactExact⟩
  rcases witness with
    ⟨source, assignmentExact, publicExact, _directBinary, sourceRows⟩
  subst logicalWidth
  have artifactEq : artifact = program.relationArtifact := eq_of_heq artifactExact
  have assignmentEq : assignment =
      (ProductionFreshLowNormEncoding.encodeCarrier source :
        ProductPaperAlgebraFor.Assignment rowVariables
          (ProductionFreshLowNormEncoding.logicalWidth privateWidth)
          (ProductionFreshLowNormEncoding.publicFits privateWidth)) :=
    eq_of_heq assignmentExact
  subst artifact
  subst assignment
  have sourceOne :=
    ProductionFreshRelationCompilerFor.SourceProgram.sourceOne_of_publicMatches
      source
      (ProductNifsCodec.publicInputOfFor
        (ProductPaperAlgebraFor.fullShapeContract rowVariables
          (ProductionFreshLowNormEncoding.logicalWidth privateWidth)
          (ProductionFreshLowNormEncoding.publicFits privateWidth))
        (ccsPublic statementId successor memory))
      publicExact (targetPublic_zero statementId successor memory)
  exact (program.encoded_ccsSatisfied_iff_sourceRows config source sourceOne).mpr
    sourceRows

end FreshRelationWitness

noncomputable def freshStatement
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductPoseidon2.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (successor : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (memory : ProductionMemoryBatchPoseidonBinding.Batch candidate)
    (assignment : FreshAssignment rowVariables logicalWidth publicFits) :
    CCS.Instance (ProductPaperAlgebraFor.Structure rowVariables logicalWidth)
      (ProductPaperAlgebraFor.PublicInput rowVariables logicalWidth publicFits)
      ProductPaperAlgebraFor.Commitment where
  constraintSystem := ProductPaperAlgebraFor.matrixSource artifact.system
  commitment := ProductNifsCodec.codecBundle
    (value candidate statementId config successor memory
      assignment).commitmentBundle
  publicInput := ProductNifsCodec.publicInputOfFor
    (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
      publicFits)
    (value candidate statementId config successor memory
      assignment).ccsPublic
  stage := .fresh

theorem freshStatement_holds
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductPoseidon2.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (authority : RelationAuthority publicFits artifact)
    (successor : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (memory : ProductionMemoryBatchPoseidonBinding.Batch candidate)
    (assignment : FreshAssignment rowVariables logicalWidth publicFits)
    (witness : FreshRelationWitness statementId config artifact authority successor
      memory assignment) :
    CCS.Holds (ProductPaperAlgebraFor.semantics config) productionGlobalParams
      (freshStatement candidate statementId config artifact
        successor memory assignment) assignment := by
  refine ⟨⟨?_, ?_, ?_⟩, witness.relation⟩
  · exact (value_bundle_opens candidate statementId config
      successor memory assignment).symm
  · simpa [freshStatement, value] using witness.publicOutput
  · simpa [freshStatement, NormStage.bound] using witness.norm

/-- Package the independently checked public projection, norm bound, and
source rows for an arbitrary accepted carrier into one CCS membership fact.
Unlike `freshStatement_holds`, this theorem does not assume that the carrier
was produced by the honest encoder. -/
theorem freshStatement_holds_from_rows
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductPoseidon2.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (authority : RelationAuthority publicFits artifact)
    (successor : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (memory : ProductionMemoryBatchPoseidonBinding.Batch candidate)
    (assignment : FreshAssignment rowVariables logicalWidth publicFits)
    (rowAssignment : Nat -> Nat)
    (witness : FreshRelationWitnessForRows statementId config artifact authority
      successor memory assignment rowAssignment) :
    CCS.Holds (ProductPaperAlgebraFor.semantics config) productionGlobalParams
      (freshStatement candidate statementId config artifact
        successor memory assignment) assignment := by
  refine ⟨⟨?_, ?_, ?_⟩, witness.relation⟩
  · exact (value_bundle_opens candidate statementId config
      successor memory assignment).symm
  · simpa [freshStatement, value] using witness.publicOutput
  · simpa [freshStatement, NormStage.bound] using witness.norm

namespace FreshRelationWitnessForRows

/-- Exact reverse bridge from arbitrary fresh CCS membership. It returns the
source-row assignment decoded from the accepted carrier. No honest encoder or
witness-generator premise occurs in this theorem. -/
theorem exists_of_ccsHolds
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductPoseidon2.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (authority : RelationAuthority publicFits artifact)
    (successor : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (memory : ProductionMemoryBatchPoseidonBinding.Batch candidate)
    (assignment : FreshAssignment rowVariables logicalWidth publicFits)
    (holds : CCS.Holds (ProductPaperAlgebraFor.semantics config)
      productionGlobalParams
      (freshStatement candidate statementId config artifact
        successor memory assignment) assignment) :
    exists rowAssignment : Nat -> Nat,
      FreshRelationWitnessForRows statementId config artifact authority
        successor memory assignment rowAssignment := by
  rcases authority with
    ⟨privateWidth, widthExact, program, _fPrimeProgram, _programRowsExact,
      artifactExact⟩
  subst logicalWidth
  have artifactEq : artifact = program.relationArtifact :=
    eq_of_heq artifactExact
  subst artifact
  have publicExact :
      Phi81Relation.projectPublicInput assignment =
        ProductNifsCodec.publicInputOfFor
          (ProductPaperAlgebraFor.fullShapeContract rowVariables
            (ProductionFreshLowNormEncoding.logicalWidth privateWidth)
            (ProductionFreshLowNormEncoding.publicFits privateWidth))
          (ccsPublic statementId successor memory) := by
    simpa [freshStatement] using holds.1.2.1
  have normExact :
      Phi81Relation.assignmentNormBounded productionGlobalParams.b
        assignment := by
    simpa [freshStatement, NormStage.bound] using holds.1.2.2
  have projectedZero :
      Phi81Relation.projectPublicInput assignment
          ⟨0, by
            change 0 < 540
            decide⟩ = 1 := by
    rw [publicExact]
    simp [ProductNifsCodec.publicInputOfFor, ccsPublic,
      ProductionMemoryBoundCcsPublic.word,
      ProductionMemoryBoundCcsPublic.encode, ProductNifsCodec.fieldOfBit]
  have logicalOne :
      program.logicalAssignment assignment
          (ProductionFreshRelationCompilerFor.NumericBridge.finiteColumnIndex
            program.logicalWidth_positive
            (ProductionFreshRelationCompilerFor.NumericBridge.sourceColumn 0)) =
        1 := by
    simpa [ProductionFreshRelationCompilerFor.SourceProgram.logicalAssignment,
      ProductionFreshRelationCompilerFor.NumericBridge.finiteColumnIndex,
      ProductionFreshRelationCompilerFor.NumericBridge.sourceColumn,
      Phi81Relation.projectPublicInput, Phi81Relation.Shape.publicColumn,
      Phi81CarrierLayout.embedLogical] using projectedZero
  have sourceRows :
      R1CS.Satisfies program.rows
        (program.decodedSourceAssignment assignment) := by
    apply (program.ccsSatisfied_iff_decodedSourceRows config assignment
      logicalOne).mp
    exact holds.2
  exact ⟨program.decodedSourceAssignment assignment,
    assignment, HEq.rfl, publicExact, normExact, rfl, sourceRows⟩

end FreshRelationWitnessForRows

/-- Arbitrary membership in the fresh CCS statement is equivalent to one
exact decoded source-row witness. The reverse direction does not use the
honest carrier encoder. This is the soundness-facing compiler boundary. -/
theorem freshStatement_holds_iff_exists_rows
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductPoseidon2.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (authority : RelationAuthority publicFits artifact)
    (successor : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (memory : ProductionMemoryBatchPoseidonBinding.Batch candidate)
    (assignment : FreshAssignment rowVariables logicalWidth publicFits) :
    CCS.Holds (ProductPaperAlgebraFor.semantics config) productionGlobalParams
        (freshStatement candidate statementId config artifact
          successor memory assignment) assignment <->
      exists rowAssignment : Nat -> Nat,
        FreshRelationWitnessForRows statementId config artifact authority
          successor memory assignment rowAssignment := by
  constructor
  · exact FreshRelationWitnessForRows.exists_of_ccsHolds candidate statementId
      config artifact authority successor memory assignment
  · rintro ⟨rowAssignment, witness⟩
    exact freshStatement_holds_from_rows candidate statementId config artifact
      authority successor memory assignment rowAssignment witness

theorem producedFresh_commitment
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductPoseidon2.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (successor : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (memory : ProductionMemoryBatchPoseidonBinding.Batch candidate)
    (assignment : FreshAssignment rowVariables logicalWidth publicFits) :
    (ProductionFieldNativeFullClaim.freshOfValue
      (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
        publicFits).toShape
      (value candidate statementId config successor memory
        assignment)).commitments
          ⟨0, by simp [ProductNifsCodec.shapeFor]⟩ =
      ProductCommitmentAlgebra.commit config assignment :=
  value_bundle_opens candidate statementId config successor memory
    assignment

theorem producedFresh_publicInput
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductPoseidon2.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (authority : RelationAuthority publicFits artifact)
    (successor : ProductionSuccessorStateBinding.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (memory : ProductionMemoryBatchPoseidonBinding.Batch candidate)
    (assignment : FreshAssignment rowVariables logicalWidth publicFits)
    (witness : FreshRelationWitness statementId config artifact authority successor
      memory assignment) :
    (ProductionFieldNativeFullClaim.freshOfValue
      (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
        publicFits).toShape
      (value candidate statementId config successor memory
        assignment)).publicInputs
          ⟨0, by simp [ProductNifsCodec.shapeFor]⟩ =
      Phi81Relation.projectPublicInput assignment :=
  witness.publicOutput.symm

theorem current_application_accesses_exact
    {Program : Type} {candidate : Id}
    {machine : Machine Program} {program : Program}
    {before after : AppStateVector}
    {layout : ProductionMemoryCheckedBatchRows.Layout candidate}
    {sourceAssignment : Nat -> Nat}
    {headers : FPrime.ChainHeaders Digest.Value}
    {result : ProductionMemoryCheckedBatchRows.Result layout sourceAssignment
      headers}
    {batch : Batch candidate machine program before after}
    (matched : ProductionApplicationBatchBridge.Matches result batch) :
    ApplicationBatch.accesses batch.rows =
      ProductionApplicationBatchBridge.memoryAccesses result :=
  matched.accesses_exact

end Nightstream.Implementation.Nebula.ProductionFreshClaimProducerFor
