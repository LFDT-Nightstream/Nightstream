import Nightstream.Implementation.Nebula.Production.FPrime.Recursive.RecursiveCoreGeometryFor
import Nightstream.Implementation.Nebula.NIFS.PiRLC.ChallengeBridge
import Nightstream.Implementation.Nebula.Production.NIFS.PiRLC.PostPiCcsBridgeFor
import Nightstream.Implementation.Nebula.Production.NIFS.Core.NifsOutputRowsFor
import Nightstream.Implementation.Nebula.Production.FPrime.Recursive.RelationRowsSoundFor
import Nightstream.Implementation.Nebula.Production.Carrier.FieldNativeCompactChainRowsFor

/-!
Contract: exact ordered row manifest for the common paper-NIFS fold core.

The base branch does not use this core. Both the recursive branch and the
terminal branch use it to authenticate the prior state, verify one complete
fresh claim, compute the complete NIFS output, and check the delayed memory
batch. Branch-specific continuation, successor, close, opening, and public
result rows are outside this list.

This split is security-relevant: a terminal call must consume the trailing
claim without also producing a recursive successor.

Assurance tier: exponent-indexed row implementation.

Does not own application rows, branch selectors, terminal openings, public
result rows, generated-artifact containment, Rust, or cryptography.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 5000000

namespace Nightstream.Implementation.Nebula.ProductionPaperFoldCoreManifestFor

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Protocol.Nebula.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

private theorem flatten_ofFn_length
    {alpha : Type} {count width : Nat} (blocks : Fin count -> List alpha)
    (each : forall index, (blocks index).length = width) :
    (List.ofFn blocks).flatten.length = count * width := by
  rw [List.length_flatten]
  have constant : forall value, value ∈ (List.ofFn blocks).map List.length ->
      value = width := by
    intro value member
    rcases List.mem_map.mp member with ⟨block, blockMember, rfl⟩
    rcases List.mem_ofFn.mp blockMember with ⟨index, rfl⟩
    exact each index
  rw [List.sum_eq_card_nsmul _ width constant]
  simp

/-- The complete source-major, coefficient-major, attempt-minor PiRLC
transcript row family. -/
def piRlcTranscriptRows (input : ProductPiRlcTranscriptRows.Input) : List Row :=
  (List.ofFn fun source : Fin ProductPiRlcTranscriptRows.scalarCount =>
    (List.ofFn fun coefficient :
        Fin ProductPiRlcTranscriptRows.coefficientCount =>
      (List.ofFn fun attempt : Fin ProductPiRlcTranscriptRows.attemptCount =>
        ProductPiRlcTranscriptRows.rows input
          { source := source, coefficient := coefficient,
            attempt := attempt }).flatten).flatten).flatten

theorem piRlcTranscriptRows_length_exact
    (input : ProductPiRlcTranscriptRows.Input) :
    (piRlcTranscriptRows input).length =
      ProductPiRlcTranscriptRows.aggregateRowCount := by
  have attempts : forall
      (source : Fin ProductPiRlcTranscriptRows.scalarCount)
      (coefficient : Fin ProductPiRlcTranscriptRows.coefficientCount),
      (List.ofFn fun attempt : Fin ProductPiRlcTranscriptRows.attemptCount =>
        ProductPiRlcTranscriptRows.rows input
          { source := source, coefficient := coefficient,
            attempt := attempt }).flatten.length =
        ProductPiRlcTranscriptRows.attemptCount *
          ProductPiRlcTranscriptRows.rowsPerCandidate := by
    intro source coefficient
    exact flatten_ofFn_length _ fun attempt =>
      ProductPiRlcTranscriptRows.rows_length input
        { source := source, coefficient := coefficient, attempt := attempt }
  have coefficients : forall
      (source : Fin ProductPiRlcTranscriptRows.scalarCount),
      (List.ofFn fun coefficient :
          Fin ProductPiRlcTranscriptRows.coefficientCount =>
        (List.ofFn fun attempt : Fin ProductPiRlcTranscriptRows.attemptCount =>
          ProductPiRlcTranscriptRows.rows input
            { source := source, coefficient := coefficient,
              attempt := attempt }).flatten).flatten.length =
        ProductPiRlcTranscriptRows.coefficientCount *
          (ProductPiRlcTranscriptRows.attemptCount *
            ProductPiRlcTranscriptRows.rowsPerCandidate) := by
    intro source
    exact flatten_ofFn_length _ (attempts source)
  rw [piRlcTranscriptRows]
  calc
    _ = ProductPiRlcTranscriptRows.scalarCount *
        (ProductPiRlcTranscriptRows.coefficientCount *
          (ProductPiRlcTranscriptRows.attemptCount *
            ProductPiRlcTranscriptRows.rowsPerCandidate)) :=
      flatten_ofFn_length _ coefficients
    _ = ProductPiRlcTranscriptRows.aggregateRowCount := by
      simp [ProductPiRlcTranscriptRows.aggregateRowCount,
        ProductPiRlcTranscriptRows.candidateCount, Nat.mul_assoc]

/-- The complete source-major, coefficient-major, attempt-minor candidate
classification row family. -/
def piRlcClassificationRows
    (input : ProductPiRlcTranscriptRows.Input) : List Row :=
  (List.ofFn fun source : Fin ProductPiRlcTranscriptRows.scalarCount =>
    (List.ofFn fun coefficient :
        Fin ProductPiRlcTranscriptRows.coefficientCount =>
      (List.ofFn fun attempt : Fin ProductPiRlcTranscriptRows.attemptCount =>
        ProductPiRlcCandidateClassificationRows.rows input
          { source := source, coefficient := coefficient,
            attempt := attempt }).flatten).flatten).flatten

theorem piRlcClassificationRows_length_exact
    (input : ProductPiRlcTranscriptRows.Input) :
    (piRlcClassificationRows input).length =
      ProductPiRlcCandidateClassificationRows.aggregateRowCount := by
  have attempts : forall
      (source : Fin ProductPiRlcTranscriptRows.scalarCount)
      (coefficient : Fin ProductPiRlcTranscriptRows.coefficientCount),
      (List.ofFn fun attempt : Fin ProductPiRlcTranscriptRows.attemptCount =>
        ProductPiRlcCandidateClassificationRows.rows input
          { source := source, coefficient := coefficient,
            attempt := attempt }).flatten.length =
        ProductPiRlcTranscriptRows.attemptCount * 89 := by
    intro source coefficient
    exact flatten_ofFn_length _ fun attempt =>
      ProductPiRlcCandidateClassificationRows.rows_length input
        { source := source, coefficient := coefficient, attempt := attempt }
  have coefficients : forall
      (source : Fin ProductPiRlcTranscriptRows.scalarCount),
      (List.ofFn fun coefficient :
          Fin ProductPiRlcTranscriptRows.coefficientCount =>
        (List.ofFn fun attempt : Fin ProductPiRlcTranscriptRows.attemptCount =>
          ProductPiRlcCandidateClassificationRows.rows input
            { source := source, coefficient := coefficient,
              attempt := attempt }).flatten).flatten.length =
        ProductPiRlcTranscriptRows.coefficientCount *
          (ProductPiRlcTranscriptRows.attemptCount * 89) := by
    intro source
    exact flatten_ofFn_length _ (attempts source)
  rw [piRlcClassificationRows]
  calc
    _ = ProductPiRlcTranscriptRows.scalarCount *
        (ProductPiRlcTranscriptRows.coefficientCount *
          (ProductPiRlcTranscriptRows.attemptCount * 89)) :=
      flatten_ofFn_length _ coefficients
    _ = ProductPiRlcCandidateClassificationRows.aggregateRowCount := by
      simp [ProductPiRlcCandidateClassificationRows.aggregateRowCount,
        ProductPiRlcTranscriptRows.candidateCount, Nat.mul_assoc]

/-- The complete source-major, coefficient-major first-accepted row family. -/
def piRlcFirstAcceptedRows
    (input : ProductPiRlcTranscriptRows.Input) : List Row :=
  (List.ofFn fun source :
      Fin ProductPiRlcFirstAcceptedBatchRows.sourceCount =>
    (List.ofFn fun coefficient :
        Fin ProductPiRlcFirstAcceptedBatchRows.coefficientCount =>
      ProductPiRlcFirstAcceptedBatchRows.rows input
        { source := source, coefficient := coefficient }).flatten).flatten

theorem piRlcFirstAcceptedRows_length_exact
    (input : ProductPiRlcTranscriptRows.Input) :
    (piRlcFirstAcceptedRows input).length =
      ProductPiRlcFirstAcceptedBatchRows.aggregateRowCount := by
  have coefficients : forall
      (source : Fin ProductPiRlcFirstAcceptedBatchRows.sourceCount),
      (List.ofFn fun coefficient :
          Fin ProductPiRlcFirstAcceptedBatchRows.coefficientCount =>
        ProductPiRlcFirstAcceptedBatchRows.rows input
          { source := source, coefficient := coefficient }).flatten.length =
        ProductPiRlcFirstAcceptedBatchRows.coefficientCount * 9 := by
    intro source
    exact flatten_ofFn_length _ fun coefficient =>
      ProductPiRlcFirstAcceptedBatchRows.rows_length input
        { source := source, coefficient := coefficient }
  rw [piRlcFirstAcceptedRows]
  calc
    _ = ProductPiRlcFirstAcceptedBatchRows.sourceCount *
        (ProductPiRlcFirstAcceptedBatchRows.coefficientCount * 9) :=
      flatten_ofFn_length _ coefficients
    _ = ProductPiRlcFirstAcceptedBatchRows.aggregateRowCount := by
      simp [ProductPiRlcFirstAcceptedBatchRows.aggregateRowCount,
        ProductPiRlcFirstAcceptedBatchRows.coordinateCount, Nat.mul_assoc]

/-- Static parent-column links from the PiRLC output into PiDEC. Child proof
placement is a separate witness/refinement obligation. -/
structure PiDecParentPlacement
    (algebraLayout : ProductPiRlcAlgebraRows.Layout)
    (piDecLayout : ProductPiDecRows.Layout) : Prop where
  parentBundleColumn : forall component row lane,
    piDecLayout.parentBundle.column component row lane =
      algebraLayout.outputBundle component row lane
  parentEvaluationColumn : forall matrix coefficient limb,
    piDecLayout.parentEvaluation.column matrix coefficient limb =
      algebraLayout.outputEvaluation matrix limb coefficient

structure Program (candidate : Id) (rowVariables : Nat) where
  piCcsInput : ProductPiCcsTranscriptRowsFor.Input rowVariables
  piCcsTermCount : piCcsInput.constraintPolynomial.terms.length = 74
  piCcsDegreeSum :
    KSparsePolynomial.totalDegreeSum piCcsInput.constraintPolynomial.terms = 324
  samplerBase : Nat
  piRlcAlgebraLayout : ProductPiRlcAlgebraRows.Layout
  piRlcChallengePlacement : ProductPiRlcChallengeBridge.Placement
    (ProductionProductPiRlcPostPiCcsBridgeFor.samplerInput piCcsInput
      samplerBase) piRlcAlgebraLayout
  piDecLayout : ProductPiDecRows.Layout
  piDecParentPlacement : PiDecParentPlacement piRlcAlgebraLayout piDecLayout
  fullShape : Nightstream.SuperNeo.Concrete.Phi81Relation.Shape
  fullShapeContract : ProductNifsCodec.FullShapeContractFor rowVariables
    fullShape
  nifsOutputLayout : ProductionProductNifsOutputRowsFor.Layout rowVariables
  nifsOutputValid : nifsOutputLayout.Valid fullShapeContract
    piRlcAlgebraLayout piDecLayout
  priorLayout :
    ProductionPaperPriorStateAuthorityRowsFor.Layout candidate rowVariables
  priorValid : priorLayout.Valid
  seedManifest : SeedSchedule.Manifest
  compactLayout : ProductionFieldNativeCompactChainRowsFor.Layout
  compactValid : compactLayout.Valid seedManifest priorLayout.ccs.carrier
    priorLayout.ccs.core.batch.frame.memory
  statementId : ProductPoseidon2.StatementId

def Program.piRlcInput
    {candidate : Id} {rowVariables : Nat}
    (program : Program candidate rowVariables) :
    ProductPiRlcTranscriptRows.Input :=
  ProductionProductPiRlcPostPiCcsBridgeFor.samplerInput program.piCcsInput
    program.samplerBase

def Program.rows
    {candidate : Id} {rowVariables : Nat}
    (program : Program candidate rowVariables) : List Row :=
  ProductPiCcsTranscriptRowsFor.rows program.piCcsInput ++
    piRlcTranscriptRows program.piRlcInput ++
    piRlcClassificationRows program.piRlcInput ++
    piRlcFirstAcceptedRows program.piRlcInput ++
    ProductPiRlcAlgebraRows.rows program.piRlcAlgebraLayout ++
    ProductPiDecRows.rows program.piDecLayout ++
    ProductionProductNifsOutputRowsFor.rows program.nifsOutputLayout
      (ProductPiCcsTranscriptRowsFor.pointAt program.piCcsInput) ++
    ProductionPaperPriorStateAuthorityRowsFor.rows program.priorLayout
      program.statementId ++
    ProductionMemoryCheckedBatchRows.rows
      program.priorLayout.ccs.core.batch.frame.memory ++
    ProductionFieldNativeCompactChainRowsFor.rows program.seedManifest
      program.compactLayout

private theorem mem_flatten_ofFn
    {alpha : Type} {count : Nat} (values : Fin count -> List alpha)
    (index : Fin count) {value : alpha} (member : value ∈ values index) :
    value ∈ (List.ofFn values).flatten := by
  exact List.mem_flatten.mpr
    ⟨values index, List.mem_ofFn.mpr ⟨index, rfl⟩, member⟩

theorem Program.piCcs_satisfied
    {candidate : Id} {rowVariables : Nat}
    {program : Program candidate rowVariables} {assignment : Nat -> Nat}
    (satisfied : Satisfies program.rows assignment) :
    Satisfies (ProductPiCcsTranscriptRowsFor.rows program.piCcsInput)
      assignment := by
  intro row member
  exact satisfied row (by simp [Program.rows, member])

theorem Program.piRlcTranscript_satisfied
    {candidate : Id} {rowVariables : Nat}
    {program : Program candidate rowVariables} {assignment : Nat -> Nat}
    (satisfied : Satisfies program.rows assignment) :
    ProductPiRlcTranscriptRows.RowsHold program.piRlcInput assignment := by
  intro index row member
  have aggregate : row ∈
      piRlcTranscriptRows program.piRlcInput := by
    unfold piRlcTranscriptRows
    exact mem_flatten_ofFn _ index.source
      (mem_flatten_ofFn _ index.coefficient
        (mem_flatten_ofFn _ index.attempt member))
  exact satisfied row (by simp [Program.rows, aggregate])

theorem Program.piRlcClassification_satisfied
    {candidate : Id} {rowVariables : Nat}
    {program : Program candidate rowVariables} {assignment : Nat -> Nat}
    (satisfied : Satisfies program.rows assignment) :
    ProductPiRlcCandidateClassificationRows.RowsHold program.piRlcInput
      assignment := by
  intro index row member
  have aggregate : row ∈
      piRlcClassificationRows program.piRlcInput := by
    unfold piRlcClassificationRows
    exact mem_flatten_ofFn _ index.source
      (mem_flatten_ofFn _ index.coefficient
        (mem_flatten_ofFn _ index.attempt member))
  exact satisfied row (by simp [Program.rows, aggregate])

theorem Program.piRlcFirstAccepted_satisfied
    {candidate : Id} {rowVariables : Nat}
    {program : Program candidate rowVariables} {assignment : Nat -> Nat}
    (satisfied : Satisfies program.rows assignment) :
    ProductPiRlcFirstAcceptedBatchRows.RowsHold program.piRlcInput
      assignment := by
  intro index row member
  have aggregate : row ∈
      piRlcFirstAcceptedRows program.piRlcInput := by
    unfold piRlcFirstAcceptedRows
    exact mem_flatten_ofFn _ index.source
      (mem_flatten_ofFn _ index.coefficient member)
  exact satisfied row (by simp [Program.rows, aggregate])

theorem Program.piRlcAlgebra_satisfied
    {candidate : Id} {rowVariables : Nat}
    {program : Program candidate rowVariables} {assignment : Nat -> Nat}
    (satisfied : Satisfies program.rows assignment) :
    Satisfies (ProductPiRlcAlgebraRows.rows program.piRlcAlgebraLayout)
      assignment := by
  intro row member
  exact satisfied row (by simp [Program.rows, member])

theorem Program.piDec_satisfied
    {candidate : Id} {rowVariables : Nat}
    {program : Program candidate rowVariables} {assignment : Nat -> Nat}
    (satisfied : Satisfies program.rows assignment) :
    Satisfies (ProductPiDecRows.rows program.piDecLayout) assignment := by
  intro row member
  exact satisfied row (by simp [Program.rows, member])

theorem Program.nifsOutput_satisfied
    {candidate : Id} {rowVariables : Nat}
    {program : Program candidate rowVariables} {assignment : Nat -> Nat}
    (satisfied : Satisfies program.rows assignment) :
    Satisfies
      (ProductionProductNifsOutputRowsFor.rows program.nifsOutputLayout
        (ProductPiCcsTranscriptRowsFor.pointAt program.piCcsInput))
      assignment := by
  intro row member
  exact satisfied row (by simp [Program.rows, member])

theorem Program.prior_satisfied
    {candidate : Id} {rowVariables : Nat}
    {program : Program candidate rowVariables} {assignment : Nat -> Nat}
    (satisfied : Satisfies program.rows assignment) :
    Satisfies
      (ProductionPaperPriorStateAuthorityRowsFor.rows program.priorLayout
        program.statementId) assignment := by
  intro row member
  exact satisfied row (by simp [Program.rows, member])

theorem Program.memory_satisfied
    {candidate : Id} {rowVariables : Nat}
    {program : Program candidate rowVariables} {assignment : Nat -> Nat}
    (satisfied : Satisfies program.rows assignment) :
    Satisfies
      (ProductionMemoryCheckedBatchRows.rows
        program.priorLayout.ccs.core.batch.frame.memory) assignment := by
  intro row member
  exact satisfied row (by simp [Program.rows, member])

theorem Program.compact_satisfied
    {candidate : Id} {rowVariables : Nat}
    {program : Program candidate rowVariables} {assignment : Nat -> Nat}
    (satisfied : Satisfies program.rows assignment) :
    Satisfies
      (ProductionFieldNativeCompactChainRowsFor.rows program.seedManifest
        program.compactLayout) assignment := by
  intro row member
  exact satisfied row (by simp [Program.rows, member])

structure Program.MatchesRows
    {candidate : Id} {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape :
      Nightstream.SuperNeo.Concrete.Phi81Relation.Shape}
    (program : Program candidate rowVariables)
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (value : ProductionFieldNativeFullClaim.Value candidate
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (wires : ProductionProductPiCcsTypedBridgeFor.Wires rowVariables)
    (samplerBase : Nat)
    (algebraLayout : ProductPiRlcAlgebraRows.Layout)
    (piDecLayout : ProductPiDecRows.Layout)
    (nifsOutputLayout : ProductionProductNifsOutputRowsFor.Layout rowVariables)
    (priorAuthority : ProductionPaperPriorStateAuthorityRowsFor.Layout
      candidate rowVariables) : Prop where
  piCcsInputExact : program.piCcsInput =
    ProductionProductPiCcsTypedBridgeFor.rowInput candidate statementId config
      artifact value.recursiveState
      (ProductionFieldNativeFullClaim.freshOfValue
        (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
          publicFits).toShape value)
      wires
  samplerBaseExact : program.samplerBase = samplerBase
  algebraLayoutExact : program.piRlcAlgebraLayout = algebraLayout
  piDecLayoutExact : program.piDecLayout = piDecLayout
  nifsOutputLayoutExact : program.nifsOutputLayout = nifsOutputLayout
  priorLayoutExact : program.priorLayout = priorAuthority
  statementIdExact : program.statementId = statementId
  nifsOutputValid : ProductionProductNifsOutputRowsFor.Layout.Valid
    (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
      publicFits) nifsOutputLayout algebraLayout piDecLayout

/-- One satisfied common manifest derives the complete row bundle consumed by
the paper recursive verifier theorem. It does not assume verification or a
memory transition. -/
theorem Program.rows_imply_recursive_rowsHold
    {candidate : Id} {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape :
      Nightstream.SuperNeo.Concrete.Phi81Relation.Shape}
    {program : Program candidate rowVariables}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {value : ProductionFieldNativeFullClaim.Value candidate
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {wires : ProductionProductPiCcsTypedBridgeFor.Wires rowVariables}
    {samplerBase : Nat}
    {algebraLayout : ProductPiRlcAlgebraRows.Layout}
    {piDecLayout : ProductPiDecRows.Layout}
    {nifsOutputLayout : ProductionProductNifsOutputRowsFor.Layout rowVariables}
    {priorAuthority : ProductionPaperPriorStateAuthorityRowsFor.Layout
      candidate rowVariables}
    {assignment : Nat -> Nat}
    (matched : program.MatchesRows statementId config artifact value wires
      samplerBase algebraLayout piDecLayout nifsOutputLayout priorAuthority)
    (satisfied : Satisfies program.rows assignment) :
    ProductionPaperRecursiveRelationRowsSoundFor.RowsHold candidate statementId
      config artifact value proof wires samplerBase algebraLayout piDecLayout
      nifsOutputLayout priorAuthority program.seedManifest program.compactLayout
      assignment := by
  refine
    { piCcs := ?_
      samplerTranscript := ?_
      samplerClassification := ?_
      samplerSelector := ?_
      algebra := ?_
      piDec := ?_
      nifsOutputValid := matched.nifsOutputValid
      nifsOutput := ?_
      memory := ?_
      priorState := ?_
      compactValid := by
        simpa only [matched.priorLayoutExact] using program.compactValid
      compact := program.compact_satisfied satisfied }
  · simpa only [matched.piCcsInputExact] using
      program.piCcs_satisfied satisfied
  · simpa only [Program.piRlcInput,
      ProductionProductPiRlcParentBridgeFor.samplerInput,
      matched.piCcsInputExact, matched.samplerBaseExact] using
      program.piRlcTranscript_satisfied satisfied
  · simpa only [Program.piRlcInput,
      ProductionProductPiRlcParentBridgeFor.samplerInput,
      matched.piCcsInputExact, matched.samplerBaseExact] using
      program.piRlcClassification_satisfied satisfied
  · simpa only [Program.piRlcInput,
      ProductionProductPiRlcParentBridgeFor.samplerInput,
      matched.piCcsInputExact, matched.samplerBaseExact] using
      program.piRlcFirstAccepted_satisfied satisfied
  · simpa only [matched.algebraLayoutExact] using
      program.piRlcAlgebra_satisfied satisfied
  · simpa only [matched.piDecLayoutExact] using program.piDec_satisfied satisfied
  · simpa only [matched.nifsOutputLayoutExact, matched.piCcsInputExact,
      ProductionProductNifsOutputRowsFor.verifierPoint] using
      program.nifsOutput_satisfied satisfied
  · simpa only [matched.priorLayoutExact] using program.memory_satisfied satisfied
  · simpa only [matched.priorLayoutExact, matched.statementIdExact] using
      program.prior_satisfied satisfied

def rowCount (candidate : Id) (rowVariables : Nat) : Nat :=
  ProductionRecursiveCoreGeometryFor.productNifsRows rowVariables +
    ProductionPaperPriorStateAuthorityRowsFor.rowCount candidate rowVariables +
    ProductionMemoryCheckedBatchRows.rowCount candidate +
      ProductionFieldNativeCompactChainRowsFor.rowCount

theorem Program.rows_length_exact
    {candidate : Id} {rowVariables : Nat}
    (program : Program candidate rowVariables) :
    program.rows.length = rowCount candidate rowVariables := by
  simp only [Program.rows, List.length_append,
    ProductPiCcsTranscriptGeometryFor.rows_length_exact program.piCcsInput
      program.piCcsTermCount program.piCcsDegreeSum,
    piRlcTranscriptRows_length_exact,
    piRlcClassificationRows_length_exact,
    piRlcFirstAcceptedRows_length_exact,
    ProductPiRlcAlgebraRows.rows_length,
    ProductPiDecRows.rows_length,
    ProductionProductNifsOutputRowsFor.rows_length,
    ProductionPaperPriorStateAuthorityRowsFor.rows_length_exact
      program.priorValid,
    ProductionMemoryCheckedBatchRows.rows_length_exact,
    ProductionFieldNativeCompactChainRowsFor.rows_length_exact
      program.compactValid]
  simp [rowCount, ProductionRecursiveCoreGeometryFor.productNifsRows,
    Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]

def Program.RowsIncluded
    {candidate : Id} {rowVariables : Nat}
    (program : Program candidate rowVariables) (finalRows : List Row) : Prop :=
  program.rows.Sublist finalRows

theorem Program.satisfies_of_rowsIncluded
    {candidate : Id} {rowVariables : Nat}
    {program : Program candidate rowVariables} {finalRows : List Row}
    {assignment : Nat -> Nat}
    (included : program.RowsIncluded finalRows)
    (satisfied : Satisfies finalRows assignment) :
    Satisfies program.rows assignment := by
  intro row member
  exact satisfied row (included.subset member)

end Nightstream.Implementation.Nebula.ProductionPaperFoldCoreManifestFor
