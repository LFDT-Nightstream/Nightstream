import Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
import Nightstream.Implementation.Nebula.NIFS.Core.ConcreteFor
import Nightstream.Implementation.Nebula.Production.FPrime.Fresh.LinearSubstitution
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.LeanCompiler.StableRows
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConstraintPolynomialPrepend
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CarrierMatrixVector

/-!
Contract: compile one exact low-norm V2 fresh-row program into the
identity-first fourteen-matrix SuperNeo relation.

This module owns a syntax-directed path from numeric R1CS rows to the exact
matrix family used by `ProductConcreteNifsFor`. The same finite assignment is
used by the source rows, the full commitment, the three lane projections, and
the CCS predicate. No caller supplies a separate matrix family or a CCS
satisfaction proposition.

The structural inputs are row-domain fit, assignment-domain fit, and column
scope. They contain no witness values and no acceptance conclusion.

Does not own generation of the source R1CS rows, proof that a deployed row
artifact equals those rows, the application compiler, Rust, the compact
terminal backend, or cryptographic binding.

Assurance tier: compiler correspondence.

Emits constraints: one selective CCS row for each lowered source R1CS row;
all unused Boolean-cube rows are zero. Matrix zero is the padded identity.
-/

set_option autoImplicit false
set_option maxHeartbeats 800000
set_option maxRecDepth 30000

namespace Nightstream.Implementation.Nebula.ProductionFreshRelationCompilerFor

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductionFreshLinearSubstitution
open Nightstream.Implementation.Nebula.ProductionFreshLowNormEncoding
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.CenteredTernaryLinearCompiler
open Nightstream.Implementation.R1CS.SelectiveCcs
open Nightstream.Implementation.R1CS.SelectiveCcs.LeanCompiler
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.UnifiedSources

namespace NumericBridge

def sourceColumn (column : Nat) : ColumnId where
  owner := .prelude
  bundleIndex := 0
  coordinateIndex := column

def finiteColumnIndex
    {columns : Nat} (positive : 0 < columns)
    (column : ColumnId) : Fin columns :=
  ⟨column.coordinateIndex % columns, Nat.mod_lt _ positive⟩

@[simp] theorem finiteColumnIndex_sourceColumn_val
    {columns column : Nat} (positive : 0 < columns) :
    (finiteColumnIndex positive (sourceColumn column)).val =
      column % columns :=
  rfl

theorem finiteColumnIndex_sourceColumn_of_lt
    {columns column : Nat} (positive : 0 < columns)
    (bounded : column < columns) :
    finiteColumnIndex positive (sourceColumn column) =
      ⟨column, bounded⟩ := by
  apply Fin.ext
  simp [finiteColumnIndex, sourceColumn, Nat.mod_eq_of_lt bounded]

def ownedRows (rows : List R1CS.Row) : List OwnedRow :=
  ownedRowsFrom .prelude 0 sourceColumn rows

def directProgram
    {columns : Nat} (positive : 0 < columns)
    (rows : List R1CS.Row) : List (DirectRows.SourceRow columns) :=
  StableRows.program (finiteColumnIndex positive) (ownedRows rows)

@[simp] theorem directProgram_length
    {columns : Nat} (positive : 0 < columns)
    (rows : List R1CS.Row) :
    (directProgram positive rows).length = rows.length := by
  simp [directProgram, StableRows.program, ownedRows,
    ownedRowsFrom_length]

def TermsBelow (columns : Nat) (terms : List (Nat × Nat)) : Prop :=
  ∀ term ∈ terms, term.1 < columns

def RowBelow (columns : Nat) (row : R1CS.Row) : Prop :=
  TermsBelow columns row.a ∧ TermsBelow columns row.b ∧
    TermsBelow columns row.c

def RowsBelow (columns : Nat) (rows : List R1CS.Row) : Prop :=
  ∀ row ∈ rows, RowBelow columns row

private theorem rawLcEval_eq_of_agree
    {columns : Nat} {left right : Nat → Nat}
    (terms : List (Nat × Nat))
    (bounded : TermsBelow columns terms)
    (agree : ∀ column, column < columns → left column = right column) :
    R1CS.Program.rawLcEval left terms =
      R1CS.Program.rawLcEval right terms := by
  induction terms with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [R1CS.Program.rawLcEval]
      rw [agree head.1 (bounded head (by simp))]
      apply congrArg (fun value => head.2 * right head.1 + value)
      exact inductionHypothesis
        (fun term member => bounded term (by simp [member]))

theorem lcEval_eq_of_agree
    {columns : Nat} {left right : Nat → Nat}
    (terms : List (Nat × Nat))
    (bounded : TermsBelow columns terms)
    (agree : ∀ column, column < columns → left column = right column) :
    R1CS.lcEval left terms = R1CS.lcEval right terms := by
  rw [R1CS.Program.lcEval_eq_raw_mod, R1CS.Program.lcEval_eq_raw_mod]
  rw [rawLcEval_eq_of_agree terms bounded agree]

theorem rowHolds_iff_of_agree
    {columns : Nat} {left right : Nat → Nat}
    (row : R1CS.Row) (bounded : RowBelow columns row)
    (agree : ∀ column, column < columns → left column = right column) :
    R1CS.RowHolds left row ↔ R1CS.RowHolds right row := by
  unfold R1CS.RowHolds
  rw [lcEval_eq_of_agree row.a bounded.1 agree,
    lcEval_eq_of_agree row.b bounded.2.1 agree,
    lcEval_eq_of_agree row.c bounded.2.2 agree]

theorem satisfies_iff_of_agree
    {columns : Nat} {left right : Nat → Nat}
    (rows : List R1CS.Row) (bounded : RowsBelow columns rows)
    (agree : ∀ column, column < columns → left column = right column) :
    R1CS.Satisfies rows left ↔ R1CS.Satisfies rows right := by
  constructor <;> intro satisfied row member
  · exact (rowHolds_iff_of_agree row (bounded row member) agree).mp
      (satisfied row member)
  · exact (rowHolds_iff_of_agree row (bounded row member) agree).mpr
      (satisfied row member)

def pulledNumericAssignment
    {columns : Nat} (positive : 0 < columns)
    (assignment : Fin columns → F) : Nat → Nat :=
  numericAssignment sourceColumn
    (StableRows.pulledAssignment (finiteColumnIndex positive) assignment)

theorem pulledNumericAssignment_of_lt
    {columns : Nat} (positive : 0 < columns)
    (assignment : Fin columns → F)
    {column : Nat} (bounded : column < columns) :
    pulledNumericAssignment positive assignment column =
      (assignment ⟨column, bounded⟩).val := by
  unfold pulledNumericAssignment numericAssignment
    StableRows.pulledAssignment
  rw [finiteColumnIndex_sourceColumn_of_lt positive bounded]

/-- Exact numeric meaning of the direct selective program. Column scope is
used only to exclude finite-index wraparound. -/
theorem directProgram_satisfied_iff
    {columns : Nat} (positive : 0 < columns)
    (rows : List R1CS.Row) (bounded : RowsBelow columns rows)
    (assignment : Fin columns → F) :
    (∀ index : Fin (directProgram positive rows).length,
        ((directProgram positive rows).get index).Holds assignment) ↔
      R1CS.Satisfies rows
        (fun column =>
          if within : column < columns then
            (assignment ⟨column, within⟩).val
          else 0) := by
  simp only [directProgram]
  rw [StableRows.program_holds_iff]
  simp only [ownedRows]
  change
    Nightstream.Implementation.Lowering.Goldilocks.Satisfies
        (ownedRowsFrom .prelude 0 sourceColumn rows)
        (StableRows.pulledAssignment (finiteColumnIndex positive) assignment) ↔
      _
  rw [ownedRowsFrom_satisfies_iff]
  apply satisfies_iff_of_agree rows bounded
  intro column columnBound
  unfold numericAssignment StableRows.pulledAssignment
  rw [finiteColumnIndex_sourceColumn_of_lt positive columnBound]
  simp only [dif_pos columnBound]

end NumericBridge

/-- Structural compiler input. These fields constrain only dimensions and
column references. They do not assert row satisfaction. -/
structure SourceProgram (privateWidth rowVariables : Nat) where
  rows : List R1CS.Row
  sourceColumns : NumericBridge.RowsBelow
    (sourceWidth privateWidth) rows
  loweredColumns : NumericBridge.RowsBelow
    (logicalWidth privateWidth) (loweredRows (layout privateWidth) rows)
  rowDomain : RelationProfile.ExactRowDomain rows.length rowVariables
  carrierFits : logicalWidth privateWidth ≤ 2 ^ rowVariables

namespace SourceProgram

def loweredRowsFor
    {privateWidth rowVariables : Nat}
    (program : SourceProgram privateWidth rowVariables) : List R1CS.Row :=
  loweredRows (layout privateWidth) program.rows

@[simp] theorem loweredRowsFor_length
    {privateWidth rowVariables : Nat}
    (program : SourceProgram privateWidth rowVariables) :
    program.loweredRowsFor.length = program.rows.length := by
  simp [loweredRowsFor, loweredRows]

theorem logicalWidth_positive
    {privateWidth rowVariables : Nat}
    (program : SourceProgram privateWidth rowVariables) :
    0 < logicalWidth privateWidth := by
  have directPositive : 0 < directWidth := by decide
  exact Nat.lt_of_lt_of_le directPositive
    (Nat.le_trans (directWidth_le_payloadWidth privateWidth)
      (payloadWidth_le_logicalWidth privateWidth))

def directProgram
    {privateWidth rowVariables : Nat}
    (program : SourceProgram privateWidth rowVariables) :
    List (DirectRows.SourceRow (logicalWidth privateWidth)) :=
  NumericBridge.directProgram program.logicalWidth_positive
    program.loweredRowsFor

@[simp] theorem directProgram_length
    {privateWidth rowVariables : Nat}
    (program : SourceProgram privateWidth rowVariables) :
    program.directProgram.length = program.rows.length := by
  simp [directProgram, NumericBridge.directProgram_length]

def relationProfile
    {privateWidth rowVariables : Nat}
    (program : SourceProgram privateWidth rowVariables) :
    RelationProfile.Profile program.directProgram.length
      (logicalWidth privateWidth) where
  rowVariables := rowVariables
  rowDomain := by simpa using program.rowDomain
  publicRingColumns := 10
  publicFits := by
    have public540 : 540 ≤ logicalWidth privateWidth :=
      publicWidth_le_logicalWidth privateWidth
    simpa [ringDegree, logicalWidth_carrier_exact] using public540

def applicationRelation
    {privateWidth rowVariables : Nat}
    (program : SourceProgram privateWidth rowVariables) :=
  DirectRows.relation
    (NumericBridge.finiteColumnIndex program.logicalWidth_positive
      (NumericBridge.sourceColumn 0))
    program.directProgram

def applicationSystem
    {privateWidth rowVariables : Nat}
    (program : SourceProgram privateWidth rowVariables) :=
  DirectRows.paperSystem program.applicationRelation program.relationProfile

def cubeLayout
    {privateWidth rowVariables : Nat}
    (program : SourceProgram privateWidth rowVariables) :
    ColumnLayout rowVariables
      (Phi81CarrierLayout.carrierWidth (logicalWidth privateWidth)) :=
  Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PrefixLayout.layout
    rowVariables
    (Phi81CarrierLayout.carrierWidth (logicalWidth privateWidth))
    (by simpa [logicalWidth_carrier_exact] using program.carrierFits)

def identityMatrix
    {privateWidth rowVariables : Nat}
    (program : SourceProgram privateWidth rowVariables) :
    BooleanMatrix F rowVariables (logicalWidth privateWidth) :=
  fun vertex column =>
    program.cubeLayout.paddedIdentityEntry 0 1 vertex
      (Phi81CarrierLayout.embedLogical column)

/-- Exact V2 relation: identity first, followed by the thirteen direct
selective matrices. -/
def system
    {privateWidth rowVariables : Nat}
    (program : SourceProgram privateWidth rowVariables) :
    Phi81Relation.Structure
      (ProductPaperAlgebraFor.FullShape rowVariables
        (logicalWidth privateWidth) (publicFits privateWidth)) where
  matrices := Fin.cases program.identityMatrix
    (fun matrix => program.applicationSystem.matrices matrix)
  constraintPolynomial :=
    ConstraintPolynomialPrepend.prependIgnoredVariable
      program.applicationSystem.constraintPolynomial

def identityIndex
    {privateWidth rowVariables : Nat}
    (program : SourceProgram privateWidth rowVariables) :
    Fin (ProductPaperAlgebraFor.FullShape rowVariables
      (logicalWidth privateWidth) (publicFits privateWidth)).matrixCount :=
  ⟨0, by simp [ProductPaperAlgebraFor.FullShape,
    ProductPaperAlgebraFor.fullShape]⟩

@[simp] theorem system_identity_entry
    {privateWidth rowVariables : Nat}
    (program : SourceProgram privateWidth rowVariables)
    (vertex : BooleanVertex rowVariables)
    (column : Fin (logicalWidth privateWidth)) :
    program.system.matrices program.identityIndex vertex column =
      program.cubeLayout.paddedIdentityEntry 0 1 vertex
        (Phi81CarrierLayout.embedLogical column) := by
  rfl

theorem system_degree_exact
    {privateWidth rowVariables : Nat}
    (program : SourceProgram privateWidth rowVariables) :
    program.system.constraintPolynomial.canonicalEqualityGatedDegreeBound =
      9 := by
  change
    (ConstraintPolynomialPrepend.prependIgnoredVariable
      Semantics.polynomial).canonicalEqualityGatedDegreeBound = 9
  rw [ConstraintPolynomialPrepend.prependIgnoredVariable_canonicalEqualityGatedDegreeBound]
  exact Semantics.canonicalEqualityGatedDegreeBound_exact

def relationArtifact
    {privateWidth rowVariables : Nat}
    (program : SourceProgram privateWidth rowVariables) :
    ProductConcreteNifsFor.RelationArtifact rowVariables
      (logicalWidth privateWidth) (publicFits privateWidth) where
  system := program.system
  cubeFits := by
    simpa [logicalWidth_carrier_exact] using program.carrierFits
  degreeBoundExact := by
    rw [ConstraintPolynomialLift.liftConstraintPolynomial_canonicalEqualityGatedDegreeBound]
    change Nat.max
      program.system.constraintPolynomial.canonicalEqualityGatedDegreeBound
        4 = 9
    rw [program.system_degree_exact]
    decide
  identityFirstEntry := by
    intro vertex column
    change
      Phi81CarrierLayout.extendMatrix 0 program.identityMatrix vertex column =
        program.cubeLayout.paddedIdentityEntry 0 1 vertex column
    rw [show column = Phi81CarrierLayout.embedLogical
        (carrierToLogical column) by
      exact (embedLogical_carrierToLogical column).symm]
    rw [Phi81CarrierLayout.extendMatrix_embedLogical]
    rfl

/-- Logical-prefix view of the exact carrier assignment. This is a view, not
a second witness. -/
def logicalAssignment
    {privateWidth rowVariables : Nat}
    (program : SourceProgram privateWidth rowVariables)
    (assignment : ProductPaperAlgebraFor.Assignment rowVariables
      (logicalWidth privateWidth) (publicFits privateWidth)) :
    Assignment F (logicalWidth privateWidth) :=
  fun column => assignment (Phi81CarrierLayout.embedLogical column)

theorem matrixVectorAt_source_succ_eq_application
    {privateWidth rowVariables : Nat}
    (program : SourceProgram privateWidth rowVariables)
    (assignment : ProductPaperAlgebraFor.Assignment rowVariables
      (logicalWidth privateWidth) (publicFits privateWidth))
    (vertex : BooleanVertex rowVariables)
    (matrix : Fin 13) :
    matrixVectorAt baseOps
        ((ProductPaperAlgebraFor.matrixSource program.system).system.matrices
          matrix.succ)
        assignment vertex =
      matrixVectorAt baseOps
        (program.applicationSystem.matrices matrix)
        (program.logicalAssignment assignment) vertex := by
  change
    matrixVectorAt baseOps
        (Phi81CarrierLayout.extendMatrix 0
          (program.applicationSystem.matrices matrix)) assignment vertex =
      _
  exact Phi81CarrierMatrixVector.matrixVectorAt_extendMatrix_eq
    (program.applicationSystem.matrices matrix) assignment vertex

/-- Prepending the authority-bearing identity matrix does not change the
application residual because the exact fourteen-variable polynomial ignores
that new head image. Carrier completion also preserves every matrix-vector
product over the same assignment's logical prefix. -/
theorem residualAt_matrixSource_eq_application
    {privateWidth rowVariables : Nat}
    (program : SourceProgram privateWidth rowVariables)
    (assignment : ProductPaperAlgebraFor.Assignment rowVariables
      (logicalWidth privateWidth) (publicFits privateWidth))
    (vertex : BooleanVertex rowVariables) :
    residualAt baseOps
        (ProductPaperAlgebraFor.matrixSource program.system).system
        assignment vertex =
      residualAt baseOps program.applicationSystem
        (program.logicalAssignment assignment) vertex := by
  unfold residualAt
  change
    evaluatePolynomial baseOps
        (ConstraintPolynomialPrepend.prependIgnoredVariable
          program.applicationSystem.constraintPolynomial)
        (matrixImagesAt baseOps
          (ProductPaperAlgebraFor.matrixSource program.system).system
          assignment vertex) =
      evaluatePolynomial baseOps program.applicationSystem.constraintPolynomial
        (matrixImagesAt baseOps program.applicationSystem
          (program.logicalAssignment assignment) vertex)
  rw [ConstraintPolynomialPrepend.evaluatePolynomial_prependIgnoredVariable
    baseOps baseLaws]
  congr 1
  funext matrix
  simpa only [matrixImagesAt] using
    program.matrixVectorAt_source_succ_eq_application assignment vertex matrix

/-- Exact relation reduction from the fourteen-matrix carrier system to the
thirteen-matrix application system on the same assignment's logical view. -/
theorem ccsSatisfied_iff_application
    {privateWidth rowVariables : Nat}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (program : SourceProgram privateWidth rowVariables)
    (config : ProductPaperAlgebraFor.Config rowVariables
      (logicalWidth privateWidth) (publicFits privateWidth)
      operationsShape snapshotShape)
    (assignment : ProductPaperAlgebraFor.Assignment rowVariables
      (logicalWidth privateWidth) (publicFits privateWidth)) :
    (ProductPaperAlgebraFor.semantics config).ccsSatisfied
        (ProductPaperAlgebraFor.matrixSource program.system) assignment ↔
      ConstraintSatisfied baseOps program.applicationSystem
        (program.logicalAssignment assignment) := by
  constructor <;> intro satisfied vertex
  · rw [← program.residualAt_matrixSource_eq_application assignment vertex]
    exact satisfied vertex
  · rw [program.residualAt_matrixSource_eq_application assignment vertex]
    exact satisfied vertex

/-- Exact reduction from carrier CCS satisfaction to the compiled numeric
R1CS rows. The only semantic side condition is the designated constant-one
coordinate required by the selective relation. -/
theorem ccsSatisfied_iff_loweredRows
    {privateWidth rowVariables : Nat}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (program : SourceProgram privateWidth rowVariables)
    (config : ProductPaperAlgebraFor.Config rowVariables
      (logicalWidth privateWidth) (publicFits privateWidth)
      operationsShape snapshotShape)
    (assignment : ProductPaperAlgebraFor.Assignment rowVariables
      (logicalWidth privateWidth) (publicFits privateWidth))
    (constantOne :
      program.logicalAssignment assignment
          (NumericBridge.finiteColumnIndex program.logicalWidth_positive
            (NumericBridge.sourceColumn 0)) = 1) :
    (ProductPaperAlgebraFor.semantics config).ccsSatisfied
        (ProductPaperAlgebraFor.matrixSource program.system) assignment ↔
      R1CS.Satisfies program.loweredRowsFor
        (fun column =>
          if within : column < logicalWidth privateWidth then
            (program.logicalAssignment assignment ⟨column, within⟩).val
          else 0) := by
  rw [program.ccsSatisfied_iff_application config assignment]
  unfold applicationSystem applicationRelation
  rw [DirectRows.constraintSatisfied_iff
    (NumericBridge.finiteColumnIndex program.logicalWidth_positive
      (NumericBridge.sourceColumn 0))
    program.directProgram program.relationProfile
    (program.logicalAssignment assignment) constantOne]
  exact NumericBridge.directProgram_satisfied_iff
    program.logicalWidth_positive program.loweredRowsFor
      program.loweredColumns (program.logicalAssignment assignment)

/-! ## Reverse extraction from an arbitrary accepted carrier

The honest encoder below chooses one shifted-ternary word for each private
field. Soundness cannot assume that an extracted bounded assignment used that
choice: distinct accepted words can decode to the same field residue. The
definitions in this section retain the exact carrier coordinates and decode
the source assignment from them. -/

/-- Numeric view of the exact committed logical carrier. No private word is
re-encoded through a semantic field value. -/
def encodedNatOfAssignment
    {privateWidth rowVariables : Nat}
    (program : SourceProgram privateWidth rowVariables)
    (assignment : ProductPaperAlgebraFor.Assignment rowVariables
      (logicalWidth privateWidth) (publicFits privateWidth)) : Nat → Nat :=
  fun column =>
    if within : column < logicalWidth privateWidth then
      (program.logicalAssignment assignment ⟨column, within⟩).val
    else 0

/-- Source-field view decoded from the exact committed words. Direct columns
are copied; private columns are the linear shifted-ternary decode. -/
def decodedSourceAssignment
    {privateWidth rowVariables : Nat}
    (program : SourceProgram privateWidth rowVariables)
    (assignment : ProductPaperAlgebraFor.Assignment rowVariables
      (logicalWidth privateWidth) (publicFits privateWidth)) : Nat → Nat :=
  CenteredTernaryLinearCompiler.decodedAssignment
    (ProductionFreshLinearSubstitution.layout privateWidth)
    (program.encodedNatOfAssignment assignment)

theorem decodedSourceAssignment_canonical
    {privateWidth rowVariables : Nat}
    (program : SourceProgram privateWidth rowVariables)
    (assignment : ProductPaperAlgebraFor.Assignment rowVariables
      (logicalWidth privateWidth) (publicFits privateWidth))
    (column : Nat) :
    program.decodedSourceAssignment assignment column < R1CS.goldilocksP := by
  unfold decodedSourceAssignment
    CenteredTernaryLinearCompiler.decodedAssignment
    LinearSubstitution.assignment R1CS.lcEval
  exact Nat.mod_lt _ (by decide)

theorem encodedNatOfAssignment_of_lt
    {privateWidth rowVariables column : Nat}
    (program : SourceProgram privateWidth rowVariables)
    (assignment : ProductPaperAlgebraFor.Assignment rowVariables
      (logicalWidth privateWidth) (publicFits privateWidth))
    (within : column < logicalWidth privateWidth) :
    program.encodedNatOfAssignment assignment column =
      (program.logicalAssignment assignment ⟨column, within⟩).val := by
  simp [encodedNatOfAssignment, within]

/-- Every direct source coordinate is the exact corresponding committed
carrier coordinate. This lemma does not assume the honest encoder image. -/
theorem decodedSourceAssignment_direct
    {privateWidth rowVariables : Nat}
    (program : SourceProgram privateWidth rowVariables)
    (assignment : ProductPaperAlgebraFor.Assignment rowVariables
      (logicalWidth privateWidth) (publicFits privateWidth))
    (column : Fin directWidth) :
    program.decodedSourceAssignment assignment column.val =
      (program.logicalAssignment assignment
        ⟨column.val, Nat.lt_of_lt_of_le column.isLt
          (Nat.le_trans (directWidth_le_payloadWidth privateWidth)
            (payloadWidth_le_logicalWidth privateWidth))⟩).val := by
  let within : column.val < logicalWidth privateWidth :=
    Nat.lt_of_lt_of_le column.isLt
      (Nat.le_trans (directWidth_le_payloadWidth privateWidth)
        (payloadWidth_le_logicalWidth privateWidth))
  unfold decodedSourceAssignment
    CenteredTernaryLinearCompiler.decodedAssignment
    LinearSubstitution.assignment
  rw [show (ProductionFreshLinearSubstitution.layout privateWidth).expansion
      column.val = [(column.val, 1)] by
    exact ProductionFreshLinearSubstitution.expansion_direct column.isLt]
  simp only [R1CS.lcEval, List.foldl, Nat.zero_add, Nat.one_mul]
  rw [program.encodedNatOfAssignment_of_lt assignment within]
  rw [Nat.mod_eq_of_lt]
  exact (program.logicalAssignment assignment ⟨column.val, within⟩).isLt

/-- Exact arbitrary-assignment compiler equivalence. The left side is the
SuperNeo CCS predicate. The right side evaluates the original source rows on
the source values decoded from the same committed carrier assignment. -/
theorem ccsSatisfied_iff_decodedSourceRows
    {privateWidth rowVariables : Nat}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (program : SourceProgram privateWidth rowVariables)
    (config : ProductPaperAlgebraFor.Config rowVariables
      (logicalWidth privateWidth) (publicFits privateWidth)
      operationsShape snapshotShape)
    (assignment : ProductPaperAlgebraFor.Assignment rowVariables
      (logicalWidth privateWidth) (publicFits privateWidth))
    (constantOne :
      program.logicalAssignment assignment
          (NumericBridge.finiteColumnIndex program.logicalWidth_positive
            (NumericBridge.sourceColumn 0)) = 1) :
    (ProductPaperAlgebraFor.semantics config).ccsSatisfied
        (ProductPaperAlgebraFor.matrixSource program.system) assignment ↔
      R1CS.Satisfies program.rows
        (program.decodedSourceAssignment assignment) := by
  rw [program.ccsSatisfied_iff_loweredRows config assignment constantOne]
  exact CenteredTernaryLinearCompiler.loweredRows_iff_sourceRows
    (ProductionFreshLinearSubstitution.layout privateWidth) program.rows
      (program.encodedNatOfAssignment assignment)

@[simp] theorem logicalAssignment_encodeCarrier
    {privateWidth rowVariables : Nat}
    (program : SourceProgram privateWidth rowVariables)
    (source : SourceAssignment privateWidth) :
    program.logicalAssignment
        (encodeCarrier source : ProductPaperAlgebraFor.Assignment rowVariables
          (logicalWidth privateWidth) (publicFits privateWidth)) =
      encodeLogical source := by
  funext column
  unfold logicalAssignment encodeCarrier
  exact Phi81CarrierLayout.extendAssignment_embedLogical 0
    (encodeLogical source) column

/-- Public projection of the exponent-indexed carrier is the exact public
prefix of the same source assignment. This is the generated-row-exponent
version of `ProductionFreshLowNormEncoding.projectPublicInput_encodeCarrier`;
it does not pass through the fixed-25 reference shape. -/
theorem projectPublicInput_encodeCarrier_for
    {privateWidth rowVariables : Nat}
    (source : SourceAssignment privateWidth) :
    @Phi81Relation.projectPublicInput
        (ProductPaperAlgebraFor.fullShape rowVariables
          (logicalWidth privateWidth) (publicFits privateWidth))
        (encodeCarrier source : ProductPaperAlgebraFor.Assignment rowVariables
          (logicalWidth privateWidth) (publicFits privateWidth)) =
      fun column => source (publicSourceColumn column) := by
  funext column
  let logicalColumn : Fin (logicalWidth privateWidth) :=
    payloadColumn (finSumFinEquiv (Sum.inl
      (Fin.castLE publicWidth_le_directWidth column) :
      Fin directWidth ⊕ Fin (privateWidth *
        Nightstream.Protocol.Nebula.ShiftedTernary41V1.digitCount)))
  have carrierEq :
      (ProductPaperAlgebraFor.fullShape rowVariables
        (logicalWidth privateWidth) (publicFits privateWidth)).publicColumn
          column =
        Phi81CarrierLayout.embedLogical logicalColumn := by
    rfl
  unfold Phi81Relation.projectPublicInput
  rw [carrierEq]
  have exactValue :
      encodeCarrier source (Phi81CarrierLayout.embedLogical logicalColumn) =
        encodeLogical source logicalColumn :=
    Phi81CarrierLayout.extendAssignment_embedLogical 0
      (encodeLogical source) logicalColumn
  exact exactValue.trans (encodeLogical_public source column)

/-- The low-norm assignment satisfies the lowered rows exactly when its
single source witness satisfies the original numeric rows. -/
theorem encoded_loweredRows_iff_sourceRows
    {privateWidth rowVariables : Nat}
    (program : SourceProgram privateWidth rowVariables)
    (source : SourceAssignment privateWidth) :
    R1CS.Satisfies program.loweredRowsFor (encodedNat source) ↔
      R1CS.Satisfies program.rows (sourceNat source) := by
  unfold loweredRowsFor
  rw [CenteredTernaryLinearCompiler.loweredRows_iff_sourceRows
    (layout privateWidth) program.rows (encodedNat source)]
  apply NumericBridge.satisfies_iff_of_agree program.rows
    program.sourceColumns
  intro column bounded
  rw [decoded_source_column source ⟨column, bounded⟩]
  exact (sourceNat_sourceColumn source ⟨column, bounded⟩).symm

/-- The first public source coordinate is the constant-one coordinate used by
the selective CCS compiler. -/
def SourceOne {privateWidth : Nat}
    (source : SourceAssignment privateWidth) : Prop :=
  source (publicSourceColumn ⟨0, by decide⟩) = 1

def PublicMatches {privateWidth : Nat}
    (source : SourceAssignment privateWidth)
    (publicInput : Fin publicWidth → F) : Prop :=
  ∀ column, source (publicSourceColumn column) = publicInput column

theorem sourceOne_of_publicMatches
    {privateWidth : Nat}
    (source : SourceAssignment privateWidth)
    (publicInput : Fin publicWidth → F)
    (publicExact : PublicMatches source publicInput)
    (publicZero : publicInput ⟨0, by decide⟩ = 1) :
    SourceOne source := by
  exact (publicExact ⟨0, by decide⟩).trans publicZero

theorem constantOne_encodeCarrier
    {privateWidth rowVariables : Nat}
    (program : SourceProgram privateWidth rowVariables)
    (source : SourceAssignment privateWidth)
    (sourceOne : SourceOne source) :
    program.logicalAssignment
        (encodeCarrier source : ProductPaperAlgebraFor.Assignment rowVariables
          (logicalWidth privateWidth) (publicFits privateWidth))
        (NumericBridge.finiteColumnIndex program.logicalWidth_positive
          (NumericBridge.sourceColumn 0)) = 1 := by
  rw [program.logicalAssignment_encodeCarrier source]
  rw [NumericBridge.finiteColumnIndex_sourceColumn_of_lt
    program.logicalWidth_positive program.logicalWidth_positive]
  apply Fin.ext
  have direct := encodedNat_direct source (⟨0, by decide⟩ : Fin directWidth)
  unfold encodedNat at direct
  simp only [dif_pos program.logicalWidth_positive] at direct
  rw [direct]
  simpa [SourceOne, publicSourceColumn] using congrArg Fin.val sourceOne

/-- End-to-end compiler equivalence for one exact low-norm source witness.
There is no separately supplied CCS assignment or satisfaction proposition. -/
theorem encoded_ccsSatisfied_iff_sourceRows
    {privateWidth rowVariables : Nat}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (program : SourceProgram privateWidth rowVariables)
    (config : ProductPaperAlgebraFor.Config rowVariables
      (logicalWidth privateWidth) (publicFits privateWidth)
      operationsShape snapshotShape)
    (source : SourceAssignment privateWidth)
    (sourceOne : SourceOne source) :
    (ProductPaperAlgebraFor.semantics config).ccsSatisfied
        (ProductPaperAlgebraFor.matrixSource program.system)
        (encodeCarrier source : ProductPaperAlgebraFor.Assignment rowVariables
          (logicalWidth privateWidth) (publicFits privateWidth)) ↔
      R1CS.Satisfies program.rows (sourceNat source) := by
  rw [program.ccsSatisfied_iff_loweredRows config
    (encodeCarrier source : ProductPaperAlgebraFor.Assignment rowVariables
      (logicalWidth privateWidth) (publicFits privateWidth))
    (program.constantOne_encodeCarrier source sourceOne)]
  rw [program.logicalAssignment_encodeCarrier source]
  change R1CS.Satisfies program.loweredRowsFor (encodedNat source) ↔ _
  exact program.encoded_loweredRows_iff_sourceRows source

end SourceProgram

end Nightstream.Implementation.Nebula.ProductionFreshRelationCompilerFor
