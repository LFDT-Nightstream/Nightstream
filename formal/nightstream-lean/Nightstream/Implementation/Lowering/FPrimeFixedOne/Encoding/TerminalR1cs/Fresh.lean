import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Ajtai
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.FreshCcs
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Norm
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Projection

/-!
Contract: uniform R1CS relation for one selected fresh SuperNeo CCS opening.

Assurance tier: model-level.

Owns: one shared physical assignment, Ajtai commitment binding, public
projection binding, strict `b = 2` norm rows, the exact native four-matrix CCS
rows, their ordered composition, exact cost, and soundness to `CCS.Holds`.

Does not own: the fourteen running CE claims, their run-time evaluation
points, terminal control flow, input codecs, a deployment manifest, Rust,
Spartan, WHIR, or Ajtai binding security.

Emits constraints: the sum of Ajtai, projection, norm, and native-CCS
lowering rows. Only norm squares and native-CCS residuals are allocated.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Fresh

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsProgram
open Nightstream.Implementation.Lowering.Typed
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

private abbrev RelationShape
    (program : NativeCcsProgram.Program)
    (domain : NativeCcsCompiler.RowDomain program)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length) :=
  NativeCcsPhi81.shape program domain publicRingColumns publicFits

/-- Batch shape whose batch-free relation is the exact native program. -/
def semanticShape
    (program : NativeCcsProgram.Program)
    (domain : NativeCcsCompiler.RowDomain program) :
    SemanticShape where
  rowVariables := domain.rowVariables
  logicalWidth := program.columnIds.length
  freshCount := 1
  runningCount := productionGlobalParams.k
  matrixCount := NativeCcsSelector.matrixCount

/-- One coherent physical placement for all fresh-membership checks. -/
structure Frame
    (program : NativeCcsProgram.Program)
    (domain : NativeCcsCompiler.RowDomain program)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length)
    (verifierRows : Nat) where
  owner : PhysicalOwner
  firstOrdinal : Nat
  one : ColumnId
  key : Commitment.Key
    (RelationShape program domain publicRingColumns publicFits) verifierRows
  witness :
    Fin (RelationShape program domain publicRingColumns publicFits).carrierWidth →
      ColumnId
  commitment : Fin verifierRows → Fin ringDegree → ColumnId
  publicColumn :
    Fin (RelationShape program domain publicRingColumns publicFits).publicWidth →
      ColumnId
  square :
    Fin (RelationShape program domain publicRingColumns publicFits).carrierWidth →
      ColumnId
  residual : Fin program.rows.length → ColumnId

def ajtaiFrame
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (frame : Frame program domain publicRingColumns publicFits verifierRows) :
    Ajtai.Frame
      (RelationShape program domain publicRingColumns publicFits)
      verifierRows where
  owner := frame.owner
  firstOrdinal := frame.firstOrdinal
  one := frame.one
  key := frame.key
  witness := frame.witness
  commitment := frame.commitment

def projectionFrame
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (frame : Frame program domain publicRingColumns publicFits verifierRows) :
    Projection.Frame
      (RelationShape program domain publicRingColumns publicFits) where
  owner := frame.owner
  firstOrdinal := frame.firstOrdinal + verifierRows * ringDegree
  one := frame.one
  witness := frame.witness
  publicColumn := frame.publicColumn

def normFrame
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (frame : Frame program domain publicRingColumns publicFits verifierRows) :
    Norm.Frame
      (RelationShape program domain publicRingColumns publicFits) where
  owner := frame.owner
  firstOrdinal :=
    frame.firstOrdinal + verifierRows * ringDegree +
      (RelationShape program domain publicRingColumns publicFits).publicWidth
  witness := frame.witness
  square := frame.square

def ccsFrame
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (frame : Frame program domain publicRingColumns publicFits verifierRows) :
    FreshCcs.Frame program domain publicRingColumns publicFits where
  owner := frame.owner
  firstOrdinal :=
    frame.firstOrdinal + verifierRows * ringDegree +
      (RelationShape program domain publicRingColumns publicFits).publicWidth +
      2 * (RelationShape program domain publicRingColumns publicFits).carrierWidth
  witness := frame.witness
  residual := frame.residual

/-- Ordered uniform fresh-membership rows. -/
def rows
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (valid : NativeCcsCompiler.Valid program)
    (frame : Frame program domain publicRingColumns publicFits verifierRows) :
    List OwnedRow :=
  Ajtai.rows (ajtaiFrame frame) ++
    (Projection.rows (projectionFrame frame) ++
      (Norm.rows (normFrame frame) ++
        FreshCcs.rows valid (ccsFrame frame)))

/-- Exact auxiliary allocation. Statement and witness columns are inputs. -/
def columns
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (frame : Frame program domain publicRingColumns publicFits verifierRows) :
    List OwnedColumn :=
  Norm.columns (normFrame frame) ++ FreshCcs.columns (ccsFrame frame)

/-- Every row in the composed fresh program keeps the frame owner. -/
theorem rows_owned
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (valid : NativeCcsCompiler.Valid program)
    (frame : Frame program domain publicRingColumns publicFits verifierRows)
    (owned : OwnedRow)
    (member : owned ∈ rows valid frame) :
    owned.id.owner = frame.owner := by
  simp only [rows, List.mem_append] at member
  rcases member with ajtaiMember |
      (projectionMember | (normMember | ccsMember))
  · exact Ajtai.rows_owned (ajtaiFrame frame) owned ajtaiMember
  · exact Projection.rows_owned (projectionFrame frame) owned
      projectionMember
  · exact Norm.rows_owned (normFrame frame) owned normMember
  · exact FreshCcs.rows_owned valid (ccsFrame frame) owned ccsMember

@[simp] theorem rows_length
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (valid : NativeCcsCompiler.Valid program)
    (frame : Frame program domain publicRingColumns publicFits verifierRows) :
    (rows valid frame).length =
      verifierRows * ringDegree +
        (RelationShape program domain publicRingColumns publicFits).publicWidth +
        2 * (RelationShape program domain publicRingColumns publicFits).carrierWidth +
        2 * program.rows.length := by
  simp [rows]
  omega

@[simp] theorem columns_length
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (frame : Frame program domain publicRingColumns publicFits verifierRows) :
    (columns frame).length =
      (RelationShape program domain publicRingColumns publicFits).carrierWidth +
        program.rows.length := by
  simp [columns]

private theorem split_satisfaction
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (valid : NativeCcsCompiler.Valid program)
    (frame : Frame program domain publicRingColumns publicFits verifierRows)
    (assignment : ColumnId → F)
    (satisfied : Satisfies (rows valid frame) assignment) :
    Satisfies (Ajtai.rows (ajtaiFrame frame)) assignment ∧
      Satisfies (Projection.rows (projectionFrame frame)) assignment ∧
      Satisfies (Norm.rows (normFrame frame)) assignment ∧
      Satisfies (FreshCcs.rows valid (ccsFrame frame)) assignment := by
  have first :=
    (satisfies_append_iff
      (Ajtai.rows (ajtaiFrame frame))
      (Projection.rows (projectionFrame frame) ++
        Norm.rows (normFrame frame) ++
          FreshCcs.rows valid (ccsFrame frame))
      assignment).mp (by simpa [rows] using satisfied)
  have firstTail :
      Satisfies
        (Projection.rows (projectionFrame frame) ++
          (Norm.rows (normFrame frame) ++
            FreshCcs.rows valid (ccsFrame frame)))
        assignment := by
    simpa only [List.append_assoc] using first.2
  have second :=
    (satisfies_append_iff
      (Projection.rows (projectionFrame frame))
      (Norm.rows (normFrame frame) ++
        FreshCcs.rows valid (ccsFrame frame))
      assignment).mp firstTail
  have third :=
    (satisfies_append_iff
      (Norm.rows (normFrame frame))
      (FreshCcs.rows valid (ccsFrame frame))
      assignment).mp second.2
  exact ⟨first.1, second.1, third.1, third.2⟩

/-- Exact four semantic facts forced by the physical fresh rows. -/
theorem rows_facts
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (valid : NativeCcsCompiler.Valid program)
    (frame : Frame program domain publicRingColumns publicFits verifierRows)
    (assignment : ColumnId → F)
    (constantOne : assignment frame.one = 1)
    (satisfied : Satisfies (rows valid frame) assignment) :
    (fun verifierRow output =>
        assignment (frame.commitment verifierRow output)) =
        Commitment.commit frame.key
          (fun coordinate => assignment (frame.witness coordinate)) ∧
      (fun coordinate => assignment (frame.publicColumn coordinate)) =
        Phi81Relation.projectPublicInput
          (fun coordinate => assignment (frame.witness coordinate)) ∧
      Phi81Relation.assignmentNormBounded 2
        (fun coordinate => assignment (frame.witness coordinate)) ∧
      Phi81Relation.ccsSatisfied
        (NativeCcsPhi81.relation program valid domain
          publicRingColumns publicFits)
        (fun coordinate => assignment (frame.witness coordinate)) := by
  rcases split_satisfaction valid frame assignment satisfied with
    ⟨ajtaiSatisfied, projectionSatisfied, normSatisfied, ccsSatisfied⟩
  exact ⟨
    Ajtai.rows_sound (ajtaiFrame frame) assignment constantOne
      ajtaiSatisfied,
    Projection.rows_sound (projectionFrame frame) assignment constantOne
      projectionSatisfied,
    Norm.rows_sound noZeroDivisors (normFrame frame) assignment normSatisfied,
    FreshCcs.rows_ccsSound valid (ccsFrame frame) assignment ccsSatisfied
  ⟩

/-- Physical fresh rows plus the statement-column interpretation establish
the exact paper `CCS.Holds` relation. -/
theorem rows_sound
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (valid : NativeCcsCompiler.Valid program)
    (frame : Frame program domain publicRingColumns publicFits verifierRows)
    (payload :
      FixedActive.Canonical.FreshPayload
        (semanticShape program domain)
        publicRingColumns publicFits verifierRows)
    (assignment : ColumnId → F)
    (constantOne : assignment frame.one = 1)
    (commitmentColumns :
      (fun verifierRow output =>
        assignment (frame.commitment verifierRow output)) =
          payload.commitment)
    (publicColumns :
      (fun coordinate => assignment (frame.publicColumn coordinate)) =
        payload.publicInput)
    (satisfied : Satisfies (rows valid frame) assignment) :
    CCS.Holds
      (Phi81Relation.relationSemantics (Commitment.commit frame.key))
      productionGlobalParams
      (payload.materialize
        (NativeCcsPhi81.relation program valid domain
          publicRingColumns publicFits))
      (fun coordinate => assignment (frame.witness coordinate)) := by
  rcases rows_facts noZeroDivisors valid frame assignment constantOne
      satisfied with
    ⟨commitmentFact, publicFact, normFact, ccsFact⟩
  apply
    (Phi81Relation.ccsMembership_iff
      (Commitment.commit frame.key) productionGlobalParams
      (payload.materialize
        (NativeCcsPhi81.relation program valid domain
          publicRingColumns publicFits))
      (fun coordinate => assignment (frame.witness coordinate))).mpr
  refine ⟨?_, ?_, ?_, ccsFact⟩
  · exact commitmentFact.symm.trans commitmentColumns
  · unfold Phi81Relation.publicInputMatches
    exact publicFact.symm.trans publicColumns
  · simpa [FixedActive.Canonical.FreshPayload.materialize,
      NormStage.bound, productionGlobalParams] using normFact

/-- Honest statement columns, witness values, norm squares, and native-CCS
residuals satisfy the complete fresh terminal program. -/
theorem rows_honest
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (valid : NativeCcsCompiler.Valid program)
    (frame : Frame program domain publicRingColumns publicFits verifierRows)
    (assignment : ColumnId → F)
    (constantOne : assignment frame.one = 1)
    (distinct : Norm.Distinct (normFrame frame))
    (commitmentMatches :
      (fun verifierRow output =>
        assignment (frame.commitment verifierRow output)) =
          Commitment.commit frame.key
            (fun coordinate => assignment (frame.witness coordinate)))
    (publicMatches :
      (fun coordinate => assignment (frame.publicColumn coordinate)) =
        Phi81Relation.projectPublicInput
          (fun coordinate => assignment (frame.witness coordinate)))
    (bounded :
      Phi81Relation.assignmentNormBounded 2
        (fun coordinate => assignment (frame.witness coordinate)))
    (squares :
      ∀ coordinate,
        assignment (frame.square coordinate) =
          assignment (frame.witness coordinate) *
            assignment (frame.witness coordinate))
    (sourceSatisfied :
      NativeCcsSelector.Satisfies program.rows
        (fun column =>
          assignment (FreshCcs.physicalColumn valid (ccsFrame frame) column)))
    (residuals :
      ∀ source,
        assignment (frame.residual source) =
          ActivatedRawProgram.residualValue
            (FreshCcs.mappedRow valid (ccsFrame frame)
              (program.rows.get source)).source.row assignment) :
    Satisfies (rows valid frame) assignment := by
  apply (satisfies_append_iff _ _ assignment).mpr
  refine ⟨Ajtai.rows_honest (ajtaiFrame frame) assignment constantOne
      commitmentMatches, ?_⟩
  apply (satisfies_append_iff _ _ assignment).mpr
  refine ⟨Projection.rows_honest (projectionFrame frame) assignment
      constantOne publicMatches, ?_⟩
  apply (satisfies_append_iff _ _ assignment).mpr
  exact ⟨
    Norm.rows_honest (normFrame frame) assignment distinct bounded squares,
    FreshCcs.rows_honest valid (ccsFrame frame) assignment
      sourceSatisfied residuals
  ⟩

/-- Exact local resource receipt for one fresh terminal relation. -/
def cost
    (program : NativeCcsProgram.Program)
    (shape : Phi81Relation.Shape)
    (verifierRows : Nat) : Cost :=
  ⟨verifierRows * ringDegree + shape.publicWidth +
      2 * shape.carrierWidth + 2 * program.rows.length,
    0, 0, shape.carrierWidth + program.rows.length⟩

@[simp] theorem cost_rows
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (valid : NativeCcsCompiler.Valid program)
    (frame : Frame program domain publicRingColumns publicFits verifierRows) :
    (rows valid frame).length =
      (cost program
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows).recurringRows := by
  simp [cost]

@[simp] theorem cost_auxiliary
    (program : NativeCcsProgram.Program)
    (shape : Phi81Relation.Shape)
    (verifierRows : Nat) :
    (cost program shape verifierRows).auxiliaryColumns =
      shape.carrierWidth + program.rows.length :=
  rfl

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Fresh
