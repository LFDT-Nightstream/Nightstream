import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Ajtai
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.FixedPointEvaluation
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Norm
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Projection

/-!
Contract: statement-specialized R1CS relation for one selected running
SuperNeo CE opening.

Assurance tier: model-level.

Owns: Ajtai commitment binding, public projection binding, strict `b = 2`
norm rows, exact evaluation rows derived from the verifier-owned relation and
public CE point, ordered composition, exact cost, soundness, and honest
completeness.

Does not own: the other thirteen running claims, the fresh CCS claim,
terminal control flow, input codecs, a deployment manifest, Rust, Spartan,
WHIR, Ajtai binding security, or authority for a prover-supplied relation or
evaluation point.

The row coefficients depend on `statement.constraintSystem` and
`statement.point`. A terminal verifier must reconstruct this program from its
authoritative public statement before Spartan setup. A proof must never
supply or select these coefficients.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Running

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Typed
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- One coherent physical placement for one running CE opening. -/
structure Frame
    (shape : Phi81Relation.Shape)
    (verifierRows : Nat) where
  owner : PhysicalOwner
  firstOrdinal : Nat
  one : ColumnId
  key : Commitment.Key shape verifierRows
  witness : Fin shape.carrierWidth → ColumnId
  commitment : Fin verifierRows → Fin ringDegree → ColumnId
  publicColumn : Fin shape.publicWidth → ColumnId
  evaluationLow :
    Fin shape.matrixCount → Fin ringDegree → ColumnId
  evaluationHigh :
    Fin shape.matrixCount → Fin ringDegree → ColumnId
  square : Fin shape.carrierWidth → ColumnId

def ajtaiFrame
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    (frame : Frame shape verifierRows) :
    Ajtai.Frame shape verifierRows where
  owner := frame.owner
  firstOrdinal := frame.firstOrdinal
  one := frame.one
  key := frame.key
  witness := frame.witness
  commitment := frame.commitment

def projectionFrame
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    (frame : Frame shape verifierRows) :
    Projection.Frame shape where
  owner := frame.owner
  firstOrdinal := frame.firstOrdinal + verifierRows * ringDegree
  one := frame.one
  witness := frame.witness
  publicColumn := frame.publicColumn

def normFrame
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    (frame : Frame shape verifierRows) :
    Norm.Frame shape where
  owner := frame.owner
  firstOrdinal :=
    frame.firstOrdinal + verifierRows * ringDegree + shape.publicWidth
  witness := frame.witness
  square := frame.square

def evaluationFrame
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    (frame : Frame shape verifierRows) :
    FixedPointEvaluation.Frame shape where
  owner := frame.owner
  firstOrdinal :=
    frame.firstOrdinal + verifierRows * ringDegree + shape.publicWidth +
      2 * shape.carrierWidth
  one := frame.one
  witness := frame.witness
  claimLow := frame.evaluationLow
  claimHigh := frame.evaluationHigh

/-- Ordered statement-specialized rows for one running CE opening. -/
def rows
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    (frame : Frame shape verifierRows)
    (statement :
      Phi81Relation.CEStatement shape (CommitmentValue verifierRows)) :
    List OwnedRow :=
  Ajtai.rows (ajtaiFrame frame) ++
    (Projection.rows (projectionFrame frame) ++
      (Norm.rows (normFrame frame) ++
        FixedPointEvaluation.rows (evaluationFrame frame)
          statement.constraintSystem statement.point))

/-- Exact auxiliary allocation. Only norm-square columns are new. -/
def columns
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    (frame : Frame shape verifierRows) :
    List OwnedColumn :=
  Norm.columns (normFrame frame)

/-- Every row in the composed running program keeps the frame owner. -/
theorem rows_owned
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    (frame : Frame shape verifierRows)
    (statement :
      Phi81Relation.CEStatement shape (CommitmentValue verifierRows))
    (owned : OwnedRow)
    (member : owned ∈ rows frame statement) :
    owned.id.owner = frame.owner := by
  simp only [rows, List.mem_append] at member
  rcases member with ajtaiMember |
      (projectionMember | (normMember | evaluationMember))
  · exact Ajtai.rows_owned (ajtaiFrame frame) owned ajtaiMember
  · exact Projection.rows_owned (projectionFrame frame) owned
      projectionMember
  · exact Norm.rows_owned (normFrame frame) owned normMember
  · exact
      FixedPointEvaluation.rows_owned (evaluationFrame frame)
        statement.constraintSystem statement.point owned evaluationMember

@[simp] theorem rows_length
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    (frame : Frame shape verifierRows)
    (statement :
      Phi81Relation.CEStatement shape (CommitmentValue verifierRows)) :
    (rows frame statement).length =
      verifierRows * ringDegree + shape.publicWidth +
        2 * shape.carrierWidth +
        2 * (shape.matrixCount * ringDegree) := by
  simp [rows]
  omega

@[simp] theorem columns_length
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    (frame : Frame shape verifierRows) :
    (columns frame).length = shape.carrierWidth := by
  simp [columns]

/-- Exact physical interpretation of the claimed evaluation columns. -/
structure EvaluationColumnsMatch
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    (frame : Frame shape verifierRows)
    (statement :
      Phi81Relation.CEStatement shape (CommitmentValue verifierRows))
    (assignment : ColumnId → F) : Prop where
  size_eq : statement.evaluations.size = shape.matrixCount
  lane_eq :
    ∀ (matrix : Fin shape.matrixCount) (lane : Fin ringDegree),
      K.mk
          (assignment (frame.evaluationLow matrix lane))
          (assignment (frame.evaluationHigh matrix lane)) =
        (statement.evaluations[matrix.val]'(by
          rw [size_eq]
          exact matrix.isLt)) lane

private theorem split_satisfaction
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    (frame : Frame shape verifierRows)
    (statement :
      Phi81Relation.CEStatement shape (CommitmentValue verifierRows))
    (assignment : ColumnId → F)
    (satisfied : Satisfies (rows frame statement) assignment) :
    Satisfies (Ajtai.rows (ajtaiFrame frame)) assignment ∧
      Satisfies (Projection.rows (projectionFrame frame)) assignment ∧
      Satisfies (Norm.rows (normFrame frame)) assignment ∧
      Satisfies
        (FixedPointEvaluation.rows (evaluationFrame frame)
          statement.constraintSystem statement.point)
        assignment := by
  have first :=
    (satisfies_append_iff
      (Ajtai.rows (ajtaiFrame frame))
      (Projection.rows (projectionFrame frame) ++
        Norm.rows (normFrame frame) ++
          FixedPointEvaluation.rows (evaluationFrame frame)
            statement.constraintSystem statement.point)
      assignment).mp (by simpa [rows] using satisfied)
  have firstTail :
      Satisfies
        (Projection.rows (projectionFrame frame) ++
          (Norm.rows (normFrame frame) ++
            FixedPointEvaluation.rows (evaluationFrame frame)
              statement.constraintSystem statement.point))
        assignment := by
    simpa only [List.append_assoc] using first.2
  have second :=
    (satisfies_append_iff
      (Projection.rows (projectionFrame frame))
      (Norm.rows (normFrame frame) ++
        FixedPointEvaluation.rows (evaluationFrame frame)
          statement.constraintSystem statement.point)
      assignment).mp firstTail
  have third :=
    (satisfies_append_iff
      (Norm.rows (normFrame frame))
      (FixedPointEvaluation.rows (evaluationFrame frame)
        statement.constraintSystem statement.point)
      assignment).mp second.2
  exact ⟨first.1, second.1, third.1, third.2⟩

/-- The physical rows force every semantic component of one running
statement. -/
theorem rows_facts
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (frame : Frame shape verifierRows)
    (statement :
      Phi81Relation.CEStatement shape (CommitmentValue verifierRows))
    (assignment : ColumnId → F)
    (constantOne : assignment frame.one = 1)
    (satisfied : Satisfies (rows frame statement) assignment) :
    (fun verifierRow output =>
        assignment (frame.commitment verifierRow output)) =
        Commitment.commit frame.key
          (fun coordinate => assignment (frame.witness coordinate)) ∧
      (fun coordinate => assignment (frame.publicColumn coordinate)) =
        Phi81Relation.projectPublicInput
          (fun coordinate => assignment (frame.witness coordinate)) ∧
      Phi81Relation.assignmentNormBounded 2
        (fun coordinate => assignment (frame.witness coordinate)) ∧
      (∀ matrix lane,
        K.mk
            (assignment (frame.evaluationLow matrix lane))
            (assignment (frame.evaluationHigh matrix lane)) =
          Phi81Relation.matrixEvaluation statement.constraintSystem
            (fun coordinate => assignment (frame.witness coordinate))
            statement.point matrix lane) := by
  rcases split_satisfaction frame statement assignment satisfied with
    ⟨ajtaiSatisfied, projectionSatisfied, normSatisfied,
      evaluationSatisfied⟩
  exact ⟨
    Ajtai.rows_sound (ajtaiFrame frame) assignment constantOne
      ajtaiSatisfied,
    Projection.rows_sound (projectionFrame frame) assignment constantOne
      projectionSatisfied,
    Norm.rows_sound noZeroDivisors (normFrame frame) assignment normSatisfied,
    FixedPointEvaluation.rows_sound (evaluationFrame frame)
      statement.constraintSystem statement.point assignment constantOne
      evaluationSatisfied
  ⟩

/-- Statement-specialized rows establish the exact paper `CE.Holds`
relation. The relation and point occur in the generated rows and must be
verifier-owned. -/
theorem rows_sound
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (frame : Frame shape verifierRows)
    (statement :
      Phi81Relation.CEStatement shape (CommitmentValue verifierRows))
    (assignment : ColumnId → F)
    (constantOne : assignment frame.one = 1)
    (freshStage :
      statement.stage.bound productionGlobalParams = 2)
    (commitmentColumns :
      (fun verifierRow output =>
        assignment (frame.commitment verifierRow output)) =
          statement.commitment)
    (publicColumns :
      (fun coordinate => assignment (frame.publicColumn coordinate)) =
        statement.publicInput)
    (evaluationColumns :
      EvaluationColumnsMatch frame statement assignment)
    (satisfied : Satisfies (rows frame statement) assignment) :
    CE.Holds
      (Phi81Relation.relationSemantics (Commitment.commit frame.key))
      productionGlobalParams statement
      (fun coordinate => assignment (frame.witness coordinate)) := by
  rcases rows_facts noZeroDivisors frame statement assignment constantOne
      satisfied with
    ⟨commitmentFact, publicFact, normFact, evaluationFact⟩
  apply
    (Phi81Relation.ceMembership_iff_evaluationsBound
      (Commitment.commit frame.key) productionGlobalParams statement
      (fun coordinate => assignment (frame.witness coordinate))).mpr
  refine ⟨?_, ?_, ?_, ?_⟩
  · exact commitmentFact.symm.trans commitmentColumns
  · unfold Phi81Relation.publicInputMatches
    exact publicFact.symm.trans publicColumns
  · simpa [freshStage] using normFact
  · refine {
      size_eq := evaluationColumns.size_eq
      lane_eq := ?_
    }
    intro matrix lane
    exact
      (evaluationColumns.lane_eq matrix lane).symm.trans
        (evaluationFact matrix lane)

/-- Honest statement columns and honest auxiliary squares satisfy the exact
statement-specialized program. -/
theorem rows_honest
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    (frame : Frame shape verifierRows)
    (statement :
      Phi81Relation.CEStatement shape (CommitmentValue verifierRows))
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
    (evaluations :
      ∀ matrix lane,
        K.mk
            (assignment (frame.evaluationLow matrix lane))
            (assignment (frame.evaluationHigh matrix lane)) =
          Phi81Relation.matrixEvaluation statement.constraintSystem
            (fun coordinate => assignment (frame.witness coordinate))
            statement.point matrix lane) :
    Satisfies (rows frame statement) assignment := by
  apply (satisfies_append_iff _ _ assignment).mpr
  refine ⟨Ajtai.rows_honest (ajtaiFrame frame) assignment constantOne
      commitmentMatches, ?_⟩
  apply (satisfies_append_iff _ _ assignment).mpr
  refine ⟨Projection.rows_honest (projectionFrame frame) assignment
      constantOne publicMatches, ?_⟩
  apply (satisfies_append_iff _ _ assignment).mpr
  exact ⟨
    Norm.rows_honest (normFrame frame) assignment distinct bounded squares,
    FixedPointEvaluation.rows_honest (evaluationFrame frame)
      statement.constraintSystem statement.point assignment constantOne
      evaluations
  ⟩

/-- Exact local receipt for one statement-specialized running relation. -/
def cost
    (shape : Phi81Relation.Shape)
    (verifierRows : Nat) : Cost :=
  ⟨verifierRows * ringDegree + shape.publicWidth +
      2 * shape.carrierWidth +
      2 * (shape.matrixCount * ringDegree),
    0, 0, shape.carrierWidth⟩

@[simp] theorem cost_rows
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    (frame : Frame shape verifierRows)
    (statement :
      Phi81Relation.CEStatement shape (CommitmentValue verifierRows)) :
    (rows frame statement).length =
      (cost shape verifierRows).recurringRows := by
  simp [cost]

@[simp] theorem cost_auxiliary
    (shape : Phi81Relation.Shape)
    (verifierRows : Nat) :
    (cost shape verifierRows).auxiliaryColumns =
      shape.carrierWidth :=
  rfl

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Running
