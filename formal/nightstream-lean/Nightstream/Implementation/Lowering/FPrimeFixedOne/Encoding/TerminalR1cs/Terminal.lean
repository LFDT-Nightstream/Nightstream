import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Fresh
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Running

/-!
Contract: one terminal R1CS program for the fourteen running CE claims and
one fresh CCS claim required by the selected fixed-one HyperNova verifier.

Assurance tier: model-level.

Owns: canonical child order, exact program concatenation, exact resource
counts, satisfaction decomposition, and composition of the fifteen local
soundness results.

Does not own: a concrete physical column layout, the selected benchmark
statement, input codecs, a deployment manifest, Spartan, WHIR, Rust, or
Ajtai binding security.

The running rows remain statement-specialized. A verifier must construct
them from its authoritative terminal statements before Spartan setup.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Terminal

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Typed
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
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

/-- The fifteen local physical frames in canonical terminal order. -/
structure Frame
    (program : NativeCcsProgram.Program)
    (domain : NativeCcsCompiler.RowDomain program)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length)
    (verifierRows : Nat) where
  running :
    Fin productionGlobalParams.k →
      Running.Frame
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows
  fresh :
    Fresh.Frame program domain publicRingColumns publicFits verifierRows

abbrev RunningStatements
    (program : NativeCcsProgram.Program)
    (domain : NativeCcsCompiler.RowDomain program)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length)
    (verifierRows : Nat) :=
  Fin productionGlobalParams.k →
    Phi81Relation.CEStatement
      (RelationShape program domain publicRingColumns publicFits)
      (CommitmentValue verifierRows)

/-- The fourteen running programs in increasing child index order. -/
def runningRows
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (frame : Frame program domain publicRingColumns publicFits verifierRows)
    (statements :
      RunningStatements program domain publicRingColumns publicFits
        verifierRows) :
    List OwnedRow :=
  (List.finRange productionGlobalParams.k).flatMap fun child =>
    Running.rows (frame.running child) (statements child)

/-- The complete terminal program: fourteen running checks, then fresh. -/
def rows
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (valid : NativeCcsCompiler.Valid program)
    (frame : Frame program domain publicRingColumns publicFits verifierRows)
    (statements :
      RunningStatements program domain publicRingColumns publicFits
        verifierRows) :
    List OwnedRow :=
  runningRows frame statements ++ Fresh.rows valid frame.fresh

/-- Exact terminal auxiliary allocation in the same child-first order. -/
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
  (List.finRange productionGlobalParams.k).flatMap
      (fun child => Running.columns (frame.running child)) ++
    Fresh.columns frame.fresh

private theorem satisfies_flatMap_iff
    {Index : Type}
    (indices : List Index)
    (piece : Index → List OwnedRow)
    (assignment : ColumnId → F) :
    Satisfies (indices.flatMap piece) assignment ↔
      ∀ index ∈ indices, Satisfies (piece index) assignment := by
  induction indices with
  | nil =>
      simp
  | cons head tail inductionHypothesis =>
      rw [List.flatMap_cons, satisfies_append_iff,
        inductionHypothesis]
      simp only [List.mem_cons]
      constructor
      · rintro ⟨headSatisfied, tailSatisfied⟩ index (rfl | member)
        · exact headSatisfied
        · exact tailSatisfied index member
      · intro all
        exact ⟨all head (Or.inl rfl),
          fun index member => all index (Or.inr member)⟩

theorem running_satisfies_iff
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (frame : Frame program domain publicRingColumns publicFits verifierRows)
    (statements :
      RunningStatements program domain publicRingColumns publicFits
        verifierRows)
    (assignment : ColumnId → F) :
    Satisfies (runningRows frame statements) assignment ↔
      ∀ child,
        Satisfies
          (Running.rows (frame.running child) (statements child))
          assignment := by
  rw [runningRows, satisfies_flatMap_iff]
  constructor
  · intro all child
    exact all child (List.mem_finRange child)
  · intro all child _member
    exact all child

theorem satisfies_iff
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (valid : NativeCcsCompiler.Valid program)
    (frame : Frame program domain publicRingColumns publicFits verifierRows)
    (statements :
      RunningStatements program domain publicRingColumns publicFits
        verifierRows)
    (assignment : ColumnId → F) :
    Satisfies (rows valid frame statements) assignment ↔
      (∀ child,
        Satisfies
          (Running.rows (frame.running child) (statements child))
          assignment) ∧
      Satisfies (Fresh.rows valid frame.fresh) assignment := by
  rw [rows, satisfies_append_iff, running_satisfies_iff]

@[simp] theorem runningRows_length
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (frame : Frame program domain publicRingColumns publicFits verifierRows)
    (statements :
      RunningStatements program domain publicRingColumns publicFits
        verifierRows) :
    (runningRows frame statements).length =
      productionGlobalParams.k *
        (verifierRows * ringDegree +
          (RelationShape program domain publicRingColumns publicFits).publicWidth +
          2 *
            (RelationShape program domain publicRingColumns publicFits).carrierWidth +
          2 *
            ((RelationShape program domain publicRingColumns publicFits).matrixCount *
              ringDegree)) := by
  simp [runningRows, Running.rows_length, List.map_const']
  omega

@[simp] theorem rows_length
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (valid : NativeCcsCompiler.Valid program)
    (frame : Frame program domain publicRingColumns publicFits verifierRows)
    (statements :
      RunningStatements program domain publicRingColumns publicFits
        verifierRows) :
    (rows valid frame statements).length =
      productionGlobalParams.k *
          (verifierRows * ringDegree +
            (RelationShape program domain publicRingColumns publicFits).publicWidth +
            2 *
              (RelationShape program domain publicRingColumns publicFits).carrierWidth +
            2 *
              ((RelationShape program domain publicRingColumns publicFits).matrixCount *
                ringDegree)) +
        (verifierRows * ringDegree +
          (RelationShape program domain publicRingColumns publicFits).publicWidth +
          2 *
            (RelationShape program domain publicRingColumns publicFits).carrierWidth +
          2 * program.rows.length) := by
  simp [rows]

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
      productionGlobalParams.k *
          (RelationShape program domain publicRingColumns publicFits).carrierWidth +
        ((RelationShape program domain publicRingColumns publicFits).carrierWidth +
          program.rows.length) := by
  simp [columns, Running.columns_length, Fresh.columns_length,
    List.map_const']
  omega

/-- Exact terminal receipt. Statement and witness columns are inputs. -/
def cost
    (program : NativeCcsProgram.Program)
    (shape : Phi81Relation.Shape)
    (verifierRows : Nat) : Cost :=
  ⟨productionGlobalParams.k *
        (verifierRows * ringDegree + shape.publicWidth +
          2 * shape.carrierWidth +
          2 * (shape.matrixCount * ringDegree)) +
      (verifierRows * ringDegree + shape.publicWidth +
        2 * shape.carrierWidth + 2 * program.rows.length),
    0, 0,
    productionGlobalParams.k * shape.carrierWidth +
      (shape.carrierWidth + program.rows.length)⟩

@[simp] theorem cost_rows
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (valid : NativeCcsCompiler.Valid program)
    (frame : Frame program domain publicRingColumns publicFits verifierRows)
    (statements :
      RunningStatements program domain publicRingColumns publicFits
        verifierRows) :
    (rows valid frame statements).length =
      (cost program
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows).recurringRows := by
  simp [cost]

@[simp] theorem cost_auxiliary
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    (verifierRows : Nat) :
    (cost program
      (RelationShape program domain publicRingColumns publicFits)
      verifierRows).auxiliaryColumns =
        productionGlobalParams.k *
            (RelationShape program domain publicRingColumns publicFits).carrierWidth +
          ((RelationShape program domain publicRingColumns publicFits).carrierWidth +
            program.rows.length) :=
  rfl

/-- Composition of all local soundness results. The result is exactly the
paper CE product and fresh CCS relation for the compiled native structure. -/
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
    (statements :
      RunningStatements program domain publicRingColumns publicFits
        verifierRows)
    (freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows)
    (assignment : ColumnId → F)
    (runningConstantOne :
      ∀ child, assignment (frame.running child).one = 1)
    (runningFreshStage :
      ∀ child,
        (statements child).stage.bound productionGlobalParams = 2)
    (runningCommitmentColumns :
      ∀ child,
        (fun verifierRow output =>
          assignment ((frame.running child).commitment verifierRow output)) =
            (statements child).commitment)
    (runningPublicColumns :
      ∀ child,
        (fun coordinate =>
          assignment ((frame.running child).publicColumn coordinate)) =
            (statements child).publicInput)
    (runningEvaluationColumns :
      ∀ child,
        Running.EvaluationColumnsMatch
          (frame.running child) (statements child) assignment)
    (freshConstantOne : assignment frame.fresh.one = 1)
    (freshCommitmentColumns :
      (fun verifierRow output =>
        assignment (frame.fresh.commitment verifierRow output)) =
          freshPayload.commitment)
    (freshPublicColumns :
      (fun coordinate =>
        assignment (frame.fresh.publicColumn coordinate)) =
          freshPayload.publicInput)
    (satisfied : Satisfies (rows valid frame statements) assignment) :
    (∀ child,
      CE.Holds
        (Phi81Relation.relationSemantics
          (Commitment.commit (frame.running child).key))
        productionGlobalParams (statements child)
        (fun coordinate =>
          assignment ((frame.running child).witness coordinate))) ∧
      CCS.Holds
        (Phi81Relation.relationSemantics
          (Commitment.commit frame.fresh.key))
        productionGlobalParams
        (freshPayload.materialize
          (NativeCcsPhi81.relation program valid domain
            publicRingColumns publicFits))
        (fun coordinate => assignment (frame.fresh.witness coordinate)) := by
  rcases
      (satisfies_iff valid frame statements assignment).mp satisfied with
    ⟨runningSatisfied, freshSatisfied⟩
  constructor
  · intro child
    exact
      Running.rows_sound noZeroDivisors
        (frame.running child) (statements child) assignment
        (runningConstantOne child) (runningFreshStage child)
        (runningCommitmentColumns child)
        (runningPublicColumns child)
        (runningEvaluationColumns child)
        (runningSatisfied child)
  · exact
      Fresh.rows_sound noZeroDivisors valid frame.fresh freshPayload
        assignment freshConstantOne freshCommitmentColumns
        freshPublicColumns freshSatisfied

/-- Local honest executions compose into one terminal assignment. -/
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
    (statements :
      RunningStatements program domain publicRingColumns publicFits
        verifierRows)
    (assignment : ColumnId → F)
    (runningHonest :
      ∀ child,
        Satisfies
          (Running.rows (frame.running child) (statements child))
          assignment)
    (freshHonest :
      Satisfies (Fresh.rows valid frame.fresh) assignment) :
    Satisfies (rows valid frame statements) assignment :=
  (satisfies_iff valid frame statements assignment).mpr
    ⟨runningHonest, freshHonest⟩

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Terminal
