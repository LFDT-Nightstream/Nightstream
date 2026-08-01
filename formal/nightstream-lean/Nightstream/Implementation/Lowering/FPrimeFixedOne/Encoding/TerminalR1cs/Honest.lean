import Mathlib.Data.List.GetD
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Scoping

/-!
Contract: honest physical assignment for the selected SuperNeo terminal R1CS.

Assurance tier: model-level.

Owns: construction of every running and fresh input, statement, square, and
native-CCS residual column from exact paper witnesses; preservation of those
values during residual completion; and honest satisfaction of the proof-free
terminal manifest.

Does not own: selection of a benchmark statement, terminal semantic
soundness, Spartan, WHIR, Rust, or Ajtai binding security.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Honest

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Typed
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
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

/-- Exact private openings for the authoritative terminal statements. -/
structure Input
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (valid : NativeCcsCompiler.Valid program)
    (key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows)
    (statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows)
    (freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows) where
  runningWitness :
    Fin productionGlobalParams.k →
      Phi81Relation.Assignment
        (RelationShape program domain publicRingColumns publicFits)
  freshWitness :
    Phi81Relation.Assignment
      (RelationShape program domain publicRingColumns publicFits)
  runningFreshStage :
    ∀ child,
      (statements child).stage.bound productionGlobalParams = 2
  runningHolds :
    ∀ child,
      CE.Holds
        (Phi81Relation.relationSemantics (Commitment.commit key))
        productionGlobalParams (statements child) (runningWitness child)
  freshHolds :
    CCS.Holds
      (Phi81Relation.relationSemantics (Commitment.commit key))
      productionGlobalParams
      (freshPayload.materialize
        (NativeCcsPhi81.relation program valid domain
          publicRingColumns publicFits))
      freshWitness

private theorem getD_ofFn
    {Item : Type}
    {count : Nat}
    (items : Fin count → Item)
    (index : Fin count)
    (default : Item) :
    (List.ofFn items).getD index.val default = items index := by
  rw [List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem (by simp), List.getElem_ofFn]
  simp

private theorem getD_append_length_add
    {Item : Type}
    (head tail : List Item)
    (index : Nat)
    (default : Item) :
    (head ++ tail).getD (head.length + index) default =
      tail.getD index default := by
  rw [List.getD_append_right _ _ _ _ (by omega)]
  congr
  omega

/-- Decode the unary owner path used by the terminal layout. Other branch
paths are not terminal-claim owners. -/
def claimIndex? : OwnerPath → Option Nat
  | .root => some 0
  | .rest parent => (claimIndex? parent).map Nat.succ
  | .trueArm _ => none
  | .falseArm _ => none
  | .continuation _ => none

@[simp] theorem claimIndex_claimPath (index : Nat) :
    claimIndex? (Layout.claimPath index) = some index := by
  induction index with
  | zero =>
      rfl
  | succ index inductionHypothesis =>
      simp [Layout.claimPath, claimIndex?, inductionHypothesis]

private def pairValues
    (count : Nat)
    (value : Fin count → Fin ringDegree → F) : List F :=
  List.ofFn fun position : Fin (count * ringDegree) =>
    value (Ajtai.verifierRowAt position) (Ajtai.outputAt position)

@[simp] private theorem pairValues_length
    (count : Nat)
    (value : Fin count → Fin ringDegree → F) :
    (pairValues count value).length = count * ringDegree := by
  simp [pairValues]

private def runningWitnessValues
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Input valid key statements freshPayload)
    (child : Fin productionGlobalParams.k) : List F :=
  List.ofFn (input.runningWitness child)

private def runningCommitmentValues
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Input valid key statements freshPayload)
    (child : Fin productionGlobalParams.k) : List F :=
  pairValues verifierRows
    (Commitment.commit key (input.runningWitness child))

private def runningPublicValues
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Input valid key statements freshPayload)
    (child : Fin productionGlobalParams.k) : List F :=
  List.ofFn
    (Phi81Relation.projectPublicInput (input.runningWitness child))

private def runningEvaluationLowValues
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Input valid key statements freshPayload)
    (child : Fin productionGlobalParams.k) : List F :=
  pairValues
    (RelationShape program domain publicRingColumns publicFits).matrixCount
    fun matrix lane =>
      (Phi81Relation.matrixEvaluation
        (statements child).constraintSystem
        (input.runningWitness child) (statements child).point matrix lane).c0

private def runningEvaluationHighValues
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Input valid key statements freshPayload)
    (child : Fin productionGlobalParams.k) : List F :=
  pairValues
    (RelationShape program domain publicRingColumns publicFits).matrixCount
    fun matrix lane =>
      (Phi81Relation.matrixEvaluation
        (statements child).constraintSystem
        (input.runningWitness child) (statements child).point matrix lane).c1

private def runningSquareValues
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Input valid key statements freshPayload)
    (child : Fin productionGlobalParams.k) : List F :=
  List.ofFn fun coordinate =>
    input.runningWitness child coordinate *
      input.runningWitness child coordinate

private def runningValues
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Input valid key statements freshPayload)
    (child : Fin productionGlobalParams.k) : List F :=
  runningWitnessValues input child ++
    (runningCommitmentValues input child ++
      (runningPublicValues input child ++
        (runningEvaluationLowValues input child ++
          (runningEvaluationHighValues input child ++
            runningSquareValues input child))))

@[simp] theorem runningValues_length
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Input valid key statements freshPayload)
    (child : Fin productionGlobalParams.k) :
    (runningValues input child).length =
      Layout.runningWidth
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows := by
  simp [runningValues, runningWitnessValues, runningCommitmentValues,
    runningPublicValues, runningEvaluationLowValues,
    runningEvaluationHighValues, runningSquareValues,
    Layout.runningWidth, Layout.runningInputWidth,
    Layout.runningStatementWidth]
  omega

private def freshWitnessValues
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Input valid key statements freshPayload) : List F :=
  List.ofFn input.freshWitness

private def freshCommitmentValues
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Input valid key statements freshPayload) : List F :=
  pairValues verifierRows (Commitment.commit key input.freshWitness)

private def freshPublicValues
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Input valid key statements freshPayload) : List F :=
  List.ofFn (Phi81Relation.projectPublicInput input.freshWitness)

private def freshSquareValues
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Input valid key statements freshPayload) : List F :=
  List.ofFn fun coordinate =>
    input.freshWitness coordinate * input.freshWitness coordinate

private def freshBaseValues
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Input valid key statements freshPayload) : List F :=
  freshWitnessValues input ++
    (freshCommitmentValues input ++
      (freshPublicValues input ++
        (freshSquareValues input ++ List.replicate program.rows.length 0)))

@[simp] theorem freshBaseValues_length
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Input valid key statements freshPayload) :
    (freshBaseValues input).length =
      Layout.freshWidth program
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows := by
  simp [freshBaseValues, freshWitnessValues, freshCommitmentValues,
    freshPublicValues, freshSquareValues, Layout.freshWidth,
    Layout.freshInputWidth, Layout.freshStatementWidth]
  omega

/-- Assignment before native-CCS residual columns are filled. -/
def baseAssignment
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Input valid key statements freshPayload)
    (column : ColumnId) : F :=
  match column.owner with
  | .prelude => if column = oneColumn then 1 else 0
  | .typed (.instruction path) =>
      match claimIndex? path with
      | none => 0
      | some index =>
          if running : index < productionGlobalParams.k then
            (runningValues input ⟨index, running⟩).getD
              column.coordinateIndex 0
          else if _fresh : index = productionGlobalParams.k then
            (freshBaseValues input).getD column.coordinateIndex 0
          else 0
  | _ => 0

@[simp] theorem baseAssignment_one
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Input valid key statements freshPayload) :
    baseAssignment input oneColumn = 1 := by
  simp [baseAssignment, oneColumn]

@[simp] theorem baseAssignment_runningLocal
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Input valid key statements freshPayload)
    (child : Fin productionGlobalParams.k)
    (coordinate : Nat) :
    baseAssignment input
        (Layout.localColumn (Layout.runningOwner child) coordinate) =
      (runningValues input child).getD coordinate 0 := by
  simp [baseAssignment, Layout.localColumn, Layout.runningOwner,
    claimIndex_claimPath, child.isLt]

@[simp] theorem baseAssignment_freshLocal
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Input valid key statements freshPayload)
    (coordinate : Nat) :
    baseAssignment input (Layout.localColumn Layout.freshOwner coordinate) =
      (freshBaseValues input).getD coordinate 0 := by
  simp [baseAssignment, Layout.localColumn, Layout.freshOwner,
    claimIndex_claimPath]

private theorem runningValues_witness
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Input valid key statements freshPayload)
    (child : Fin productionGlobalParams.k)
    (coordinate :
      Fin (RelationShape program domain publicRingColumns publicFits).carrierWidth) :
    (runningValues input child).getD coordinate.val 0 =
      input.runningWitness child coordinate := by
  rw [runningValues, List.getD_append _ _ _ _ (by
    simp [runningWitnessValues])]
  exact getD_ofFn _ coordinate 0

private theorem runningValues_commitment
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Input valid key statements freshPayload)
    (child : Fin productionGlobalParams.k)
    (verifierRow : Fin verifierRows)
    (output : Fin ringDegree) :
    (runningValues input child).getD
        ((RelationShape program domain publicRingColumns publicFits).carrierWidth +
          verifierRow.val * ringDegree + output.val) 0 =
      Commitment.commit key (input.runningWitness child)
        verifierRow output := by
  have indexEq :
      (RelationShape program domain publicRingColumns publicFits).carrierWidth +
          verifierRow.val * ringDegree + output.val =
        (runningWitnessValues input child).length +
          (Ajtai.pairIndex verifierRow output).val := by
    simp [runningWitnessValues, Ajtai.pairIndex]
    omega
  rw [indexEq, runningValues, getD_append_length_add]
  rw [List.getD_append _ _ _ _ (by
    simpa [runningCommitmentValues] using
      (Ajtai.pairIndex verifierRow output).isLt)]
  rw [runningCommitmentValues, pairValues, getD_ofFn]
  simp

private theorem runningValues_public
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Input valid key statements freshPayload)
    (child : Fin productionGlobalParams.k)
    (coordinate :
      Fin (RelationShape program domain publicRingColumns publicFits).publicWidth) :
    (runningValues input child).getD
        ((RelationShape program domain publicRingColumns publicFits).carrierWidth +
          verifierRows * ringDegree + coordinate.val) 0 =
      Phi81Relation.projectPublicInput
        (input.runningWitness child) coordinate := by
  have indexEq :
      (RelationShape program domain publicRingColumns publicFits).carrierWidth +
          verifierRows * ringDegree + coordinate.val =
        (runningWitnessValues input child).length +
          ((runningCommitmentValues input child).length + coordinate.val) := by
    simp [runningWitnessValues, runningCommitmentValues]
    omega
  rw [indexEq, runningValues, getD_append_length_add,
    getD_append_length_add]
  rw [List.getD_append _ _ _ _ (by
    simp [runningPublicValues])]
  exact getD_ofFn _ coordinate 0

private theorem runningValues_evaluationLow
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Input valid key statements freshPayload)
    (child : Fin productionGlobalParams.k)
    (matrix :
      Fin (RelationShape program domain publicRingColumns publicFits).matrixCount)
    (lane : Fin ringDegree) :
    (runningValues input child).getD
        ((RelationShape program domain publicRingColumns publicFits).carrierWidth +
          verifierRows * ringDegree +
          (RelationShape program domain publicRingColumns publicFits).publicWidth +
          matrix.val * ringDegree + lane.val) 0 =
      (Phi81Relation.matrixEvaluation
        (statements child).constraintSystem
        (input.runningWitness child) (statements child).point matrix lane).c0 := by
  have indexEq :
      (RelationShape program domain publicRingColumns publicFits).carrierWidth +
            verifierRows * ringDegree +
            (RelationShape program domain publicRingColumns publicFits).publicWidth +
          matrix.val * ringDegree + lane.val =
        (runningWitnessValues input child).length +
          ((runningCommitmentValues input child).length +
            ((runningPublicValues input child).length +
              (Ajtai.pairIndex matrix lane).val)) := by
    simp [runningWitnessValues, runningCommitmentValues,
      runningPublicValues, Ajtai.pairIndex]
    omega
  rw [indexEq, runningValues, getD_append_length_add,
    getD_append_length_add, getD_append_length_add]
  rw [List.getD_append _ _ _ _ (by
    simpa [runningEvaluationLowValues] using
      (Ajtai.pairIndex matrix lane).isLt)]
  rw [runningEvaluationLowValues, pairValues, getD_ofFn]
  simp

private theorem runningValues_evaluationHigh
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Input valid key statements freshPayload)
    (child : Fin productionGlobalParams.k)
    (matrix :
      Fin (RelationShape program domain publicRingColumns publicFits).matrixCount)
    (lane : Fin ringDegree) :
    (runningValues input child).getD
        ((RelationShape program domain publicRingColumns publicFits).carrierWidth +
          verifierRows * ringDegree +
          (RelationShape program domain publicRingColumns publicFits).publicWidth +
          (RelationShape program domain publicRingColumns publicFits).matrixCount *
            ringDegree +
          matrix.val * ringDegree + lane.val) 0 =
      (Phi81Relation.matrixEvaluation
        (statements child).constraintSystem
        (input.runningWitness child) (statements child).point matrix lane).c1 := by
  have indexEq :
      (RelationShape program domain publicRingColumns publicFits).carrierWidth +
              verifierRows * ringDegree +
              (RelationShape program domain publicRingColumns publicFits).publicWidth +
            (RelationShape program domain publicRingColumns publicFits).matrixCount *
              ringDegree +
          matrix.val * ringDegree + lane.val =
        (runningWitnessValues input child).length +
          ((runningCommitmentValues input child).length +
            ((runningPublicValues input child).length +
              ((runningEvaluationLowValues input child).length +
                (Ajtai.pairIndex matrix lane).val))) := by
    simp [runningWitnessValues, runningCommitmentValues,
      runningPublicValues, runningEvaluationLowValues,
      Ajtai.pairIndex]
    omega
  rw [indexEq, runningValues, getD_append_length_add,
    getD_append_length_add, getD_append_length_add,
    getD_append_length_add]
  rw [List.getD_append _ _ _ _ (by
    simpa [runningEvaluationHighValues] using
      (Ajtai.pairIndex matrix lane).isLt)]
  rw [runningEvaluationHighValues, pairValues, getD_ofFn]
  simp

private theorem runningValues_square
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Input valid key statements freshPayload)
    (child : Fin productionGlobalParams.k)
    (coordinate :
      Fin (RelationShape program domain publicRingColumns publicFits).carrierWidth) :
    (runningValues input child).getD
        (Layout.runningInputWidth
          (RelationShape program domain publicRingColumns publicFits)
          verifierRows + coordinate.val) 0 =
      input.runningWitness child coordinate *
        input.runningWitness child coordinate := by
  have indexEq :
      Layout.runningInputWidth
            (RelationShape program domain publicRingColumns publicFits)
            verifierRows + coordinate.val =
        (runningWitnessValues input child).length +
          ((runningCommitmentValues input child).length +
            ((runningPublicValues input child).length +
              ((runningEvaluationLowValues input child).length +
                ((runningEvaluationHighValues input child).length +
                  coordinate.val)))) := by
    simp [runningWitnessValues, runningCommitmentValues,
      runningPublicValues, runningEvaluationLowValues,
      runningEvaluationHighValues, Layout.runningInputWidth,
      Layout.runningStatementWidth]
    omega
  rw [indexEq, runningValues, getD_append_length_add,
    getD_append_length_add, getD_append_length_add,
    getD_append_length_add, getD_append_length_add]
  exact getD_ofFn _ coordinate 0

private theorem freshBaseValues_witness
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Input valid key statements freshPayload)
    (coordinate :
      Fin (RelationShape program domain publicRingColumns publicFits).carrierWidth) :
    (freshBaseValues input).getD coordinate.val 0 =
      input.freshWitness coordinate := by
  rw [freshBaseValues, List.getD_append _ _ _ _ (by
    simp [freshWitnessValues])]
  exact getD_ofFn _ coordinate 0

private theorem freshBaseValues_commitment
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Input valid key statements freshPayload)
    (verifierRow : Fin verifierRows)
    (output : Fin ringDegree) :
    (freshBaseValues input).getD
        ((RelationShape program domain publicRingColumns publicFits).carrierWidth +
          verifierRow.val * ringDegree + output.val) 0 =
      Commitment.commit key input.freshWitness verifierRow output := by
  have indexEq :
      (RelationShape program domain publicRingColumns publicFits).carrierWidth +
          verifierRow.val * ringDegree + output.val =
        (freshWitnessValues input).length +
          (Ajtai.pairIndex verifierRow output).val := by
    simp [freshWitnessValues, Ajtai.pairIndex]
    omega
  rw [indexEq, freshBaseValues, getD_append_length_add]
  rw [List.getD_append _ _ _ _ (by
    simpa [freshCommitmentValues] using
      (Ajtai.pairIndex verifierRow output).isLt)]
  rw [freshCommitmentValues, pairValues, getD_ofFn]
  simp

private theorem freshBaseValues_public
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Input valid key statements freshPayload)
    (coordinate :
      Fin (RelationShape program domain publicRingColumns publicFits).publicWidth) :
    (freshBaseValues input).getD
        ((RelationShape program domain publicRingColumns publicFits).carrierWidth +
          verifierRows * ringDegree + coordinate.val) 0 =
      Phi81Relation.projectPublicInput input.freshWitness coordinate := by
  have indexEq :
      (RelationShape program domain publicRingColumns publicFits).carrierWidth +
          verifierRows * ringDegree + coordinate.val =
        (freshWitnessValues input).length +
          ((freshCommitmentValues input).length + coordinate.val) := by
    simp [freshWitnessValues, freshCommitmentValues]
    omega
  rw [indexEq, freshBaseValues, getD_append_length_add,
    getD_append_length_add]
  rw [List.getD_append _ _ _ _ (by
    simp [freshPublicValues])]
  exact getD_ofFn _ coordinate 0

private theorem freshBaseValues_square
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Input valid key statements freshPayload)
    (coordinate :
      Fin (RelationShape program domain publicRingColumns publicFits).carrierWidth) :
    (freshBaseValues input).getD
        (Layout.freshInputWidth
          (RelationShape program domain publicRingColumns publicFits)
          verifierRows + coordinate.val) 0 =
      input.freshWitness coordinate * input.freshWitness coordinate := by
  have indexEq :
      Layout.freshInputWidth
            (RelationShape program domain publicRingColumns publicFits)
            verifierRows + coordinate.val =
        (freshWitnessValues input).length +
          ((freshCommitmentValues input).length +
            ((freshPublicValues input).length + coordinate.val)) := by
    simp [freshWitnessValues, freshCommitmentValues, freshPublicValues,
      Layout.freshInputWidth, Layout.freshStatementWidth]
    omega
  rw [indexEq, freshBaseValues, getD_append_length_add,
    getD_append_length_add, getD_append_length_add]
  rw [List.getD_append _ _ _ _ (by
    simp [freshSquareValues])]
  exact getD_ofFn _ coordinate 0

@[simp] theorem baseAssignment_runningWitness
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {valid : NativeCcsCompiler.Valid program}
    {key : Commitment.Key
      (RelationShape program domain publicRingColumns publicFits) verifierRows}
    {statements : Terminal.RunningStatements program domain
      publicRingColumns publicFits verifierRows}
    {freshPayload : FixedActive.Canonical.FreshPayload
      (Fresh.semanticShape program domain) publicRingColumns publicFits
      verifierRows}
    (input : Input valid key statements freshPayload)
    (child : Fin productionGlobalParams.k)
    (coordinate :
      Fin (RelationShape program domain publicRingColumns publicFits).carrierWidth) :
    baseAssignment input ((Layout.runningFrame key child).witness coordinate) =
      input.runningWitness child coordinate := by
  rw [show (Layout.runningFrame key child).witness coordinate =
      Layout.localColumn (Layout.runningOwner child) coordinate.val by
    rfl]
  exact (baseAssignment_runningLocal input child coordinate.val).trans
    (runningValues_witness input child coordinate)

@[simp] theorem baseAssignment_runningCommitment
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {valid : NativeCcsCompiler.Valid program}
    {key : Commitment.Key
      (RelationShape program domain publicRingColumns publicFits) verifierRows}
    {statements : Terminal.RunningStatements program domain
      publicRingColumns publicFits verifierRows}
    {freshPayload : FixedActive.Canonical.FreshPayload
      (Fresh.semanticShape program domain) publicRingColumns publicFits
      verifierRows}
    (input : Input valid key statements freshPayload)
    (child : Fin productionGlobalParams.k)
    (verifierRow : Fin verifierRows)
    (output : Fin ringDegree) :
    baseAssignment input
        ((Layout.runningFrame key child).commitment verifierRow output) =
      Commitment.commit key (input.runningWitness child)
        verifierRow output := by
  rw [show (Layout.runningFrame key child).commitment verifierRow output =
      Layout.localColumn (Layout.runningOwner child)
        ((RelationShape program domain publicRingColumns publicFits).carrierWidth +
          verifierRow.val * ringDegree + output.val) by rfl]
  exact (baseAssignment_runningLocal input child _).trans
    (runningValues_commitment input child verifierRow output)

@[simp] theorem baseAssignment_runningPublic
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {valid : NativeCcsCompiler.Valid program}
    {key : Commitment.Key
      (RelationShape program domain publicRingColumns publicFits) verifierRows}
    {statements : Terminal.RunningStatements program domain
      publicRingColumns publicFits verifierRows}
    {freshPayload : FixedActive.Canonical.FreshPayload
      (Fresh.semanticShape program domain) publicRingColumns publicFits
      verifierRows}
    (input : Input valid key statements freshPayload)
    (child : Fin productionGlobalParams.k)
    (coordinate :
      Fin (RelationShape program domain publicRingColumns publicFits).publicWidth) :
    baseAssignment input
        ((Layout.runningFrame key child).publicColumn coordinate) =
      Phi81Relation.projectPublicInput
        (input.runningWitness child) coordinate := by
  rw [show (Layout.runningFrame key child).publicColumn coordinate =
      Layout.localColumn (Layout.runningOwner child)
        ((RelationShape program domain publicRingColumns publicFits).carrierWidth +
          verifierRows * ringDegree + coordinate.val) by rfl]
  exact (baseAssignment_runningLocal input child _).trans
    (runningValues_public input child coordinate)

@[simp] theorem baseAssignment_runningEvaluationLow
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {valid : NativeCcsCompiler.Valid program}
    {key : Commitment.Key
      (RelationShape program domain publicRingColumns publicFits) verifierRows}
    {statements : Terminal.RunningStatements program domain
      publicRingColumns publicFits verifierRows}
    {freshPayload : FixedActive.Canonical.FreshPayload
      (Fresh.semanticShape program domain) publicRingColumns publicFits
      verifierRows}
    (input : Input valid key statements freshPayload)
    (child : Fin productionGlobalParams.k)
    (matrix :
      Fin (RelationShape program domain publicRingColumns publicFits).matrixCount)
    (lane : Fin ringDegree) :
    baseAssignment input
        ((Layout.runningFrame key child).evaluationLow matrix lane) =
      (Phi81Relation.matrixEvaluation
        (statements child).constraintSystem
        (input.runningWitness child) (statements child).point matrix lane).c0 := by
  rw [show (Layout.runningFrame key child).evaluationLow matrix lane =
      Layout.localColumn (Layout.runningOwner child)
        ((RelationShape program domain publicRingColumns publicFits).carrierWidth +
          verifierRows * ringDegree +
          (RelationShape program domain publicRingColumns publicFits).publicWidth +
          matrix.val * ringDegree + lane.val) by rfl]
  exact (baseAssignment_runningLocal input child _).trans
    (runningValues_evaluationLow input child matrix lane)

@[simp] theorem baseAssignment_runningEvaluationHigh
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {valid : NativeCcsCompiler.Valid program}
    {key : Commitment.Key
      (RelationShape program domain publicRingColumns publicFits) verifierRows}
    {statements : Terminal.RunningStatements program domain
      publicRingColumns publicFits verifierRows}
    {freshPayload : FixedActive.Canonical.FreshPayload
      (Fresh.semanticShape program domain) publicRingColumns publicFits
      verifierRows}
    (input : Input valid key statements freshPayload)
    (child : Fin productionGlobalParams.k)
    (matrix :
      Fin (RelationShape program domain publicRingColumns publicFits).matrixCount)
    (lane : Fin ringDegree) :
    baseAssignment input
        ((Layout.runningFrame key child).evaluationHigh matrix lane) =
      (Phi81Relation.matrixEvaluation
        (statements child).constraintSystem
        (input.runningWitness child) (statements child).point matrix lane).c1 := by
  rw [show (Layout.runningFrame key child).evaluationHigh matrix lane =
      Layout.localColumn (Layout.runningOwner child)
        ((RelationShape program domain publicRingColumns publicFits).carrierWidth +
          verifierRows * ringDegree +
          (RelationShape program domain publicRingColumns publicFits).publicWidth +
          (RelationShape program domain publicRingColumns publicFits).matrixCount *
            ringDegree + matrix.val * ringDegree + lane.val) by rfl]
  exact (baseAssignment_runningLocal input child _).trans
    (runningValues_evaluationHigh input child matrix lane)

@[simp] theorem baseAssignment_runningSquare
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {valid : NativeCcsCompiler.Valid program}
    {key : Commitment.Key
      (RelationShape program domain publicRingColumns publicFits) verifierRows}
    {statements : Terminal.RunningStatements program domain
      publicRingColumns publicFits verifierRows}
    {freshPayload : FixedActive.Canonical.FreshPayload
      (Fresh.semanticShape program domain) publicRingColumns publicFits
      verifierRows}
    (input : Input valid key statements freshPayload)
    (child : Fin productionGlobalParams.k)
    (coordinate :
      Fin (RelationShape program domain publicRingColumns publicFits).carrierWidth) :
    baseAssignment input ((Layout.runningFrame key child).square coordinate) =
      input.runningWitness child coordinate *
        input.runningWitness child coordinate := by
  rw [show (Layout.runningFrame key child).square coordinate =
      Layout.localColumn (Layout.runningOwner child)
        (Layout.runningInputWidth
          (RelationShape program domain publicRingColumns publicFits)
          verifierRows + coordinate.val) by rfl]
  exact (baseAssignment_runningLocal input child _).trans
    (runningValues_square input child coordinate)

@[simp] theorem baseAssignment_freshWitness
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {valid : NativeCcsCompiler.Valid program}
    {key : Commitment.Key
      (RelationShape program domain publicRingColumns publicFits) verifierRows}
    {statements : Terminal.RunningStatements program domain
      publicRingColumns publicFits verifierRows}
    {freshPayload : FixedActive.Canonical.FreshPayload
      (Fresh.semanticShape program domain) publicRingColumns publicFits
      verifierRows}
    (input : Input valid key statements freshPayload)
    (coordinate :
      Fin (RelationShape program domain publicRingColumns publicFits).carrierWidth) :
    baseAssignment input ((Layout.freshFrame key).witness coordinate) =
      input.freshWitness coordinate := by
  rw [show (Layout.freshFrame key).witness coordinate =
      Layout.localColumn Layout.freshOwner coordinate.val by rfl]
  exact (baseAssignment_freshLocal input coordinate.val).trans
    (freshBaseValues_witness input coordinate)

@[simp] theorem baseAssignment_freshCommitment
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {valid : NativeCcsCompiler.Valid program}
    {key : Commitment.Key
      (RelationShape program domain publicRingColumns publicFits) verifierRows}
    {statements : Terminal.RunningStatements program domain
      publicRingColumns publicFits verifierRows}
    {freshPayload : FixedActive.Canonical.FreshPayload
      (Fresh.semanticShape program domain) publicRingColumns publicFits
      verifierRows}
    (input : Input valid key statements freshPayload)
    (verifierRow : Fin verifierRows)
    (output : Fin ringDegree) :
    baseAssignment input
        ((Layout.freshFrame key).commitment verifierRow output) =
      Commitment.commit key input.freshWitness verifierRow output := by
  rw [show (Layout.freshFrame key).commitment verifierRow output =
      Layout.localColumn Layout.freshOwner
        ((RelationShape program domain publicRingColumns publicFits).carrierWidth +
          verifierRow.val * ringDegree + output.val) by rfl]
  exact (baseAssignment_freshLocal input _).trans
    (freshBaseValues_commitment input verifierRow output)

@[simp] theorem baseAssignment_freshPublic
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {valid : NativeCcsCompiler.Valid program}
    {key : Commitment.Key
      (RelationShape program domain publicRingColumns publicFits) verifierRows}
    {statements : Terminal.RunningStatements program domain
      publicRingColumns publicFits verifierRows}
    {freshPayload : FixedActive.Canonical.FreshPayload
      (Fresh.semanticShape program domain) publicRingColumns publicFits
      verifierRows}
    (input : Input valid key statements freshPayload)
    (coordinate :
      Fin (RelationShape program domain publicRingColumns publicFits).publicWidth) :
    baseAssignment input ((Layout.freshFrame key).publicColumn coordinate) =
      Phi81Relation.projectPublicInput input.freshWitness coordinate := by
  rw [show (Layout.freshFrame key).publicColumn coordinate =
      Layout.localColumn Layout.freshOwner
        ((RelationShape program domain publicRingColumns publicFits).carrierWidth +
          verifierRows * ringDegree + coordinate.val) by rfl]
  exact (baseAssignment_freshLocal input _).trans
    (freshBaseValues_public input coordinate)

@[simp] theorem baseAssignment_freshSquare
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {valid : NativeCcsCompiler.Valid program}
    {key : Commitment.Key
      (RelationShape program domain publicRingColumns publicFits) verifierRows}
    {statements : Terminal.RunningStatements program domain
      publicRingColumns publicFits verifierRows}
    {freshPayload : FixedActive.Canonical.FreshPayload
      (Fresh.semanticShape program domain) publicRingColumns publicFits
      verifierRows}
    (input : Input valid key statements freshPayload)
    (coordinate :
      Fin (RelationShape program domain publicRingColumns publicFits).carrierWidth) :
    baseAssignment input ((Layout.freshFrame key).square coordinate) =
      input.freshWitness coordinate * input.freshWitness coordinate := by
  rw [show (Layout.freshFrame key).square coordinate =
      Layout.localColumn Layout.freshOwner
        (Layout.freshInputWidth
          (RelationShape program domain publicRingColumns publicFits)
          verifierRows + coordinate.val) by rfl]
  exact (baseAssignment_freshLocal input _).trans
    (freshBaseValues_square input coordinate)

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Honest
