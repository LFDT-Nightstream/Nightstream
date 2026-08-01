import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
import Nightstream.SuperNeo.Concrete.Phi81Relation.Semantics
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.CarrierEquality

/-!
Lean-owned terminal relations for the selected ConcretePhi81 NIFS.

Assurance tier: model-level.

Owns: the private witness types for the fourteen running CE claims and the
one fresh CCS claim; finite executable checks for commitment, public
projection, strict norm, carried evaluations, and CCS satisfaction; and exact
Boolean-to-relation theorems.

Does not own: delayed packed-`yZcol` closure, physical R1CS rows, a terminal
call recipe, Ajtai binding security, Rust, or generated artifacts.

Emits constraints: none.

The checked parent carried by `SelectedRunning` is NIFS transport state. The
paper terminal relation checks the fourteen persistent CE children. The
production delayed-projection extension must additionally carry and check its
pending value; this module does not treat the parent as a fifteenth CE claim.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsTerminalRelations

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Protocol.FPrime.CanonicalTerminalVerifier
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.CarrierEquality
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

universe uTranscriptState

variable
  {shape : SemanticShape}
  {TranscriptState : Type uTranscriptState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

private abbrev RelationShape
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits : ringDegree * publicRingColumns <= shape.carrierWidth) :=
  Phi81Relation.Shape.ofSemantic shape publicRingColumns publicFits

/-- One private assignment for every persistent running CE claim. -/
abbrev RunningWitness
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits : ringDegree * publicRingColumns <= shape.carrierWidth) :=
  Fin productionGlobalParams.k ->
    Phi81Relation.Assignment (RelationShape shape publicRingColumns publicFits)

/-- One private assignment for the selected fresh CCS claim. -/
abbrev FreshWitness
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits : ringDegree * publicRingColumns <= shape.carrierWidth) :=
  Phi81Relation.Assignment
    (RelationShape shape publicRingColumns publicFits)

private def commitmentMap
    (key :
      SelectedKey shape TranscriptState publicRingColumns publicFits
        verifierRows) :
    Phi81Relation.Assignment
      (RelationShape shape publicRingColumns publicFits) ->
        CommitmentValue verifierRows :=
  PiRLCAlgebra.Commitment.commit key.template.key

/-- Exact paper membership of one persistent running child. -/
def ChildHolds
    (key :
      SelectedKey shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (witness : RunningWitness shape publicRingColumns publicFits)
    (child : Fin productionGlobalParams.k) : Prop :=
  CE.Holds
    (Phi81Relation.relationSemantics (commitmentMap key))
    productionGlobalParams
    ((running.children child).materialize key.system)
    (witness child)

/-- HyperNova's running relation is the product of all fourteen CE claims. -/
def RunningHolds
    (key :
      SelectedKey shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (witness : RunningWitness shape publicRingColumns publicFits) : Prop :=
  ∀ child, ChildHolds key running witness child

/-- HyperNova's fresh relation is one complete norm-bounded CCS claim. -/
def FreshHolds
    (key :
      SelectedKey shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (witness : FreshWitness shape publicRingColumns publicFits) : Prop :=
  CCS.Holds
    (Phi81Relation.relationSemantics (commitmentMap key))
    productionGlobalParams
    (fresh.materialize key.system)
    witness

/-- Finite complete-carrier strict-norm check. -/
def normCheck
    (bound : Nat)
    (assignment :
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits)) : Bool :=
  (List.finRange
      (RelationShape shape publicRingColumns publicFits).carrierWidth).all
    fun column =>
    decide (centeredMagnitude (assignment column) < bound)

theorem normCheck_eq_true_iff
    (bound : Nat)
    (assignment :
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits)) :
    normCheck bound assignment = true ↔
      Phi81Relation.assignmentNormBounded bound assignment := by
  constructor
  · intro checked column
    exact of_decide_eq_true
      ((List.all_eq_true.mp checked) column (by simp))
  · intro bounded
    apply List.all_eq_true.mpr
    intro column _member
    exact decide_eq_true (bounded column)

/-- Execute the explicit sparse CCS residual on every Boolean row. -/
def ccsCheck
    (system :
      Phi81Relation.Structure
        (RelationShape shape publicRingColumns publicFits))
    (assignment :
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits)) : Bool :=
  (CCSResidualTable.residualTable ConcreteCarrier.baseOps
      system.matrixSource.system assignment).entries.all fun value =>
    decide (value = ConcreteCarrier.baseOps.zero)

theorem ccsCheck_eq_true_iff
    (system :
      Phi81Relation.Structure
        (RelationShape shape publicRingColumns publicFits))
    (assignment :
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits)) :
    ccsCheck system assignment = true ↔
      Phi81Relation.ccsSatisfied system assignment := by
  let table :=
    CCSResidualTable.residualTable ConcreteCarrier.baseOps
      system.matrixSource.system assignment
  have tableExact :
      table.AllEntriesZero ConcreteCarrier.baseOps ↔
        Phi81Relation.ccsSatisfied system assignment := by
    exact
      CCSResidualTable.residualTable_allEntriesZero_iff_constraintSatisfied
        ConcreteCarrier.baseOps system.matrixSource.system assignment
  rw [← tableExact]
  constructor
  · intro checked value member
    exact of_decide_eq_true
      ((List.all_eq_true.mp checked) value member)
  · intro zero
    apply List.all_eq_true.mpr
    intro value member
    exact decide_eq_true (zero value member)

/-- Finite executable check for one running CE child. -/
def childCheck
    (key :
      SelectedKey shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (witness : RunningWitness shape publicRingColumns publicFits)
    (child : Fin productionGlobalParams.k) : Bool :=
  let statement := (running.children child).materialize key.system
  let assignment := witness child
  commitmentEqual (commitmentMap key assignment) statement.commitment &&
    (publicInputEqual (Phi81Relation.projectPublicInput assignment)
        statement.publicInput &&
      (normCheck (statement.stage.bound productionGlobalParams) assignment &&
        evaluationsEqual
          (Phi81Relation.evaluations statement.constraintSystem assignment
            statement.point)
          statement.evaluations))

theorem childCheck_eq_true_iff
    (key :
      SelectedKey shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (witness : RunningWitness shape publicRingColumns publicFits)
    (child : Fin productionGlobalParams.k) :
    childCheck key running witness child = true ↔
      ChildHolds key running witness child := by
  simp only [childCheck, Bool.and_eq_true,
    commitmentEqual_eq_true_iff, publicInputEqual_eq_true_iff,
    normCheck_eq_true_iff, evaluationsEqual_eq_true_iff]
  exact
    Phi81Relation.ceMembership_iff
      (commitmentMap key) productionGlobalParams
      ((running.children child).materialize key.system) (witness child)
      |>.symm

/-- Execute all fourteen running checks in canonical child order. -/
def runningCheck
    (key :
      SelectedKey shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (witness : RunningWitness shape publicRingColumns publicFits) : Bool :=
  (List.finRange productionGlobalParams.k).all fun child =>
    childCheck key running witness child

theorem runningCheck_eq_true_iff
    (key :
      SelectedKey shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (witness : RunningWitness shape publicRingColumns publicFits) :
    runningCheck key running witness = true ↔
      RunningHolds key running witness := by
  constructor
  · intro checked child
    exact
      (childCheck_eq_true_iff key running witness child).mp
        ((List.all_eq_true.mp checked) child (List.mem_finRange child))
  · intro holds
    apply List.all_eq_true.mpr
    intro child _member
    exact (childCheck_eq_true_iff key running witness child).mpr
      (holds child)

/-- Execute the complete fresh CCS opening and relation check. -/
def freshCheck
    (key :
      SelectedKey shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (witness : FreshWitness shape publicRingColumns publicFits) : Bool :=
  let statement := fresh.materialize key.system
  commitmentEqual (commitmentMap key witness) statement.commitment &&
    (publicInputEqual (Phi81Relation.projectPublicInput witness)
        statement.publicInput &&
      (normCheck (statement.stage.bound productionGlobalParams) witness &&
        ccsCheck statement.constraintSystem witness))

theorem freshCheck_eq_true_iff
    (key :
      SelectedKey shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (witness : FreshWitness shape publicRingColumns publicFits) :
    freshCheck key fresh witness = true ↔
      FreshHolds key fresh witness := by
  simp only [freshCheck, Bool.and_eq_true,
    commitmentEqual_eq_true_iff, publicInputEqual_eq_true_iff,
    normCheck_eq_true_iff, ccsCheck_eq_true_iff]
  exact
    Phi81Relation.ccsMembership_iff
      (commitmentMap key) productionGlobalParams
      (fresh.materialize key.system) witness
      |>.symm

/-- Exact terminal relation family for the fixed-one selected NIFS. -/
def relations :
    TerminalRelations
      (SelectedKey shape TranscriptState publicRingColumns publicFits
        verifierRows)
      (SelectedRunning shape publicRingColumns publicFits verifierRows)
      (RunningWitness shape publicRingColumns publicFits)
      (SelectedFresh shape publicRingColumns publicFits verifierRows)
      (FreshWitness shape publicRingColumns publicFits)
      1 where
  runningHolds := fun _slot key running witness =>
    RunningHolds key running witness
  freshHolds := fun _slot key fresh witness =>
    FreshHolds key fresh witness

/-- Exact executable checkers for the same relation family. -/
def checks : RelationChecks (relations
    (shape := shape)
    (TranscriptState := TranscriptState)
    (publicRingColumns := publicRingColumns)
    (verifierRows := verifierRows)
    (publicFits := publicFits)) where
  runningCheck := fun _slot key running witness =>
    runningCheck key running witness
  freshCheck := fun _slot key fresh witness =>
    freshCheck key fresh witness
  runningCheck_iff := by
    intro slot key running witness
    exact runningCheck_eq_true_iff key running witness
  freshCheck_iff := by
    intro slot key fresh witness
    exact freshCheck_eq_true_iff key fresh witness

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsTerminalRelations
