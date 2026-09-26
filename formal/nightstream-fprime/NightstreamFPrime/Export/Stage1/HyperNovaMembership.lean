import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalAssignment
import NightstreamFPrime.Export.Stage1.PerApplicationTerminal
import NightstreamFPrime.Layout.ProductionRelation.CcsOpening

/-! Derive fresh and terminal membership from the exact canonical rows,
opening bounds, and carried claims. These lemmas do not select a sampler key. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaMembership

open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.ProductionRelation
open PerApplicationCanonicalAssignment

private theorem completeAssignment_eq_extend {program : Program} (raw : RawValues program) :
    raw.completeAssignment = Phi81CarrierLayout.extendAssignment 0 raw.assignment := by
  funext column
  by_cases below : column.val < PerApplicationFixedPoint.logicalWidth program
  · simp only [RawValues.completeAssignment, Phi81CarrierLayout.extendAssignment,
      Phi81CarrierLayout.logicalColumn?, dif_pos below]
  · simp only [RawValues.completeAssignment, Phi81CarrierLayout.extendAssignment,
      Phi81CarrierLayout.logicalColumn?, dif_neg below]

theorem freshHolds_of_rows
    (program : Program) (fit : PerApplicationFixedPoint.FitsTwoPow28 program)
    (ajtai : AjtaiKey (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program))
    (raw : RawValues program)
    (rows : (PerApplicationFixedPoint.structuralPlan program fit).RowsZero raw.assignment)
    (bounded : ∀ column, centeredMagnitude (raw.completeAssignment column) < 2) :
    CCS.Holds (semantics ajtai) productionGlobalParams
      (Lifecycle.freshStatement (PerApplicationFixedPoint.relation program fit)
        { commitments := fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commit ajtai raw.completeAssignment
          publicInputs := fun _ => encHash raw.outputDigest }) raw.completeAssignment := by
  have completed := completeAssignment_eq_extend raw
  have logicalBound : ∀ column, centeredMagnitude (raw.assignment column) < 2 := by
    intro column
    have same : raw.completeAssignment (Phi81CarrierLayout.embedLogical column) =
        raw.assignment column :=
      (congrFun completed (Phi81CarrierLayout.embedLogical column)).trans
        (Phi81CarrierLayout.extendAssignment_embedLogical (0 : F) raw.assignment column)
    exact Eq.mp (congrArg (fun value : F => centeredMagnitude value < 2) same)
      (bounded (Phi81CarrierLayout.embedLogical column))
  have publicInput := PerApplicationCanonicalAssignment.projectPublicInput_completeAssignment raw
  rw [completed] at publicInput
  have member := Plan.rowsZero_implies_freshHolds
    (PerApplicationFixedPoint.structuralPlan program fit) fit.carrier ajtai
    raw.assignment (encHash raw.outputDigest) rows logicalBound publicInput
  exact Eq.mpr (congrArg (fun assignment : PaperAlgebra.Assignment
      (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program) =>
    CCS.Holds (semantics ajtai) productionGlobalParams
      (Lifecycle.freshStatement (PerApplicationFixedPoint.relation program fit)
        { commitments := fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commit ajtai assignment
          publicInputs := fun _ => encHash raw.outputDigest }) assignment) completed) member

theorem terminal_of_memberships
    (program : Program) (fit : PerApplicationFixedPoint.FitsTwoPow28 program)
    (commitmentSetup : PerApplicationCanonicalPackage.CommitmentSetup program)
    (statement : TerminalStatement AppState)
    (running : Running (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program))
    (children : Stage1.Terminal.RunningWitness
      (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program))
    (raw : RawValues program)
    (valid : Stage1.Terminal.StatementValid statement)
    (positive : 0 < statement.iteration)
    (digest : raw.outputDigest = stateHash {
      verifierKeys := fun _ => PerApplicationCanonicalPackage.verifierContextDigest fit commitmentSetup
      iteration := statement.iteration
      z0 := statement.z0
      current := statement.zi
      running := fun _ => running
      pc := 1 })
    (runningMember : ∀ child,
      CE.Holds (semantics (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)) productionGlobalParams
        (Lifecycle.runningStatement (PerApplicationFixedPoint.relation program fit) running child)
        (children child))
    (freshMember : CCS.Holds
      (semantics (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)) productionGlobalParams
      (Lifecycle.freshStatement (PerApplicationFixedPoint.relation program fit)
        { commitments := fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commit
            (PerApplicationCanonicalPackage.commitmentKey commitmentSetup) raw.completeAssignment
          publicInputs := fun _ => encHash raw.outputDigest }) raw.completeAssignment) :
    PerApplicationTerminal.Holds program fit commitmentSetup statement (.recursive {
      running := fun _ => running
      runningWitness := fun _ => children
      fresh := {
        commitments := fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commit
          (PerApplicationCanonicalPackage.commitmentKey commitmentSetup) raw.completeAssignment
        publicInputs := fun _ => encHash raw.outputDigest }
      freshWitness := raw.completeAssignment
      pc := 1 }) := by
  apply (PerApplicationTerminal.holds_recursive_iff program fit commitmentSetup statement _).mpr
  refine ⟨valid, ?_⟩
  refine ⟨(show InRange slotCount 1 from ⟨Nat.le_refl 1, Nat.le_refl 1⟩),
    positive, ?_, ?_, ?_⟩
  · exact congrArg (encHash (publicFits := PerApplicationFixedPoint.publicFits program)) digest
  · intro slot
    exact runningMember
  · exact freshMember

end NightstreamFPrime.Export.Stage1.HyperNovaMembership
