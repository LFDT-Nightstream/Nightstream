import NightstreamFPrime.Export.Stage1.Package
import NightstreamFPrime.Layout.Stage1.RunningTransitionPreservation

/-!
Owns the soundness bridge from the canonical Stage 1 package rows to the
running-transition semantics. It adds no package row and no alternate
transition relation.
-/

namespace NightstreamFPrime.Export.Stage1.RunningTransitionPackage

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Lifecycle.Stage1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

theorem circuitPackage_implies_physicalHolds
    (env : Env)
    (holds : (Data.circuitPackage ()).RowsHold env) :
    NightstreamFPrime.Layout.Stage1.RunningTransitionLayout.PhysicalHolds
      Data.logicalWidth Data.publicFits
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  have packageRows :=
    Package.circuitPackage_implies_runningTransitionArithmeticRows env holds
  have exactRows := RunningTransitionArithmetic.Plan.rows_to_layout
    (RunningTransitionArithmetic.canonicalPlan
      Data.logicalWidth Data.publicFits)
    (RunningTransitionArithmetic.canonicalLayoutPlan
      Data.logicalWidth Data.publicFits)
    (RunningTransitionArithmetic.canonicalPlan_matches
      Data.logicalWidth Data.publicFits)
  rw [exactRows] at packageRows
  exact
    (NightstreamFPrime.Layout.Stage1.Spartan.remapRows_hold env _).mp
      packageRows

theorem circuitPackage_implies_specHolds
    (relation : ProductionKey.LogicalRelation Data.logicalWidth
      Data.publicFits)
    (env : Env)
    (holds : (Data.circuitPackage ()).RowsHold env) :
    RunningTransition.SpecHolds
      (NightstreamFPrime.Layout.Stage1.RunningTransitionInputs.interface
        Data.logicalWidth Data.publicFits)
      NightstreamFPrime.Layout.Stage1.RunningTransitionInputs.phaseOffset
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  exact
    NightstreamFPrime.Layout.Stage1.RunningTransitionLayout.physical_implies_specHolds
      relation _
        (circuitPackage_implies_physicalHolds env holds)

theorem circuitPackage_implies_typed_base
    (relation : ProductionKey.LogicalRelation Data.logicalWidth
      Data.publicFits)
    (env : Env)
    (holds : (Data.circuitPackage ()).RowsHold env)
    (iterationZero : RunningTransition.iterationValue
      (NightstreamFPrime.Layout.Stage1.RunningTransitionInputs.interface
        Data.logicalWidth Data.publicFits)
      NightstreamFPrime.Layout.Stage1.RunningTransitionInputs.phaseOffset
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) = 0) :
    StatementAbsorption.evalRunning
        (NightstreamFPrime.Layout.Stage1.RunningTransitionInputs.outputRunningExpr
          Data.logicalWidth Data.publicFits)
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) =
      defaultRunning (logicalWidth := Data.logicalWidth)
        (publicFits := Data.publicFits) := by
  exact
    NightstreamFPrime.Layout.Stage1.RunningTransitionLayout.physical_implies_typed_base
      relation _
        (circuitPackage_implies_physicalHolds env holds) iterationZero

theorem circuitPackage_implies_typed_recursive
    (relation : ProductionKey.LogicalRelation Data.logicalWidth
      Data.publicFits)
    (env : Env)
    (holds : (Data.circuitPackage ()).RowsHold env)
    (iterationNonzero : RunningTransition.iterationValue
      (NightstreamFPrime.Layout.Stage1.RunningTransitionInputs.interface
        Data.logicalWidth Data.publicFits)
      NightstreamFPrime.Layout.Stage1.RunningTransitionInputs.phaseOffset
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) ≠ 0) :
    StatementAbsorption.evalRunning
        (NightstreamFPrime.Layout.Stage1.RunningTransitionInputs.outputRunningExpr
          Data.logicalWidth Data.publicFits)
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) =
      NightstreamFPrime.Layout.Stage1.RunningTransitionInputs.piDecRunningOutput
        relation
          (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  exact
    NightstreamFPrime.Layout.Stage1.RunningTransitionLayout.physical_implies_typed_recursive
      relation _
        (circuitPackage_implies_physicalHolds env holds) iterationNonzero

end NightstreamFPrime.Export.Stage1.RunningTransitionPackage
