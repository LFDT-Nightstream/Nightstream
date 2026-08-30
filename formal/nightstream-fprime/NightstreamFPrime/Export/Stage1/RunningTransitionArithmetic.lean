import NightstreamFPrime.Export.Stage1.PiDECArithmetic
import NightstreamFPrime.Layout.Stage1.RunningTransitionLowering

/-!
Owns the ordinary-row package plan for the Stage 1 running transition.
The plan is generative and is proved equal to the canonical Lean lowering.
-/

namespace NightstreamFPrime.Export.Stage1.RunningTransitionArithmetic

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1.RunningTransitionLayout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-- The running-transition packet starts at the exact end of PiDEC. -/
def rowStart : Nat :=
  NightstreamFPrime.Layout.Stage1.PiDECStarts.outputRowStart

theorem rowStart_eq_prefix
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    rowStart =
      NightstreamFPrime.Layout.Stage1.PilotPiCCSPiRLCPiDEC.physicalRowCount
        relation := by
  rw [NightstreamFPrime.Layout.Stage1.PilotPiCCSPiRLCPiDEC.physicalRowCount_eq]
  rfl

structure Plan where
  rowStart : Nat
  freshStart : Nat
  constraints : List Expr

abbrev Plan.rows (plan : Plan) : List Rows.CompiledRow :=
  PiCCSArithmetic.compilePacket plan.rowStart plan.freshStart plan.constraints

def canonicalPlan
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : Plan where
  rowStart := rowStart
  freshStart :=
    NightstreamFPrime.Layout.Stage1.RunningTransitionLayout.logicalColumnCount
  constraints :=
    NightstreamFPrime.Lifecycle.Stage1.RunningTransition.constraintsFast
      (NightstreamFPrime.Layout.Stage1.RunningTransitionInputs.interface
        logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.RunningTransitionInputs.phaseOffset

abbrev canonicalLayoutPlan
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : R1CS.LoweringPlan :=
  NightstreamFPrime.Layout.Stage1.RunningTransitionLayout.plan
    logicalWidth publicFits

structure Plan.MatchesLayout (packagePlan : Plan)
    (layoutPlan : R1CS.LoweringPlan) : Prop where
  constraints : packagePlan.constraints = layoutPlan.constraints
  freshStart : packagePlan.freshStart = layoutPlan.firstFresh

theorem Plan.rows_toR1CS (plan : Plan) :
    plan.rows.map Rows.CompiledRow.toR1CS =
      NightstreamFPrime.Layout.Stage1.Spartan.remapRows
        (R1CS.lowerConstraints plan.constraints plan.freshStart).rows :=
  PiCCSArithmetic.compilePacket_toR1CS
    plan.rowStart plan.freshStart plan.constraints

theorem Plan.rows_to_layout (packagePlan : Plan)
    (layoutPlan : R1CS.LoweringPlan)
    (agreement : packagePlan.MatchesLayout layoutPlan) :
    packagePlan.rows.map Rows.CompiledRow.toR1CS =
      NightstreamFPrime.Layout.Stage1.Spartan.remapRows layoutPlan.rows := by
  calc
    _ = NightstreamFPrime.Layout.Stage1.Spartan.remapRows
        (R1CS.lowerConstraints packagePlan.constraints
          packagePlan.freshStart).rows := packagePlan.rows_toR1CS
    _ = NightstreamFPrime.Layout.Stage1.Spartan.remapRows layoutPlan.rows := by
      apply congrArg NightstreamFPrime.Layout.Stage1.Spartan.remapRows
      rw [agreement.constraints, agreement.freshStart]
      rfl

theorem canonicalPlan_matches
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    (canonicalPlan logicalWidth publicFits).MatchesLayout
      (canonicalLayoutPlan logicalWidth publicFits) := by
  constructor
  · change
      NightstreamFPrime.Lifecycle.Stage1.RunningTransition.constraintsFast
          (NightstreamFPrime.Layout.Stage1.RunningTransitionInputs.interface
            logicalWidth publicFits)
          NightstreamFPrime.Layout.Stage1.RunningTransitionInputs.phaseOffset =
        logicalConstraints logicalWidth publicFits
    rw [NightstreamFPrime.Lifecycle.Stage1.RunningTransition.constraintsFast_eq_constraints]
    exact (logicalConstraints_eq logicalWidth publicFits).symm
  · rfl

theorem Plan.rows_length (plan : Plan) :
    plan.rows.length = R1CS.totalRowCount plan.constraints :=
  PiCCSArithmetic.compilePacket_length
    plan.rowStart plan.freshStart plan.constraints

theorem canonicalPlan_rowCount
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    R1CS.totalRowCount (canonicalPlan logicalWidth publicFits).constraints =
      321303 := by
  change R1CS.totalRowCount
      (NightstreamFPrime.Lifecycle.Stage1.RunningTransition.constraintsFast
        (NightstreamFPrime.Layout.Stage1.RunningTransitionInputs.interface
          logicalWidth publicFits)
        NightstreamFPrime.Layout.Stage1.RunningTransitionInputs.phaseOffset) =
    321303
  rw [NightstreamFPrime.Lifecycle.Stage1.RunningTransition.constraintsFast_eq_constraints,
    ← logicalConstraints_eq]
  exact NightstreamFPrime.Layout.Stage1.RunningTransitionLayout.totalRowCount_eq
    relation

end NightstreamFPrime.Export.Stage1.RunningTransitionArithmetic
