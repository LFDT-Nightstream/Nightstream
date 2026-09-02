import NightstreamFPrime.Export.Stage1.PiCCSArithmetic
import NightstreamFPrime.Layout.Stage1.PiDECInputs
import NightstreamFPrime.Layout.Stage1.PiDECStarts

/-!
Owns the ordinary-row encoding of the exact PiDEC v1_1 phase.

The relation-parameterized input and output binding children have no rows.
The exported list therefore contains, in parent order, only the public split,
commitment recomposition, separate `Eval_K` recomposition, and separate
14-matrix `Eval_A` recomposition constraints. The equality theorem below ties
that relation-independent executable list to the canonical Lean lowering for
every verifier-owned logical relation.
-/

namespace NightstreamFPrime.Export.Stage1.PiDECArithmetic

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiDEC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

def phaseInterface
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    Formal.Interface logicalWidth publicFits :=
  NightstreamFPrime.Layout.Stage1.PiDECInputs.interface logicalWidth publicFits

def publicInputConstraints
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List Expr :=
  let shared := Formal.atOffset (phaseInterface logicalWidth publicFits)
    NightstreamFPrime.Layout.Stage1.PiDECInputs.phaseOffset
  NightstreamFPrime.Layout.PiDEC.v1_1.childConstraints
    (Formal.publicInputCircuit shared)
    (Formal.publicInputOffset
      NightstreamFPrime.Layout.Stage1.PiDECInputs.phaseOffset)

def commitmentConstraints
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List Expr :=
  let shared := Formal.atOffset (phaseInterface logicalWidth publicFits)
    NightstreamFPrime.Layout.Stage1.PiDECInputs.phaseOffset
  NightstreamFPrime.Layout.PiDEC.v1_1.childConstraints
    (Formal.commitmentCircuit shared)
    (Formal.commitmentOffset
      NightstreamFPrime.Layout.Stage1.PiDECInputs.phaseOffset)

def evalKConstraints
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List Expr :=
  let shared := Formal.atOffset (phaseInterface logicalWidth publicFits)
    NightstreamFPrime.Layout.Stage1.PiDECInputs.phaseOffset
  NightstreamFPrime.Layout.PiDEC.v1_1.childConstraints
    (Formal.evalKCircuit shared)
    (Formal.evalKOffset NightstreamFPrime.Layout.Stage1.PiDECInputs.phaseOffset)

def evalAConstraints
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List Expr :=
  let shared := Formal.atOffset (phaseInterface logicalWidth publicFits)
    NightstreamFPrime.Layout.Stage1.PiDECInputs.phaseOffset
  NightstreamFPrime.Layout.PiDEC.v1_1.childConstraints
    (Formal.evalACircuit shared)
    (Formal.evalAOffset NightstreamFPrime.Layout.Stage1.PiDECInputs.phaseOffset)

/-- The four nonempty PiDEC child lists in exact parent order. -/
def constraints
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List Expr :=
  publicInputConstraints logicalWidth publicFits ++
    commitmentConstraints logicalWidth publicFits ++
      evalKConstraints logicalWidth publicFits ++
        evalAConstraints logicalWidth publicFits

theorem constraints_eq_nonBoundary
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    constraints logicalWidth publicFits =
      NightstreamFPrime.Layout.PiDEC.v1_1.nonBoundaryConstraints
        (phaseInterface logicalWidth publicFits)
        NightstreamFPrime.Layout.Stage1.PiDECInputs.phaseOffset := by
  rfl

theorem constraints_eq_logical
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    constraints logicalWidth publicFits =
      NightstreamFPrime.Layout.PiDEC.v1_1.logicalConstraints relation
        (phaseInterface logicalWidth publicFits)
        NightstreamFPrime.Layout.Stage1.PiDECInputs.phaseOffset := by
  rw [constraints_eq_nonBoundary]
  exact (NightstreamFPrime.Layout.PiDEC.v1_1.logicalConstraints_eq_nonBoundary
    relation (phaseInterface logicalWidth publicFits)
    NightstreamFPrime.Layout.Stage1.PiDECInputs.phaseOffset).symm

structure Plan where
  rowStart : Nat
  freshStart : Nat
  constraints : List Expr

/-- Generic executable expansion. Its inputs remain explicit so Lean does not
specialize a phase-sized row list while it checks this module. -/
abbrev Plan.rows (plan : Plan) : List Rows.CompiledRow :=
  PiCCSArithmetic.compilePacket plan.rowStart plan.freshStart plan.constraints

def canonicalPlan
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : Plan where
  rowStart := NightstreamFPrime.Layout.Stage1.PiDECStarts.phaseRowStart
  freshStart := NightstreamFPrime.Layout.Stage1.PiDECStarts.phaseFreshStart
  constraints := constraints logicalWidth publicFits

abbrev canonicalLayoutPlan
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    R1CS.LoweringPlan :=
  NightstreamFPrime.Layout.PiDEC.v1_1.plan relation
    (phaseInterface logicalWidth publicFits)
    NightstreamFPrime.Layout.Stage1.PiDECInputs.phaseOffset

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

/-- A matching generative plan expands to the exact Lean layout rows. This
theorem is parametric in both plans, so its proof cost is independent of the
number of rows in a concrete phase. -/
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
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (canonicalPlan logicalWidth publicFits).MatchesLayout
      (canonicalLayoutPlan relation) := by
  constructor
  · exact constraints_eq_logical relation
  · rfl

theorem Plan.rows_length (plan : Plan) :
    plan.rows.length = R1CS.totalRowCount plan.constraints :=
  PiCCSArithmetic.compilePacket_length
    plan.rowStart plan.freshStart plan.constraints

theorem canonicalPlan_rowCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    R1CS.totalRowCount (canonicalPlan logicalWidth publicFits).constraints =
      25488 := by
  change R1CS.totalRowCount (constraints logicalWidth publicFits) = 25488
  rw [constraints_eq_logical relation]
  exact NightstreamFPrime.Layout.PiDEC.v1_1.totalRowCount_eq relation
    (phaseInterface logicalWidth publicFits)
    NightstreamFPrime.Layout.Stage1.PiDECInputs.phaseOffset
    (NightstreamFPrime.Layout.Stage1.PiDECInputs.inputShapes relation)

end NightstreamFPrime.Export.Stage1.PiDECArithmetic
