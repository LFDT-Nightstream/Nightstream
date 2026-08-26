import NightstreamFPrime.Lifecycle.PiRLC.v1_1.InputBinding

/-!
Paper authority: SuperNeo v1.1, Section 7.4, PiRLC output.
Obligation: package the verifier-computed commitment, public input, separate
Pad evaluation, and 14 matrix evaluations as one `CE(B)` claim at stage
`.combined` for the shared relation source and point.

This leaf reuses the computed expressions directly. It adds no witness cell,
copy row, transcript action, or alternate relation.
-/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_1.OutputBinding

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

structure Interface (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) where
  point : Nat → Fin productionShape.cubeVariables → KExpr
  commitment : Nat → Fin productionProfile.commitmentWidth →
    Fin ringDegree → Expr
  publicInput : Nat → Fin (FullShape logicalWidth publicFits).publicWidth → Expr
  eval_K : Nat → Fin productionShape.coefficientCount → KExpr
  eval_A : Nat → Fin productionShape.matrixCount →
    Fin productionShape.coefficientCount → KExpr

def evalEvaluation
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    (env : Env) : PaperAlgebra.Evaluation where
  pad := fun coefficient => (interface.eval_K offset coefficient).eval env
  matrix := fun matrix coefficient =>
    (interface.eval_A offset matrix coefficient).eval env

def evalOutput
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    (env : Env) : InputBinding.InputInstance logicalWidth publicFits where
  constraintSystem := InputBinding.relationSource relation
  commitment := fun row lane => (interface.commitment offset row lane).eval env
  publicInput := fun column => (interface.publicInput offset column).eval env
  point := InputBinding.evalPoint (interface.point offset) env
  evaluations := #[evalEvaluation interface offset env]
  stage := .combined

structure SpecHolds
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    (env : Env) : Prop where
  outputCombined : (evalOutput relation interface offset env).stage = .combined
  sameStructure : (evalOutput relation interface offset env).constraintSystem =
    InputBinding.relationSource relation
  samePoint : (evalOutput relation interface offset env).point =
    InputBinding.evalPoint (interface.point offset) env

def circuit
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) : FormalCircuit where
  main := pure ()
  spec := SpecHolds relation interface
  soundness := by
    intro _ _ _ _
    exact ⟨rfl, rfl, rfl⟩
  completeness := by
    intro env _ _ _
    exact ⟨env, fun _ _ => rfl, fun _ member => by cases member⟩

theorem soundness
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset : Nat) (env : Env)
    (rows : holds env (Circuit.ops (circuit relation interface).main offset)) :
    SpecHolds relation interface offset env :=
  (circuit relation interface).soundness env offset trivial rows

theorem completeness
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset : Nat) (env : Env)
    (specification : SpecHolds relation interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (circuit relation interface).main offset)) ∧
      holdsFlat completed
        (Circuit.ops (circuit relation interface).main offset) :=
  (circuit relation interface).completeness env offset trivial specification

theorem parentCoverage
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset : Nat) (env : Env)
    (specification : SpecHolds relation interface offset env) :
    (evalOutput relation interface offset env).stage = .combined ∧
      (evalOutput relation interface offset env).constraintSystem =
        InputBinding.relationSource relation ∧
      (evalOutput relation interface offset env).point =
        InputBinding.evalPoint (interface.point offset) env :=
  ⟨specification.outputCombined, specification.sameStructure,
    specification.samePoint⟩

theorem localLength_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    localLength (Circuit.ops (circuit relation interface).main offset) = 0 := by
  rfl

theorem flatConstraints_length
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    (flatConstraints
      (Circuit.ops (circuit relation interface).main offset)).length = 0 := by
  rfl

theorem flatConstraints_varsBelow
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits) (offset : Nat) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops (circuit relation interface).main offset),
      expression.VarsBelow offset := by
  intro _ member
  cases member

end NightstreamFPrime.Lifecycle.PiRLC.v1_1.OutputBinding
