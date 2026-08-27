import NightstreamFPrime.Lifecycle.PiDEC.v1_1.InputBinding
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit

/-!
Paper authority: SuperNeo v1.1, Section 7.5, PiDEC output.

Obligation: package exactly sixteen fresh child claims. Each child reuses the
prover message commitment and evaluation, the verifier-computed public digit,
the parent's fixed relation, and the parent's shared point.

This leaf adds no witness cell, copy row, or assertion row. The PiDEC parent
proves that this family equals `PiDEC.PaperVerifier.children` after the public
split child succeeds.
-/

namespace NightstreamFPrime.Lifecycle.PiDEC.v1_1.OutputBinding

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra

structure Interface (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) where
  point : Nat → Fin productionShape.cubeVariables → KExpr
  message : Nat → Radix.ChildIndex → InputBinding.ChildMessageExpr
  publicInput : Nat → Radix.ChildIndex →
    Fin (PublicInputSplit.coordinateCount logicalWidth publicFits) → Expr

abbrev Output (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :=
  CE.Instance (PaperAlgebra.Structure logicalWidth)
    (PaperAlgebra.PublicInput
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    PaperAlgebra.Point PaperAlgebra.Evaluation PaperAlgebra.Commitment

def evalOutput
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) :
    Radix.ChildIndex → Output logicalWidth publicFits :=
  fun child => {
    constraintSystem := InputBinding.relationSource relation
    commitment := fun row lane =>
      ((interface.message offset child).commitment row lane).eval env
    publicInput := fun coordinate =>
      (interface.publicInput offset child coordinate).eval env
    point := InputBinding.evalPoint (interface.point offset) env
    evaluations := #[
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.evalEvaluation
        (interface.message offset child).evaluation env]
    stage := NormStage.fresh }

@[simp] theorem evalOutput_commitment
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) (child : Radix.ChildIndex) :
    (evalOutput relation interface offset env child).commitment =
      (InputBinding.evalMessage (interface.message offset child) env).commitment := by
  rfl

@[simp] theorem evalOutput_evaluations
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) (child : Radix.ChildIndex) :
    (evalOutput relation interface offset env child).evaluations =
      (InputBinding.evalMessage
        (interface.message offset child) env).evaluations := by
  rfl

@[simp] theorem evalOutput_publicInput
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) (child : Radix.ChildIndex) :
    (evalOutput relation interface offset env child).publicInput =
      fun coordinate =>
        (interface.publicInput offset child coordinate).eval env := by
  rfl

structure SpecHolds
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) : Prop where
  outputFresh : ∀ child,
    (evalOutput relation interface offset env child).stage = NormStage.fresh
  sameStructure : ∀ child,
    (evalOutput relation interface offset env child).constraintSystem =
      InputBinding.relationSource relation
  samePoint : ∀ child,
    (evalOutput relation interface offset env child).point =
      InputBinding.evalPoint (interface.point offset) env
  evaluationSize : ∀ child,
    (evalOutput relation interface offset env child).evaluations.size = 1

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
    exact ⟨fun _ => rfl, fun _ => rfl, fun _ => rfl, fun _ => rfl⟩
  completeness := by
    intro env _ _ _
    exact ⟨env, fun _ _ => rfl, fun _ member => by cases member⟩

theorem soundness
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (rows : holds env (Circuit.ops (circuit relation interface).main offset)) :
    SpecHolds relation interface offset env :=
  (circuit relation interface).soundness env offset trivial rows

theorem completeness
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
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
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (specification : SpecHolds relation interface offset env) :
    (∀ child,
      (evalOutput relation interface offset env child).stage = NormStage.fresh) ∧
    (∀ child,
      (evalOutput relation interface offset env child).constraintSystem =
        InputBinding.relationSource relation) ∧
    (∀ child,
      (evalOutput relation interface offset env child).point =
        InputBinding.evalPoint (interface.point offset) env) ∧
    (∀ child,
      (evalOutput relation interface offset env child).evaluations.size = 1) :=
  ⟨specification.outputFresh, specification.sameStructure,
    specification.samePoint, specification.evaluationSize⟩

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

end NightstreamFPrime.Lifecycle.PiDEC.v1_1.OutputBinding
