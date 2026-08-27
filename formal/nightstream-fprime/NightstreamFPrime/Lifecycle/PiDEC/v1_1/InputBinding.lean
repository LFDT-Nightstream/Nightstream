import NightstreamFPrime.Circuit.Quadratic
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.InputBinding
import NightstreamFPrime.Spec.Folding.PiDEC.PaperVerifier
import NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Radix

/-!
Paper authority: SuperNeo v1.1, Section 7.5, input/output declarations and
verifier Step 2.

Obligation: bind one combined parent claim and exactly sixteen prover child
messages to the operational `PiDEC.PaperVerifier.Attempt`. The verifier copies
the relation structure and point, computes child public inputs, and marks every
child fresh.

Inputs:
- one parent commitment, public input, point, separate `Eval_K`, and separate
  14-matrix `Eval_A` family;
- sixteen child commitment and evaluation messages.

Outputs:
- the same expressions viewed as the exact operational PiDEC attempt.

Constraint groups:
- C1: none; all fields share the same symbolic expressions;
- C2: parent stage, child stage, structure, point, and singleton evaluation
  arity are fixed by typed construction.

Parent coverage:
- `PiDEC.PaperVerifier.Accepted.parentCombined`;
- `Accepted.parentEvaluationSize`;
- `Accepted.messageEvaluationSize`;
- verifier-computed fields of `PiDEC.PaperVerifier.children`.

The Stage 1 assembler owns the PiRLC-output-to-parent wiring.
-/

namespace NightstreamFPrime.Lifecycle.PiDEC.v1_1.InputBinding

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra

abbrev ParentExpr (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.InputBinding.InputExpr
    logicalWidth publicFits

structure ChildMessageExpr where
  commitment : Fin productionProfile.commitmentWidth → Fin ringDegree → Expr
  evaluation :
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.EvaluationExpr

structure Interface (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) where
  parent : Nat → ParentExpr logicalWidth publicFits
  point : Nat → Fin productionShape.cubeVariables → KExpr
  message : Nat → Radix.ChildIndex → ChildMessageExpr

abbrev Attempt (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :=
  PiDEC.PaperVerifier.Attempt
    (PaperAlgebra.Structure logicalWidth)
    (PaperAlgebra.PublicInput
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    PaperAlgebra.Point PaperAlgebra.Evaluation PaperAlgebra.Commitment
    productionGlobalParams

def relationSource
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    PaperAlgebra.Structure logicalWidth :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.InputBinding.relationSource relation

def evalPoint
    (point : Fin productionShape.cubeVariables → KExpr) (env : Env) :
    PaperAlgebra.Point :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.InputBinding.evalPoint point env

def evalParent
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (parent : ParentExpr logicalWidth publicFits)
    (point : Fin productionShape.cubeVariables → KExpr)
    (env : Env) :
    CE.Instance (PaperAlgebra.Structure logicalWidth)
      (PaperAlgebra.PublicInput
        (logicalWidth := logicalWidth) (publicFits := publicFits))
      PaperAlgebra.Point PaperAlgebra.Evaluation PaperAlgebra.Commitment where
  constraintSystem := relationSource relation
  commitment := fun row coefficient =>
    (parent.commitment row coefficient).eval env
  publicInput := fun column => (parent.publicInput column).eval env
  point := evalPoint point env
  evaluations := #[
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.evalEvaluation
      parent.evaluation env]
  stage := .combined

def evalMessage (message : ChildMessageExpr) (env : Env) :
    PiDEC.PaperVerifier.ChildMessage
      PaperAlgebra.Evaluation PaperAlgebra.Commitment where
  commitment := fun row coefficient =>
    (message.commitment row coefficient).eval env
  evaluations := #[
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.evalEvaluation
      message.evaluation env]

def evalAttempt
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) : Attempt logicalWidth publicFits where
  parent := evalParent relation (interface.parent offset)
    (interface.point offset) env
  messages := fun child => evalMessage (interface.message offset child) env

structure StructuralHolds
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) : Prop where
  parentCombined :
    (evalAttempt relation interface offset env).parent.stage = .combined
  parentEvaluationSize :
    (evalAttempt relation interface offset env).parent.evaluations.size = 1
  messageEvaluationSize : ∀ child,
    ((evalAttempt relation interface offset env).messages child).evaluations.size = 1

abbrev SpecHolds
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) : Prop :=
  StructuralHolds relation interface offset env

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
    exact ⟨rfl, rfl, fun _ => rfl⟩
  completeness := by
    intro env _ _ _
    refine ⟨env, ?_, ?_⟩
    · intro _ _
      rfl
    · exact fun _ member => by cases member

theorem soundness
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits)
    (env : Env) (offset : Nat)
    (rows : holds env (Circuit.ops (circuit relation interface).main offset)) :
    SpecHolds relation interface offset env :=
  (circuit relation interface).soundness env offset trivial rows

theorem completeness
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits)
    (env : Env) (offset : Nat)
    (specification : SpecHolds relation interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (circuit relation interface).main offset)) ∧
      holdsFlat completed
        (Circuit.ops (circuit relation interface).main offset) :=
  (circuit relation interface).completeness env offset trivial specification

theorem accepted_parentEvaluationSize
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (specification : SpecHolds relation interface offset env) :
    (evalAttempt relation interface offset env).parent.evaluations.size =
      (PaperAlgebra.evaluationArity ajtai).count
        (evalAttempt relation interface offset env).parent.constraintSystem := by
  exact specification.parentEvaluationSize

theorem accepted_messageEvaluationSize
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (specification : SpecHolds relation interface offset env) :
    ∀ child,
      ((evalAttempt relation interface offset env).messages child).evaluations.size =
        (PaperAlgebra.evaluationArity ajtai).count
          (evalAttempt relation interface offset env).parent.constraintSystem := by
  exact specification.messageEvaluationSize

@[simp] theorem childFresh
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) (child : Radix.ChildIndex) :
    (PiDEC.PaperVerifier.children (PaperAlgebra.publicInputSplit ajtai)
      (evalAttempt relation interface offset env) child).stage = .fresh := by
  rfl

@[simp] theorem childSameStructure
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) (child : Radix.ChildIndex) :
    (PiDEC.PaperVerifier.children (PaperAlgebra.publicInputSplit ajtai)
        (evalAttempt relation interface offset env) child).constraintSystem =
      (evalAttempt relation interface offset env).parent.constraintSystem := by
  rfl

@[simp] theorem childSamePoint
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) (child : Radix.ChildIndex) :
    (PiDEC.PaperVerifier.children (PaperAlgebra.publicInputSplit ajtai)
        (evalAttempt relation interface offset env) child).point =
      (evalAttempt relation interface offset env).parent.point := by
  rfl

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

end NightstreamFPrime.Lifecycle.PiDEC.v1_1.InputBinding
