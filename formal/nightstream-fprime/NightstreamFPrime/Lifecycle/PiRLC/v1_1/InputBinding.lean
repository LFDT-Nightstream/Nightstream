import NightstreamFPrime.Circuit.Quadratic
import NightstreamFPrime.Lifecycle.ProductionKey
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption
import NightstreamFPrime.Spec.Folding.PiRLC.v1_1.InputBinding

/-!
Paper authority: SuperNeo v1.1, Section 7.4, PiRLC input and output.
Obligation: Bind the 17 PiCCS output claims, in `K+k` order, as fresh PiRLC
inputs for one production structure and one verifier-derived point.

Inputs:
- one shared point;
- 17 commitments and packed public inputs;
- 17 separate `Eval_K` families;
- 17 families of 14 `Eval_A` values.

Outputs:
- the same symbolic values viewed as the PiRLC input vector.

Constraint groups:
- C1: none; source and PiRLC views use the same expressions.

Parent coverage:
- `PiRLC.Equations.inputFresh`;
- `PiRLC.Equations.sameStructure`;
- `PiRLC.Equations.samePoint`.

The Stage 1 assembler owns the later PiCCS-output-to-this-interface wiring.
-/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_1.InputBinding

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-- One symbolic CE claim. The point is shared by the parent interface. -/
structure InputExpr (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) where
  commitment : Fin productionProfile.commitmentWidth → Fin ringDegree → Expr
  publicInput : Fin (FullShape logicalWidth publicFits).publicWidth → Expr
  evaluation :
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.EvaluationExpr

/-- One shared symbolic carrier for all 17 exact PiRLC inputs. -/
structure Interface (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) where
  point : Nat → Fin productionShape.cubeVariables → KExpr
  input : Nat → Fin productionShape.sourceCount →
    InputExpr logicalWidth publicFits

/-- The same computable relation source selected by the production key. -/
def relationSource
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    PaperAlgebra.Structure logicalWidth where
  cubeLayout :=
    NightstreamFPrime.Spec.Folding.PiCCS.CanonicalRowLayout.layout
      cubeVariables (Phi81CarrierLayout.carrierWidth logicalWidth)
      relation.cubeFits
  matrixSource := PaperAlgebra.matrixSource relation.system

abbrev InputInstance (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :=
  CE.Instance (PaperAlgebra.Structure logicalWidth)
    (PaperAlgebra.PublicInput
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    PaperAlgebra.Point PaperAlgebra.Evaluation PaperAlgebra.Commitment

def evalPoint
    (point : Fin productionShape.cubeVariables → KExpr) (env : Env) :
    PaperAlgebra.Point :=
  NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.evalPoint point env

def evalInput
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (input : InputExpr logicalWidth publicFits)
    (point : Fin productionShape.cubeVariables → KExpr)
    (env : Env) : InputInstance logicalWidth publicFits where
  constraintSystem := relationSource relation
  commitment := fun row coefficient =>
    (input.commitment row coefficient).eval env
  publicInput := fun column => (input.publicInput column).eval env
  point := evalPoint point env
  evaluations := #[
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.evalEvaluation
      input.evaluation env]
  stage := .fresh

def evalInputs
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) :
    Fin productionShape.sourceCount → InputInstance logicalWidth publicFits :=
  fun source => evalInput relation (interface.input offset source)
    (interface.point offset) env

/-- Symbolic form of the exact three-field PiRLC input-binding predicate. -/
def SpecHolds
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) : Prop :=
  NightstreamFPrime.Spec.Folding.PiRLC.v1_1.InputBinding.Holds
    (evalInputs relation interface offset env)
    (relationSource relation) (evalPoint (interface.point offset) env)

/-- The sole logical circuit for the input-binding leaf. -/
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
    exact ⟨fun _ => rfl, fun _ => rfl, fun _ => rfl⟩
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

/-- The leaf predicate is definitionally the exact parent relation prefix. -/
theorem parentCoverage
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (specification : SpecHolds relation interface offset env) :
    NightstreamFPrime.Spec.Folding.PiRLC.v1_1.InputBinding.Holds
      (evalInputs relation interface offset env)
      (relationSource relation) (evalPoint (interface.point offset) env) :=
  specification

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
      expression.VarsBelow
        (offset + localLength
          (Circuit.ops (circuit relation interface).main offset)) := by
  intro _ member
  cases member

end NightstreamFPrime.Lifecycle.PiRLC.v1_1.InputBinding
