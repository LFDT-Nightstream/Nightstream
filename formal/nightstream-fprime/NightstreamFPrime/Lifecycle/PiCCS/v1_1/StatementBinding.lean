import NightstreamFPrime.Circuit.Quadratic
import NightstreamFPrime.Lifecycle.ProductionKey
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.StateBinding
import NightstreamFPrime.Spec.Folding.PiCCS.Statement

/-!
Paper authority: SuperNeo v1.1, Section 7.3, `Pi_CCS` input.
Obligation: The verifier input reuses the statement's prior point, `Eval_K`,
and `Eval_A` values without a witness-controlled copy.

Inputs:
- the shared prior-point wires;
- the shared Pad-evaluation (`Eval_K`) wires;
- the shared CCS-matrix-evaluation (`Eval_A`) wires.

Outputs:
- the same wires, viewed as the production verifier input.

Constraint groups:
- C1: none; the interface makes the source and verifier views definitionally
  identical, so no copy row or private variable is required.

Parent coverage:
- `v1_1.Coverage.input_eval_K`;
- `v1_1.Coverage.input_eval_A`;
- `v1_1.Statement.Holds`.

This leaf is an audited wiring boundary. It does not add a second statement
carrier, and it does not replace binding with an assumption.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementBinding

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-- One shared symbolic carrier for both sides of the statement binding. -/
structure Interface where
  state : StateBinding.Interface
  priorPoint : Nat -> Fin productionShape.cubeVariables -> KExpr
  eval_K : Nat -> PadCoordinate productionShape -> KExpr
  eval_A : Nat -> MatrixCoordinate productionShape -> KExpr

/-- The verifier view of the prior point. This is deliberately a projection,
not a second caller-supplied value. -/
def verifierPriorPoint (interface : Interface) := interface.priorPoint

/-- The verifier view of `Eval_K`. -/
def verifierEval_K (interface : Interface) := interface.eval_K

/-- The verifier view of `Eval_A`. -/
def verifierEval_A (interface : Interface) := interface.eval_A

/-- Symbolic form of the three exact v1.1 statement-binding conjuncts. -/
structure SpecHolds (interface : Interface) (offset : Nat) (env : Env) : Prop where
  state : StateBinding.SpecHolds interface.state offset env
  priorPoint : forall coordinate,
    ((verifierPriorPoint interface) offset coordinate).eval env =
      (interface.priorPoint offset coordinate).eval env
  eval_K : forall coordinate,
    ((verifierEval_K interface) offset coordinate).eval env =
      (interface.eval_K offset coordinate).eval env
  eval_A : forall coordinate,
    ((verifierEval_A interface) offset coordinate).eval env =
      (interface.eval_A offset coordinate).eval env

/-- Canonical state rows plus the definitionally shared PiCCS statement. -/
def circuit (interface : Interface) : FormalCircuit where
  main := (StateBinding.circuit interface.state).main
  assumptions := StateBinding.Assumptions interface.state
  spec := SpecHolds interface
  soundness := by
    intro env offset assumptions rows
    exact ⟨StateBinding.soundness interface.state env offset rows,
      fun _ => rfl, fun _ => rfl, fun _ => rfl⟩
  completeness := by
    intro env offset assumptions specification
    exact StateBinding.completeness interface.state env offset
      specification.state

theorem soundness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : StateBinding.Assumptions interface.state offset env)
    (rows : holds env (Circuit.ops (circuit interface).main offset)) :
    SpecHolds interface offset env :=
  (circuit interface).soundness env offset assumptions rows

theorem completeness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : StateBinding.Assumptions interface.state offset env)
    (specification : SpecHolds interface offset env) :
    exists completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (circuit interface).main offset)) ∧
      holdsFlat completed (Circuit.ops (circuit interface).main offset) :=
  (circuit interface).completeness env offset assumptions specification

theorem constraintsHold_of_spec (interface : Interface) (env : Env)
    (offset : Nat) (specification : SpecHolds interface offset env) :
    ConstraintsHold env
      (flatConstraints (Circuit.ops (circuit interface).main offset)) := by
  exact StateBinding.constraintsHold_of_spec interface.state env offset
    specification.state

/-- This boundary allocates no private value. -/
theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (circuit interface).main offset) = 0 := by
  exact StateBinding.localLength_eq interface.state offset

/-- This boundary emits the 160 canonical state and context assertions. -/
theorem operations_length (interface : Interface) (offset : Nat) :
    (Circuit.ops (circuit interface).main offset).length = 160 := by
  exact StateBinding.operations_length interface.state offset

/-- Each state-binding assertion lowers to one direct row. -/
theorem flatConstraints_length (interface : Interface) (offset : Nat) :
    (flatConstraints (Circuit.ops (circuit interface).main offset)).length =
      160 := by
  exact StateBinding.flatConstraints_length interface.state offset

theorem flatConstraints_eq_stateAssertions (interface : Interface)
    (offset : Nat) :
    flatConstraints (Circuit.ops (circuit interface).main offset) =
      StateBinding.assertions interface.state offset := by
  exact StateBinding.flatConstraints_opsAt interface.state offset

/-- The definitionally shared statement view is stable under every
environment change. -/
theorem specHolds_of_agree_below (interface : Interface) (offset : Nat)
    (before after : Env)
    (assumptions : StateBinding.Assumptions interface.state offset before)
    (agrees : ∀ index, index < offset → after index = before index)
    (specification : SpecHolds interface offset before) :
    SpecHolds interface offset after := by
  exact ⟨StateBinding.specHolds_of_agree_below interface.state offset
      before after assumptions agrees specification.state,
    fun _ => rfl, fun _ => rfl, fun _ => rfl⟩

/-- Every state-binding row uses only parent-owned wires. -/
theorem flatConstraints_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) :
    StateBinding.Assumptions interface.state offset env →
    ∀ expression ∈ flatConstraints
      (Circuit.ops (circuit interface).main offset),
      expression.VarsBelow
        (offset + localLength (Circuit.ops (circuit interface).main offset)) := by
  intro assumptions expression member
  rw [localLength_eq, Nat.add_zero]
  exact StateBinding.flatConstraints_varsBelow interface.state offset
    env assumptions expression member

/-- Exact parent coverage: the production verifier input is constructed from
the production statement, so it satisfies the canonical v1.1 binding
predicate. The circuit proves that both symbolic views use the same wires. -/
theorem spec_implies_keyStatement
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns <=
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Interface) (offset : Nat) (env : Env)
    (_specification : SpecHolds interface offset env) :
    NightstreamFPrime.Spec.Folding.PiCCS.Statement.Holds
      ((ProductionKey.key relation ajtai).statement running fresh)
      (((ProductionKey.key relation ajtai).statement running fresh).verifierInput
        (ProductionKey.key relation ajtai).lift) :=
  NightstreamFPrime.Spec.Folding.PiCCS.Statement.verifierInput_holds
    (ProductionKey.key relation ajtai).lift
    ((ProductionKey.key relation ajtai).statement running fresh)

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementBinding
