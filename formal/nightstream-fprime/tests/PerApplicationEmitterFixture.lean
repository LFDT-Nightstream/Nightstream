import NightstreamFPrime.Export.Stage1.PerApplicationFixedPoint

/-!
Test-only application used to compare the generic streamed per-application
emitter with the canonical materialized codec value.

This module does not select a production application. The circuit proves the
four-word identity transition, has no witness words, and allocates no local
variables.
-/

namespace NightstreamFPrime.Tests.PerApplicationEmitterFixture

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.Stage1
open NightstreamFPrime.Spec

abbrev Interface := Application.Interface 0

def step (input : AppState) (_witness : AppWitness) : AppState :=
  input

def assertions (interface : Interface) (offset : Nat) : List Expr :=
  List.ofFn fun index : Application.StateIndex =>
    interface.output offset index - interface.input offset index

def opsAt (interface : Interface) (offset : Nat) : List Op :=
  (assertions interface offset).map Op.assertZero

def main (interface : Interface) : Circuit Unit := fun offset =>
  ((), offset, opsAt interface offset)

@[simp] theorem main_ops (interface : Interface) (offset : Nat) :
    Circuit.ops (main interface) offset = opsAt interface offset := by
  rfl

private theorem flatConstraints_assertions (expressions : List Expr) :
    flatConstraints (expressions.map Op.assertZero) = expressions := by
  induction expressions with
  | nil => rfl
  | cons expression rest inductionHypothesis =>
      change [expression] ++ flatConstraints (rest.map Op.assertZero) =
        expression :: rest
      rw [inductionHypothesis]
      rfl

@[simp] theorem flatConstraints_opsAt (interface : Interface) (offset : Nat) :
    flatConstraints (opsAt interface offset) = assertions interface offset := by
  exact flatConstraints_assertions _

private theorem assertion_holds_iff (left right : Expr) (env : Env) :
    (left - right).eval env = 0 ↔ left.eval env = right.eval env := by
  constructor
  · intro row
    exact sub_eq_zero.mp (by simpa using row)
  · intro equal
    simpa using sub_eq_zero.mpr equal

def SpecHolds (interface : Interface) (offset : Nat) (env : Env) : Prop :=
  ∀ index,
    (interface.output offset index).eval env =
      (interface.input offset index).eval env

private theorem applicationHolds_iff_specHolds
    (interface : Interface) (offset : Nat) (env : Env) :
    Application.Holds step interface offset env ↔
      SpecHolds interface offset env := by
  unfold Application.Holds Application.outputState Application.inputState step
  rw [List.ofFn_inj]
  constructor
  · intro equal index
    exact congrFun equal index
  · intro equal
    exact funext equal

theorem soundness (interface : Interface) (env : Env) (offset : Nat)
    (rows : holds env (Circuit.ops (main interface) offset)) :
    SpecHolds interface offset env := by
  intro index
  have row := rows
    (Op.assertZero
      (interface.output offset index - interface.input offset index)) (by
        rw [main_ops, opsAt, List.mem_map]
        exact ⟨_, List.mem_ofFn.mpr ⟨index, rfl⟩, rfl⟩)
  exact (assertion_holds_iff _ _ env).mp row

theorem completeness (interface : Interface) (env : Env) (offset : Nat)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
          (localLength (Circuit.ops (main interface) offset)) ∧
        holdsFlat completed (Circuit.ops (main interface) offset) := by
  refine ⟨env, ?_, ?_⟩
  · intro _index _outside
    rfl
  · rw [main_ops]
    change ConstraintsHold env (flatConstraints (opsAt interface offset))
    rw [flatConstraints_opsAt]
    intro expression member
    rcases List.mem_ofFn.mp member with ⟨index, rfl⟩
    exact (assertion_holds_iff _ _ env).mpr (specification index)

private theorem localLength_assertions (expressions : List Expr) :
    localLength (expressions.map Op.assertZero) = 0 := by
  induction expressions with
  | nil => rfl
  | cons _ rest inductionHypothesis =>
      change 0 + localLength (rest.map Op.assertZero) = 0
      simpa using inductionHypothesis

@[simp] theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (main interface) offset) = 0 := by
  rw [main_ops, opsAt, localLength_assertions]

@[simp] theorem flatConstraints_length (interface : Interface) (offset : Nat) :
    (flatConstraints (Circuit.ops (main interface) offset)).length = 4 := by
  rw [main_ops, flatConstraints_opsAt]
  simp [assertions, Application.stateWordCount]

def circuit (interface : Interface) : FormalCircuit where
  main := main interface
  spec := SpecHolds interface
  soundness := by
    intro env offset _assumptions rows
    exact soundness interface env offset rows
  completeness := by
    intro env offset _assumptions specification
    exact completeness interface env offset specification

def program (_ : Unit) : Application.Program where
  witnessWordCount := 0
  step := step
  circuit := circuit
  spec_iff := by
    intro interface offset env
    exact (applicationHolds_iff_specHolds interface offset env).symm
  assumptions_of_inputsBelow := by
    intro _interface _offset _env _inputs
    trivial
  constraintsSupported := by
    intro interface offset _env allowed _assumptions inputs _localSupport
      expression member
    rw [circuit, main_ops, flatConstraints_opsAt] at member
    rcases List.mem_ofFn.mp member with ⟨index, rfl⟩
    exact Expr.VarsSatisfy.sub _ _ _ (inputs.output index)
      (inputs.input index)

@[simp] theorem applicationPlan_rowCount :
    (PerApplicationPackage.applicationPlan (program ())).rowCount = 4 := by
  rfl

@[simp] theorem addedPrivateColumnCount_eq :
    PerApplicationPackage.addedPrivateColumnCount (program ()) = 0 := by
  rfl

@[simp] theorem retainedLocalCount_eq :
    ApplicationRetainedBlocks.localCount (program ()) = 0 := by
  rfl

def fits (_ : Unit) : PerApplicationFixedPoint.FitsTwoPow28 (program ()) :=
  PerApplicationFixedPoint.fitsTwoPow28OfApplicationBounds (program ())
    (by simp) (by simp) (by
      change 0 ≤ _
      exact Nat.zero_le _)

end NightstreamFPrime.Tests.PerApplicationEmitterFixture
