import NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal
import NightstreamFPrime.Lifecycle.Transcript

/-!
Owns the Π_RLC scalar-domain entry action.

For coordinate `i`, the verifier absorbs exactly `[4, i]` from the incoming
post-Π_CCS transcript state. The generic owned Duplex circuit owns the one
Poseidon2 permutation. This leaf owns only the domain words, order, named
semantics, and direct outgoing-state wiring.

The coordinate is a compile-time verifier index. It is not a witness or a
public input. The leaf adds no boundary-copy or assertion row.
-/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex

abbrev State := NightstreamFPrime.Lifecycle.Transcript.State
abbrev EState := Layer.EState

structure Interface where
  initialState : Nat → EState

def frameWords (coordinate : Nat) : List F :=
  [NightstreamFPrime.Lifecycle.natWord 4,
    NightstreamFPrime.Lifecycle.natWord coordinate]

def constantWords (words : List F) : List Expr :=
  words.map Expr.const

def actions (coordinate : Nat) : List Formal.Action :=
  [.absorb (constantWords (frameWords coordinate))]

def ownedInterface (interface : Interface) (coordinate : Nat) :
    Formal.Owned.Interface where
  initial := interface.initialState
  actions := fun _ => actions coordinate

def output (interface : Interface) (coordinate offset : Nat) : EState :=
  Formal.Owned.output (ownedInterface interface coordinate) offset

def evalState (env : Env) (state : EState) : State :=
  List.ofFn (Layer.evalState env state)

def Assumptions (interface : Interface) (offset : Nat) (_env : Env) : Prop :=
  ∀ lane, (interface.initialState offset lane).VarsBelow offset

/-- Exact verifier-owned scalar-domain entry. -/
def SpecHolds (interface : Interface) (coordinate offset : Nat)
    (env : Env) : Prop :=
  NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.enterScalar
      (evalState env (interface.initialState offset)) coordinate =
    evalState env (output interface coordinate offset)

private theorem eval_constantWords (env : Env) (words : List F) :
    Hash.evalList env (constantWords words) = words := by
  simp [Hash.evalList, constantWords, Function.comp_def]

private theorem actions_below (coordinate bound : Nat) :
    Formal.ActionsBelow bound (actions coordinate) := by
  intro action member
  simp only [actions, List.mem_singleton] at member
  subst action
  intro expression expressionMember
  rcases List.mem_map.mp expressionMember with ⟨word, _, rfl⟩
  exact trivial

private theorem ownedAssumptions (interface : Interface) (coordinate : Nat)
    (offset : Nat) {env : Env}
    (assumptions : Assumptions interface offset env) :
    Formal.Owned.Assumptions (ownedInterface interface coordinate) offset env :=
  ⟨assumptions, actions_below coordinate offset⟩

theorem ownedSpec_iff_specHolds (interface : Interface)
    (coordinate offset : Nat) (env : Env) :
    Formal.Owned.SpecHolds (ownedInterface interface coordinate) offset env ↔
      SpecHolds interface coordinate offset env := by
  unfold Formal.Owned.SpecHolds SpecHolds
  simp only [ownedInterface, actions, List.map_singleton,
    Formal.Action.eval, Formal.TraceHolds]
  rw [eval_constantWords]
  unfold NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.enterScalar
  unfold NightstreamFPrime.Lifecycle.Transcript.absorb
    Absorb.reference Hash.inputChunks frameWords
  rfl

/-- The sole logical circuit for scalar-domain entry. -/
def circuit (interface : Interface) (coordinate : Nat) : FormalCircuit where
  main := Formal.Owned.main (ownedInterface interface coordinate)
  assumptions := Assumptions interface
  spec := SpecHolds interface coordinate
  soundness := by
    intro env offset assumptions rows
    have owned := Formal.Owned.soundness
      (ownedInterface interface coordinate) env offset
      (ownedAssumptions interface coordinate offset assumptions) rows
    exact (ownedSpec_iff_specHolds interface coordinate offset env).mp owned
  completeness := by
    intro env offset assumptions specification
    apply Formal.Owned.completeness
      (ownedInterface interface coordinate) env offset
      (ownedAssumptions interface coordinate offset assumptions)
    exact (ownedSpec_iff_specHolds interface coordinate offset env).mpr
      specification

@[simp] private theorem circuit_ops (interface : Interface)
    (coordinate offset : Nat) :
    Circuit.ops (circuit interface coordinate).main offset =
      Formal.Owned.opsAt (ownedInterface interface coordinate) offset := by
  rfl

theorem soundness (interface : Interface) (coordinate : Nat)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (rows : holds env
      (Circuit.ops (circuit interface coordinate).main offset)) :
    SpecHolds interface coordinate offset env :=
  (circuit interface coordinate).soundness env offset assumptions rows

/-- The absorb-only scalar-entry schedule has a deterministic witness builder
that needs no semantic output premise. -/
theorem complete (interface : Interface) (coordinate : Nat)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength
          (Circuit.ops (circuit interface coordinate).main offset)) ∧
      holdsFlat completed
        (Circuit.ops (circuit interface coordinate).main offset) := by
  apply Formal.Owned.build_of_no_assertions
    (ownedInterface interface coordinate) env offset
    (ownedAssumptions interface coordinate offset assumptions)
  rfl

theorem flatConstraints_varsBelow (interface : Interface) (coordinate : Nat)
    (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env) :
    ∀ expression ∈
        flatConstraints (Circuit.ops (circuit interface coordinate).main offset),
      expression.VarsBelow
        (offset + localLength
          (Circuit.ops (circuit interface coordinate).main offset)) := by
  exact Formal.Owned.flatConstraints_varsBelow
    (ownedInterface interface coordinate) offset env
      (ownedAssumptions interface coordinate offset assumptions)

theorem completeness (interface : Interface) (coordinate : Nat)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface coordinate offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength
          (Circuit.ops (circuit interface coordinate).main offset)) ∧
      holdsFlat completed
        (Circuit.ops (circuit interface coordinate).main offset) :=
  (circuit interface coordinate).completeness env offset assumptions
    specification

private theorem recipeCount_eq (coordinate : Nat) :
    Formal.recipeCount (actions coordinate) = 592 := by
  norm_num [actions, constantWords, frameWords, Formal.recipeCount,
    Formal.Action.recipeCount, Hash.inputChunks, Poseidon2.rate]

private theorem assertionCount_eq (coordinate : Nat) :
    Formal.assertionCount (actions coordinate) = 0 := by
  rfl

/-- Exact private allocation: one Poseidon2 permutation. -/
theorem localLength_eq (interface : Interface) (coordinate offset : Nat) :
    localLength
      (Circuit.ops (circuit interface coordinate).main offset) = 592 := by
  rw [circuit_ops, Formal.Owned.opsAt_localLength]
  change (Formal.compile offset (interface.initialState offset)
    (actions coordinate)).recipes.length = 592
  rw [Formal.compile_recipes_length]
  exact recipeCount_eq coordinate

/-- One witness batch and no assertion or copy operation. -/
theorem operations_length (interface : Interface) (coordinate offset : Nat) :
    (Circuit.ops (circuit interface coordinate).main offset).length = 1 := by
  rw [circuit_ops, Formal.Owned.operations_length]
  change 1 + Formal.assertionCount (actions coordinate) = 1
  rw [assertionCount_eq]

/-- Exact logical rows: one row for each of the 592 recipes. -/
theorem flatConstraints_length (interface : Interface)
    (coordinate offset : Nat) :
    (flatConstraints
      (Circuit.ops (circuit interface coordinate).main offset)).length = 592 := by
  rw [circuit_ops, Formal.Owned.flatConstraints_length]
  change Formal.recipeCount (actions coordinate) +
    Formal.assertionCount (actions coordinate) = 592
  rw [recipeCount_eq, assertionCount_eq]

/-- The owned outgoing state lies inside the exact private interval. -/
theorem output_varsBelow (interface : Interface) (coordinate offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ lane, (output interface coordinate offset lane).VarsBelow
      (offset +
        localLength
          (Circuit.ops (circuit interface coordinate).main offset)) := by
  exact Formal.Owned.output_varsBelow
    (ownedInterface interface coordinate) offset env
    (ownedAssumptions interface coordinate offset assumptions)

end NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption
