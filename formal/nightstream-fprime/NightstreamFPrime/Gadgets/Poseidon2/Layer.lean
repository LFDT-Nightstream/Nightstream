import Mathlib.Data.List.GetD
import Mathlib.Data.List.OfFn
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.IntervalCases
import NightstreamFPrime.Circuit.StraightLine
import NightstreamFPrime.Spec.Poseidon2

/-!
Owns the fixed-width symbolic Poseidon2 layer formulas. It connects expression
evaluation to the executable field reference one lane at a time. No circuit
schedule or physical row layout is owned here.
-/

namespace NightstreamFPrime.Gadgets.Poseidon2.Layer

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit

abbrev EState := Fin 8 → Expr
abbrev FState := Fin 8 → F

def evalState (env : Env) (state : EState) : FState :=
  fun lane => (state lane).eval env

def getE (state : EState) (index : Nat) : Expr :=
  if h : index < 8 then state ⟨index, h⟩ else 0

def getF (state : FState) (index : Nat) : F :=
  if h : index < 8 then state ⟨index, h⟩ else 0

def sboxE (value : Expr) : Expr :=
  let square := value * value
  let fourth := square * square
  fourth * square * value

def sboxF (value : F) : F :=
  Spec.Poseidon2.sbox value

def mat4E (state : EState) (base lane : Nat) : Expr :=
  match lane with
  | 0 => 2 * getE state base + 3 * getE state (base + 1) +
      getE state (base + 2) + getE state (base + 3)
  | 1 => getE state base + 2 * getE state (base + 1) +
      3 * getE state (base + 2) + getE state (base + 3)
  | 2 => getE state base + getE state (base + 1) +
      2 * getE state (base + 2) + 3 * getE state (base + 3)
  | _ => 3 * getE state base + getE state (base + 1) +
      getE state (base + 2) + 2 * getE state (base + 3)

def mat4F (state : FState) (base lane : Nat) : F :=
  match lane with
  | 0 => 2 * getF state base + 3 * getF state (base + 1) +
      getF state (base + 2) + getF state (base + 3)
  | 1 => getF state base + 2 * getF state (base + 1) +
      3 * getF state (base + 2) + getF state (base + 3)
  | 2 => getF state base + getF state (base + 1) +
      2 * getF state (base + 2) + 3 * getF state (base + 3)
  | _ => 3 * getF state base + getF state (base + 1) +
      getF state (base + 2) + 2 * getF state (base + 3)

def blockE (state : EState) (index : Nat) : Expr :=
  mat4E state (if index < 4 then 0 else 4) (index % 4)

def blockF (state : FState) (index : Nat) : F :=
  mat4F state (if index < 4 then 0 else 4) (index % 4)

def externalE (state : EState) : EState :=
  fun lane => blockE state lane.val + blockE state (lane.val % 4) +
    blockE state (lane.val % 4 + 4)

def externalF (state : FState) : FState :=
  fun lane => blockF state lane.val + blockF state (lane.val % 4) +
    blockF state (lane.val % 4 + 4)

def sumE (state : EState) : Expr :=
  getE state 0 + getE state 1 + getE state 2 + getE state 3 +
    getE state 4 + getE state 5 + getE state 6 + getE state 7

def sumF (state : FState) : F :=
  getF state 0 + getF state 1 + getF state 2 + getF state 3 +
    getF state 4 + getF state 5 + getF state 6 + getF state 7

def internalE (state : EState) : EState :=
  fun lane => Expr.const (Spec.Poseidon2.ofNat
    (Spec.Poseidon2.internalDiagonal.getD lane.val 0)) * state lane + sumE state

def internalF (state : FState) : FState :=
  fun lane => Spec.Poseidon2.ofNat
    (Spec.Poseidon2.internalDiagonal.getD lane.val 0) * state lane + sumF state

def fullE (rows : List (List Nat)) (round : Nat) (state : EState) : EState :=
  externalE fun lane => sboxE
    (state lane + Expr.const (Spec.Poseidon2.constantAt rows round lane.val))

def fullF (rows : List (List Nat)) (round : Nat) (state : FState) : FState :=
  externalF fun lane => sboxF
    (state lane + Spec.Poseidon2.constantAt rows round lane.val)

def partialE (round : Nat) (state : EState) : EState :=
  internalE fun lane =>
    if lane.val = 0 then sboxE (state lane + Expr.const (Spec.Poseidon2.ofNat
      (Spec.Poseidon2.internalConstants.getD round 0)))
    else state lane

def partialF (round : Nat) (state : FState) : FState :=
  internalF fun lane =>
    if lane.val = 0 then sboxF (state lane + Spec.Poseidon2.ofNat
      (Spec.Poseidon2.internalConstants.getD round 0))
    else state lane

@[simp] theorem eval_getE (env : Env) (state : EState) (index : Nat) :
    (getE state index).eval env = getF (evalState env state) index := by
  simp only [getE, getF, evalState]
  split <;> rfl

@[simp] theorem eval_sboxE (env : Env) (value : Expr) :
    (sboxE value).eval env = sboxF (value.eval env) := by
  simp [sboxE, sboxF, Spec.Poseidon2.sbox]

@[simp] theorem eval_two (env : Env) : (2 : Expr).eval env = (2 : F) := rfl
@[simp] theorem eval_three (env : Env) : (3 : Expr).eval env = (3 : F) := rfl

@[simp] theorem eval_mat4E (env : Env) (state : EState) (base lane : Nat) :
    (mat4E state base lane).eval env = mat4F (evalState env state) base lane := by
  rcases lane with _ | _ | _ | lane <;>
    simp [mat4E, mat4F, Expr.eval_add, Expr.eval_mul]

@[simp] theorem eval_externalE (env : Env) (state : EState)
    (lane : Fin 8) :
    (externalE state lane).eval env = externalF (evalState env state) lane := by
  simp [externalE, externalF, blockE, blockF]

@[simp] theorem eval_sumE (env : Env) (state : EState) :
    (sumE state).eval env = sumF (evalState env state) := by
  simp [sumE, sumF]

@[simp] theorem eval_internalE (env : Env) (state : EState)
    (lane : Fin 8) :
    (internalE state lane).eval env = internalF (evalState env state) lane := by
  simp [internalE, internalF, evalState]

@[simp] theorem eval_fullE (env : Env) (rows : List (List Nat)) (round : Nat)
    (state : EState) (lane : Fin 8) :
    (fullE rows round state lane).eval env = fullF rows round (evalState env state) lane := by
  unfold fullE fullF
  rw [eval_externalE]
  apply congrFun (congrArg externalF ?_) lane
  funext index
  simp [evalState]

@[simp] theorem eval_partialE (env : Env) (round : Nat) (state : EState)
    (lane : Fin 8) :
    (partialE round state lane).eval env = partialF round (evalState env state) lane := by
  unfold partialE partialF
  rw [eval_internalE]
  apply congrFun (congrArg internalF ?_) lane
  funext index
  by_cases hzero : index.val = 0
  · simp [evalState, hzero]
  · simp [evalState, hzero]

private theorem ofFn_state {α : Type} (state : Fin 8 → α) :
    List.ofFn state =
      [state 0, state 1, state 2, state 3, state 4, state 5, state 6, state 7] := by
  simp [Spec.Poseidon2.width, List.ofFn_succ]

theorem externalF_eq_reference (state : FState) :
    List.ofFn (externalF state) = Spec.Poseidon2.externalLayer (List.ofFn state) := by
  rw [ofFn_state (externalF state), ofFn_state state]
  simp [externalF, blockF, mat4F, getF, Spec.Poseidon2.externalLayer,
    Spec.Poseidon2.mat4, Spec.Poseidon2.width, List.range_succ]

theorem internalF_eq_reference (state : FState) :
    List.ofFn (internalF state) = Spec.Poseidon2.internalLayer (List.ofFn state) := by
  rw [ofFn_state (internalF state), ofFn_state state]
  simp [internalF, sumF, getF, Spec.Poseidon2.internalLayer,
    Spec.Poseidon2.width, List.range_succ]

theorem fullF_eq_reference (rows : List (List Nat)) (round : Nat) (state : FState) :
    List.ofFn (fullF rows round state) =
      Spec.Poseidon2.fullRound rows round (List.ofFn state) := by
  unfold fullF Spec.Poseidon2.fullRound
  rw [externalF_eq_reference]
  congr 1

theorem partialF_eq_reference (round : Nat) (state : FState) :
    List.ofFn (partialF round state) =
      Spec.Poseidon2.partialRound round (List.ofFn state) := by
  unfold partialF Spec.Poseidon2.partialRound
  rw [internalF_eq_reference]
  congr 1

end NightstreamFPrime.Gadgets.Poseidon2.Layer
