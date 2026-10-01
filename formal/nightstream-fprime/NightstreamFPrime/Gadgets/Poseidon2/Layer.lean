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

/-- Lane states use the literal width so that `omega` and `decide` see it. -/
abbrev EState := Fin 16 → Expr
abbrev FState := Fin 16 → F

theorem width_eq : Spec.Poseidon2.width = 16 := rfl

def evalState (env : Env) (state : EState) : FState :=
  fun lane => (state lane).eval env

def getE (state : EState) (index : Nat) : Expr :=
  if h : index < 16 then state ⟨index, h⟩ else 0

def getF (state : FState) (index : Nat) : F :=
  if h : index < 16 then state ⟨index, h⟩ else 0

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

/-- `M₄` applied to the four-lane block that contains `index`. -/
def blockE (state : EState) (index : Nat) : Expr :=
  mat4E state (4 * (index / 4)) (index % 4)

def blockF (state : FState) (index : Nat) : F :=
  mat4F state (4 * (index / 4)) (index % 4)

/-- Sum over all blocks of the block lanes congruent to `index` mod 4. -/
def columnE (state : EState) (index : Nat) : Expr :=
  ((List.range 4).map fun block =>
    blockE state (4 * block + index % 4)).foldl (· + ·) 0

def columnF (state : FState) (index : Nat) : F :=
  ((List.range 4).map fun block =>
    blockF state (4 * block + index % 4)).foldl (· + ·) 0

def externalE (state : EState) : EState :=
  fun lane => blockE state lane.val + columnE state lane.val

def externalF (state : FState) : FState :=
  fun lane => blockF state lane.val + columnF state lane.val

def sumE (state : EState) : Expr :=
  ((List.range 16).map (getE state)).foldl (· + ·) 0

def sumF (state : FState) : F :=
  ((List.range 16).map (getF state)).foldl (· + ·) 0

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
    simp [mat4E, mat4F]

private theorem eval_foldl_add (env : Env) (values : List Expr) (initial : Expr) :
    (values.foldl (· + ·) initial).eval env =
      (values.map (Expr.eval env)).foldl (· + ·) (initial.eval env) := by
  induction values generalizing initial with
  | nil => rfl
  | cons value values inductionHypothesis =>
      simp only [List.foldl_cons, List.map_cons]
      rw [inductionHypothesis]
      rfl

@[simp] theorem eval_blockE (env : Env) (state : EState) (index : Nat) :
    (blockE state index).eval env = blockF (evalState env state) index := by
  simp [blockE, blockF]

@[simp] theorem eval_columnE (env : Env) (state : EState) (index : Nat) :
    (columnE state index).eval env = columnF (evalState env state) index := by
  unfold columnE columnF
  rw [eval_foldl_add, List.map_map]
  simp only [Function.comp_def, eval_blockE]
  rfl

@[simp] theorem eval_externalE (env : Env) (state : EState)
    (lane : Fin 16) :
    (externalE state lane).eval env = externalF (evalState env state) lane := by
  simp [externalE, externalF]

@[simp] theorem eval_sumE (env : Env) (state : EState) :
    (sumE state).eval env = sumF (evalState env state) := by
  unfold sumE sumF
  rw [eval_foldl_add, List.map_map]
  simp only [Function.comp_def, eval_getE]
  rfl

@[simp] theorem eval_internalE (env : Env) (state : EState)
    (lane : Fin 16) :
    (internalE state lane).eval env = internalF (evalState env state) lane := by
  simp [internalE, internalF, evalState]

@[simp] theorem eval_fullE (env : Env) (rows : List (List Nat)) (round : Nat)
    (state : EState) (lane : Fin 16) :
    (fullE rows round state lane).eval env = fullF rows round (evalState env state) lane := by
  unfold fullE fullF
  rw [eval_externalE]
  apply congrFun (congrArg externalF ?_) lane
  funext index
  simp [evalState]

@[simp] theorem eval_partialE (env : Env) (round : Nat) (state : EState)
    (lane : Fin 16) :
    (partialE round state lane).eval env = partialF round (evalState env state) lane := by
  unfold partialE partialF
  rw [eval_internalE]
  apply congrFun (congrArg internalF ?_) lane
  funext index
  by_cases hzero : index.val = 0
  · simp [evalState, hzero]
  · simp [evalState, hzero]

theorem ofFn_state {α : Type} (state : Fin 16 → α) :
    List.ofFn state =
      [state 0, state 1, state 2, state 3, state 4, state 5, state 6, state 7,
        state 8, state 9, state 10, state 11, state 12, state 13, state 14,
        state 15] := by
  simp [List.ofFn_succ]

theorem externalF_eq_reference (state : FState) :
    List.ofFn (externalF state) = Spec.Poseidon2.externalLayer (List.ofFn state) := by
  rw [ofFn_state (externalF state), ofFn_state state]
  simp [externalF, blockF, columnF, mat4F, getF, Spec.Poseidon2.externalLayer,
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
