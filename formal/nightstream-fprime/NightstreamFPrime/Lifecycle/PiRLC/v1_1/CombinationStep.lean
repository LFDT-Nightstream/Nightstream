import NightstreamFPrime.Circuit.StraightLine
import Mathlib.Logic.Equiv.Fin.Basic

/-!
Paper authority: SuperNeo v1.1, Section 7.4, verifier Step 1.
Obligation: one indexed source updates an accumulated family by the exact
Phi81 ring-module equation `next = prior + rho * value`.

The same child is instantiated for commitment, public-input, `Eval_K`, and
`Eval_A` families. `cellCount = 1` represents `RingF`; `cellCount = 2`
represents the two base-field cells of `RingK`. The child owns one fresh
output cell and one arithmetic row per block, ring lane, and coefficient cell.
It does not own transcript sampling, family ordering, or output binding.
-/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit

def privateCount (blockCount cellCount : Nat) : Nat :=
  blockCount * (ringDegree * cellCount)

structure Interface (blockCount cellCount : Nat) where
  challenge : Nat → Fin ringDegree → Expr
  prior : Nat → Fin blockCount → Fin ringDegree → Fin cellCount → Expr
  value : Nat → Fin blockCount → Fin ringDegree → Fin cellCount → Expr

def coordinates {blockCount cellCount : Nat}
    (index : Fin (privateCount blockCount cellCount)) :
    Fin blockCount × Fin ringDegree × Fin cellCount :=
  let outer : Fin (blockCount * (ringDegree * cellCount)) := index
  let outerPair := finProdFinEquiv.symm outer
  (outerPair.1, finProdFinEquiv.symm outerPair.2)

def indexOf {blockCount cellCount : Nat}
    (block : Fin blockCount) (lane : Fin ringDegree)
    (cell : Fin cellCount) : Fin (privateCount blockCount cellCount) :=
  finProdFinEquiv (block, finProdFinEquiv (lane, cell))

def cellOf {blockCount cellCount : Nat} [NeZero cellCount]
    (index : Fin (privateCount blockCount cellCount)) : Fin cellCount :=
  (coordinates index).2.2

def laneOf {blockCount cellCount : Nat} [NeZero cellCount]
    (index : Fin (privateCount blockCount cellCount)) : Fin ringDegree :=
  (coordinates index).2.1

def blockOf {blockCount cellCount : Nat} [NeZero cellCount]
    (index : Fin (privateCount blockCount cellCount)) : Fin blockCount :=
  (coordinates index).1

@[simp] theorem blockOf_indexOf
    {blockCount cellCount : Nat} [NeZero cellCount]
    (block : Fin blockCount) (lane : Fin ringDegree)
    (cell : Fin cellCount) :
    blockOf (indexOf block lane cell) = block := by
  simp [blockOf, coordinates, indexOf]

@[simp] theorem laneOf_indexOf
    {blockCount cellCount : Nat} [NeZero cellCount]
    (block : Fin blockCount) (lane : Fin ringDegree)
    (cell : Fin cellCount) :
    laneOf (indexOf block lane cell) = lane := by
  simp [laneOf, coordinates, indexOf]

@[simp] theorem cellOf_indexOf
    {blockCount cellCount : Nat} [NeZero cellCount]
    (block : Fin blockCount) (lane : Fin ringDegree)
    (cell : Fin cellCount) :
    cellOf (indexOf block lane cell) = cell := by
  simp [cellOf, coordinates, indexOf]

def output {blockCount cellCount : Nat} [NeZero cellCount]
    (offset : Nat) (index : Fin (privateCount blockCount cellCount)) : Expr :=
  Expr.var (offset + index.val)

def ringExpr
    {blockCount cellCount : Nat} [NeZero cellCount]
    (family : Nat → Fin blockCount → Fin ringDegree → Fin cellCount → Expr)
    (offset : Nat) (block : Fin blockCount) (cell : Fin cellCount) :
    Fin ringDegree → Expr :=
  fun lane => family offset block lane cell

def evalRing (env : Env) (value : Fin ringDegree → Expr) : RingF :=
  fun lane => (value lane).eval env

def exprCoeff (value : Fin ringDegree → Expr) (index : Nat) : Expr :=
  if indexLt : index < ringDegree then value ⟨index, indexLt⟩ else 0

def rawExpr (challenge value : Fin ringDegree → Expr)
    (degree : Nat) : Expr :=
  (List.range ringDegree).foldl (fun accumulated source =>
    if source ≤ degree ∧ degree - source < ringDegree then
      accumulated + exprCoeff challenge source *
        exprCoeff value (degree - source)
    else accumulated) 0

def mulExpr (challenge value : Fin ringDegree → Expr)
    (lane : Fin ringDegree) : Expr :=
  let folded := if lane.val < ringMiddleDegree then
      rawExpr challenge value (lane.val + ringDegree)
    else
      rawExpr challenge value (lane.val + ringMiddleDegree)
  let twice := if lane.val + 81 ≤ 106 then
      rawExpr challenge value (lane.val + 81)
    else 0
  rawExpr challenge value lane.val - folded + twice

def recipe {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat)
    (index : Fin (privateCount blockCount cellCount)) : Expr :=
  let block := blockOf index
  let lane := laneOf index
  let cell := cellOf index
  interface.prior offset block lane cell +
    mulExpr (interface.challenge offset)
      (ringExpr interface.value offset block cell) lane

def recipes {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat) : List Expr :=
  List.ofFn (recipe interface offset)

def operations {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat) : List Op :=
  [.witness (WitnessBatch.arithmetic offset (recipes interface offset))]

def main {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) : Circuit Unit := fun offset =>
  ((), offset + privateCount blockCount cellCount,
    operations interface offset)

structure Assumptions {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat)
    (_env : Env) : Prop where
  challengeBelow : ∀ lane, (interface.challenge offset lane).VarsBelow offset
  priorBelow : ∀ block lane cell,
    (interface.prior offset block lane cell).VarsBelow offset
  valueBelow : ∀ block lane cell,
    (interface.value offset block lane cell).VarsBelow offset

def SpecHolds {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat)
    (env : Env) : Prop :=
  ∀ index,
    (output offset index).eval env =
      (interface.prior offset (blockOf index) (laneOf index)
        (cellOf index)).eval env +
      ringFMul
        (fun lane => (interface.challenge offset lane).eval env)
        (fun lane =>
          (interface.value offset (blockOf index) lane
            (cellOf index)).eval env)
        (laneOf index)

private theorem foldl_eval
    (env : Env) (indices : List Nat) (degree : Nat)
    (challenge value : Fin ringDegree → Expr) (initial : Expr) :
    (indices.foldl (fun accumulated source =>
      if source ≤ degree ∧ degree - source < ringDegree then
        accumulated +
          exprCoeff challenge source * exprCoeff value (degree - source)
      else accumulated) initial).eval env =
    indices.foldl (fun accumulated source =>
      if source ≤ degree ∧ degree - source < ringDegree then
        accumulated +
          ringFCoeff (evalRing env challenge) source *
          ringFCoeff (evalRing env value) (degree - source)
      else accumulated) (initial.eval env) := by
  induction indices generalizing initial with
  | nil => rfl
  | cons source rest inductionHypothesis =>
      simp only [List.foldl_cons]
      split
      · rw [inductionHypothesis]
        apply congrArg (fun start =>
          rest.foldl (fun accumulated source =>
            if source ≤ degree ∧ degree - source < ringDegree then
              accumulated +
                ringFCoeff (evalRing env challenge) source *
                ringFCoeff (evalRing env value) (degree - source)
            else accumulated) start)
        simp only [Expr.eval_hadd, Expr.eval_hmul, exprCoeff, ringFCoeff,
          evalRing]
        split <;> split <;> rfl
      · exact inductionHypothesis initial

private theorem rawExpr_eval (env : Env)
    (challenge value : Fin ringDegree → Expr) (degree : Nat) :
    (rawExpr challenge value degree).eval env =
      rawMulCoeffF (evalRing env challenge) (evalRing env value) degree := by
  unfold rawExpr rawMulCoeffF
  simpa using foldl_eval env (List.range ringDegree) degree
    challenge value 0

private theorem mulExpr_eval (env : Env)
    (challenge value : Fin ringDegree → Expr) (lane : Fin ringDegree) :
    (mulExpr challenge value lane).eval env =
      ringFMul (evalRing env challenge) (evalRing env value) lane := by
  unfold mulExpr ringFMul
  by_cases foldedLow : lane.val < ringMiddleDegree
  · by_cases hasTwice : lane.val + 81 ≤ 106
    · simp only [if_pos foldedLow, if_pos hasTwice, Expr.eval_hadd,
        Expr.eval_sub, rawExpr_eval]
    · simp only [if_pos foldedLow, if_neg hasTwice, Expr.eval_hadd,
        Expr.eval_sub, rawExpr_eval]
      rfl
  · by_cases hasTwice : lane.val + 81 ≤ 106
    · simp only [if_neg foldedLow, if_pos hasTwice, Expr.eval_hadd,
        Expr.eval_sub, rawExpr_eval]
    · simp only [if_neg foldedLow, if_neg hasTwice, Expr.eval_hadd,
        Expr.eval_sub, rawExpr_eval]
      rfl

theorem recipe_eval
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat)
    (env : Env) (index : Fin (privateCount blockCount cellCount)) :
    (recipe interface offset index).eval env =
      (interface.prior offset (blockOf index) (laneOf index)
        (cellOf index)).eval env +
      ringFMul
        (fun lane => (interface.challenge offset lane).eval env)
        (fun lane =>
          (interface.value offset (blockOf index) lane
            (cellOf index)).eval env)
        (laneOf index) := by
  unfold recipe
  rw [Expr.eval_hadd, mulExpr_eval]
  rfl

@[simp] theorem recipes_length
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat) :
    (recipes interface offset).length = privateCount blockCount cellCount := by
  simp [recipes]

private theorem exprCoeff_varsBelow
    (value : Fin ringDegree → Expr) (bound index : Nat)
    (below : ∀ lane, (value lane).VarsBelow bound) :
    (exprCoeff value index).VarsBelow bound := by
  unfold exprCoeff
  split
  · exact below _
  · trivial

private theorem foldl_varsBelow
    (indices : List Nat) (degree bound : Nat)
    (challenge value : Fin ringDegree → Expr)
    (challengeBelow : ∀ lane, (challenge lane).VarsBelow bound)
    (valueBelow : ∀ lane, (value lane).VarsBelow bound)
    (initial : Expr) (initialBelow : initial.VarsBelow bound) :
    (indices.foldl (fun accumulated source =>
      if source ≤ degree ∧ degree - source < ringDegree then
        accumulated + exprCoeff challenge source *
          exprCoeff value (degree - source)
      else accumulated) initial).VarsBelow bound := by
  induction indices generalizing initial with
  | nil => exact initialBelow
  | cons source rest inductionHypothesis =>
      simp only [List.foldl_cons]
      split
      · apply inductionHypothesis
        apply Expr.VarsBelow.add _ _ _ initialBelow
        apply Expr.VarsBelow.mul
        · exact exprCoeff_varsBelow challenge bound source challengeBelow
        · exact exprCoeff_varsBelow value bound (degree - source) valueBelow
      · exact inductionHypothesis initial initialBelow

private theorem rawExpr_varsBelow
    (challenge value : Fin ringDegree → Expr) (degree bound : Nat)
    (challengeBelow : ∀ lane, (challenge lane).VarsBelow bound)
    (valueBelow : ∀ lane, (value lane).VarsBelow bound) :
    (rawExpr challenge value degree).VarsBelow bound := by
  unfold rawExpr
  exact foldl_varsBelow (List.range ringDegree) degree bound
    challenge value challengeBelow valueBelow 0 trivial

theorem mulExpr_varsBelow
    (challenge value : Fin ringDegree → Expr) (lane : Fin ringDegree)
    (bound : Nat)
    (challengeBelow : ∀ current, (challenge current).VarsBelow bound)
    (valueBelow : ∀ current, (value current).VarsBelow bound) :
    (mulExpr challenge value lane).VarsBelow bound := by
  unfold mulExpr
  apply Expr.VarsBelow.add
  · apply Expr.VarsBelow.sub
    · exact rawExpr_varsBelow challenge value lane.val bound
        challengeBelow valueBelow
    · split
      · exact rawExpr_varsBelow challenge value
          (lane.val + ringDegree) bound challengeBelow valueBelow
      · exact rawExpr_varsBelow challenge value
          (lane.val + ringMiddleDegree) bound challengeBelow valueBelow
  · split
    · exact rawExpr_varsBelow challenge value (lane.val + 81) bound
        challengeBelow valueBelow
    · trivial

private theorem recipe_varsBelow
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat)
    {env : Env} (assumptions : Assumptions interface offset env)
    (index : Fin (privateCount blockCount cellCount)) :
    (recipe interface offset index).VarsBelow offset := by
  unfold recipe
  apply Expr.VarsBelow.add
  · exact assumptions.priorBelow _ _ _
  · apply mulExpr_varsBelow
    · exact assumptions.challengeBelow
    · intro current
      exact assumptions.valueBelow (blockOf index) current (cellOf index)

theorem recipes_causal
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env) :
    RecipesCausal offset (recipes interface offset) := by
  apply recipesCausal_of_all_below
  intro expression member
  rcases List.mem_ofFn.mp member with ⟨index, rfl⟩
  exact recipe_varsBelow interface offset assumptions index

theorem flatConstraints_operations
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat) :
    flatConstraints (operations interface offset) =
      recipeConstraints offset (recipes interface offset) := by
  simp [operations, flatConstraints, Op.flatConstraints,
    WitnessBatch.arithmetic]

theorem localLength_eq
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat) :
    localLength (operations interface offset) =
      privateCount blockCount cellCount := by
  simp [operations, localLength, Op.localLength, recipes_length]

theorem flatConstraints_length
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat) :
    (flatConstraints (operations interface offset)).length =
      privateCount blockCount cellCount := by
  rw [flatConstraints_operations, recipeConstraints_length, recipes_length]

theorem flatConstraints_varsBelow
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints (operations interface offset),
      expression.VarsBelow (offset + privateCount blockCount cellCount) := by
  rw [flatConstraints_operations]
  have scope := recipeConstraints_varsBelow_of_causal offset
    (recipes interface offset) (recipes_causal interface offset env assumptions)
  rw [recipes_length] at scope
  exact scope

theorem soundness
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (env : Env) (offset : Nat)
    (_assumptions : Assumptions interface offset env)
    (rows : holds env (operations interface offset)) :
    SpecHolds interface offset env := by
  have recipeRows := rows
    (.witness (WitnessBatch.arithmetic offset (recipes interface offset)))
    (by simp [operations])
  intro index
  have value := recipeConstraints_value env offset (recipes interface offset)
    recipeRows index.val (by simpa [recipes_length] using index.isLt)
  rw [show (recipes interface offset).get
      ⟨index.val, by simpa [recipes_length] using index.isLt⟩ =
        recipe interface offset index by simp [recipes]] at value
  simpa [output, recipe_eval] using value

theorem complete
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (operations interface offset)) ∧
      holdsFlat completed (operations interface offset) := by
  let completed := executeRecipes env offset (recipes interface offset)
  refine ⟨completed, ?_, ?_⟩
  · have agrees := executeRecipes_agreesOutside env offset
      (recipes interface offset)
    rw [localLength_eq]
    simpa [recipes_length] using agrees
  · unfold holdsFlat
    rw [flatConstraints_operations]
    exact executeRecipes_holds_recipeConstraints env offset
      (recipes interface offset) (recipes_causal interface offset env assumptions)

theorem completeness
    {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (_specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (operations interface offset)) ∧
      holdsFlat completed (operations interface offset) :=
  complete interface env offset assumptions

def circuit {blockCount cellCount : Nat} [NeZero cellCount]
    (interface : Interface blockCount cellCount) : FormalCircuit where
  main := main interface
  assumptions := Assumptions interface
  spec := SpecHolds interface
  privateCount := fun _ => privateCount blockCount cellCount
  rowCount := fun _ => privateCount blockCount cellCount
  privateCount_eq := localLength_eq interface
  rowCount_eq := flatConstraints_length interface
  soundness := soundness interface
  completeness := completeness interface

end NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep
