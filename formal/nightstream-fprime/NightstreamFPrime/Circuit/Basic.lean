import Mathlib.Data.ZMod.Basic
import NightstreamFPrime.Spec.Algebra

/-!
Owns the circuit DSL: expressions over offset-indexed variables, the three
operations (`witness`, `assertZero`, `subcircuit`), the offset-state circuit
monad, evaluation under an environment, and the meaning of "all constraints
hold". No column numbers, no rows, no artifact data: those belong to `Layout/`.

Design credit: the operation list, the opaque proof-carrying subcircuit, the
`FormalCircuit` record (`main`, `assumptions`, `spec`, `soundness`,
`completeness`), `asSubcircuit`, and `Circuit.call` follow the shape of Clean
(https://github.com/Verified-zkEVM/clean, Copyright (c) 2024-2025 zkSecurity,
LLC, MIT License). This file is a fresh implementation, not a copy: one circuit
record, no typed input/output classes, no lookups, exportable witness recipes
with certified row counts, and `AgreesOutside` completeness.
-/

namespace NightstreamFPrime.Circuit

set_option linter.dupNamespace false

open NightstreamFPrime.Spec

inductive Expr where
  | var (index : Nat)
  | const (value : F)
  | add (left right : Expr)
  | mul (left right : Expr)
deriving Repr, DecidableEq, Inhabited

abbrev Env := Nat → F

namespace Expr

def eval (env : Env) : Expr → F
  | var i => env i
  | const v => v
  | add a b => a.eval env + b.eval env
  | mul a b => a.eval env * b.eval env

instance : Add Expr := ⟨Expr.add⟩
instance : Mul Expr := ⟨Expr.mul⟩
instance (n : Nat) : OfNat Expr n := ⟨Expr.const (⟨n % goldilocksModulus, Nat.mod_lt _ (by decide)⟩)⟩

def neg (e : Expr) : Expr := mul (const (-1)) e
def sub (a b : Expr) : Expr := add a (neg b)
instance : Neg Expr := ⟨Expr.neg⟩
instance : Sub Expr := ⟨Expr.sub⟩

@[simp] theorem eval_var (env : Env) (i : Nat) : (var i).eval env = env i := rfl
@[simp] theorem eval_const (env : Env) (v : F) : (const v).eval env = v := rfl
@[simp] theorem eval_add (env : Env) (a b : Expr) : (add a b).eval env = a.eval env + b.eval env := rfl
@[simp] theorem eval_mul (env : Env) (a b : Expr) : (mul a b).eval env = a.eval env * b.eval env := rfl
@[simp] theorem eval_hadd (env : Env) (a b : Expr) : (a + b).eval env = a.eval env + b.eval env := rfl
@[simp] theorem eval_hmul (env : Env) (a b : Expr) : (a * b).eval env = a.eval env * b.eval env := rfl
@[simp] theorem eval_neg (env : Env) (a : Expr) : (-a).eval env = -(a.eval env) := by
  show (mul (const (-1)) a).eval env = _
  simp [eval]
@[simp] theorem eval_sub (env : Env) (a b : Expr) : (a - b).eval env = a.eval env - b.eval env := by
  show (add a (neg b)).eval env = _
  simp [eval, neg, sub_eq_add_neg]

end Expr

inductive Hint where
  | bit (source : Expr) (index : Nat)
  | inverseOrZero (source : Expr)
  | quotientFive (source : Expr)
  | remainderFive (source : Expr)
deriving Repr, DecidableEq

namespace Hint

def ofNat (value : Nat) : F :=
  ⟨value % goldilocksModulus, Nat.mod_lt _ (by decide)⟩

def inverse (value : F) : F :=
  ZMod.inv goldilocksModulus value

def eval (env : Env) : Hint → F
  | .bit source index =>
      ofNat (((source.eval env).val >>> index) &&& 1)
  | .inverseOrZero source => inverse (source.eval env)
  | .quotientFive source => ofNat ((source.eval env).val / 5)
  | .remainderFive source => ofNat ((source.eval env).val % 5)

def source : Hint → Expr
  | .bit expression _ => expression
  | .inverseOrZero expression => expression
  | .quotientFive expression => expression
  | .remainderFive expression => expression

end Hint

def ConstraintsHold (env : Env) (constraints : List Expr) : Prop :=
  ∀ e ∈ constraints, e.eval env = 0

def recipeConstraints (start : Nat) : List Expr → List Expr
  | [] => []
  | recipe :: rest =>
      (Expr.var start - recipe) :: recipeConstraints (start + 1) rest

@[inline] def recipeConstraintsTR (start : Nat) (recipes : List Expr) :
    List Expr :=
  go start recipes []
where
  go : Nat → List Expr → List Expr → List Expr
    | _, [], constraintsRev => constraintsRev.reverse
    | output, recipe :: rest, constraintsRev =>
        go (output + 1) rest
          ((Expr.var output - recipe) :: constraintsRev)

@[csimp] theorem recipeConstraints_eq_recipeConstraintsTR :
    @recipeConstraints = @recipeConstraintsTR := by
  funext start recipes
  let rec go : ∀ (output : Nat) (remaining : List Expr)
      (constraintsRev : List Expr),
      recipeConstraintsTR.go output remaining constraintsRev =
        constraintsRev.reverse ++ recipeConstraints output remaining
    | _, [], constraintsRev => by
        simp [recipeConstraintsTR.go, recipeConstraints]
    | output, recipe :: rest, constraintsRev => by
        simp only [recipeConstraintsTR.go, recipeConstraints]
        rw [go (output + 1) rest
          ((Expr.var output - recipe) :: constraintsRev)]
        simp [List.reverse_cons, List.append_assoc]
  exact (go start recipes []).symm

@[simp] theorem recipeConstraints_length (start : Nat) (recipes : List Expr) :
    (recipeConstraints start recipes).length = recipes.length := by
  induction recipes generalizing start with
  | nil => rfl
  | cons recipe rest ih => simp [recipeConstraints, ih]

structure WitnessBatch where
  start : Nat
  recipes : List Expr
  hints : List Hint := []
deriving Repr

def WitnessBatch.outputLength (batch : WitnessBatch) : Nat :=
  batch.recipes.length + batch.hints.length

def WitnessBatch.arithmetic (start : Nat) (recipes : List Expr) :
    WitnessBatch :=
  { start := start, recipes := recipes, hints := [] }

def WitnessBatch.hinted (start : Nat) (hints : List Hint) : WitnessBatch :=
  { start := start, recipes := [], hints := hints }

@[simp] theorem WitnessBatch.arithmetic_start
    (start : Nat) (recipes : List Expr) :
    (WitnessBatch.arithmetic start recipes).start = start := by
  rfl

@[simp] theorem WitnessBatch.arithmetic_recipes
    (start : Nat) (recipes : List Expr) :
    (WitnessBatch.arithmetic start recipes).recipes = recipes := by
  rfl

@[simp] theorem WitnessBatch.arithmetic_hints
    (start : Nat) (recipes : List Expr) :
    (WitnessBatch.arithmetic start recipes).hints = [] := by
  rfl

@[simp] theorem WitnessBatch.hinted_start
    (start : Nat) (hints : List Hint) :
    (WitnessBatch.hinted start hints).start = start := by
  rfl

@[simp] theorem WitnessBatch.hinted_recipes
    (start : Nat) (hints : List Hint) :
    (WitnessBatch.hinted start hints).recipes = [] := by
  rfl

@[simp] theorem WitnessBatch.hinted_hints
    (start : Nat) (hints : List Hint) :
    (WitnessBatch.hinted start hints).hints = hints := by
  rfl

@[simp] theorem WitnessBatch.arithmetic_outputLength
    (start : Nat) (recipes : List Expr) :
    (WitnessBatch.arithmetic start recipes).outputLength = recipes.length := by
  simp [WitnessBatch.arithmetic, WitnessBatch.outputLength]

@[simp] theorem WitnessBatch.hinted_outputLength
    (start : Nat) (hints : List Hint) :
    (WitnessBatch.hinted start hints).outputLength = hints.length := by
  simp [WitnessBatch.hinted, WitnessBatch.outputLength]

structure Subcircuit where
  name : String
  localLength : Nat
  witnesses : List WitnessBatch
  constraints : List Expr
  rowCount : Nat
  rowCount_eq : constraints.length = rowCount
  spec : Env → Prop
  soundness : ∀ env, ConstraintsHold env constraints → spec env

inductive Op where
  | witness (batch : WitnessBatch)
  | assertZero (e : Expr)
  | subcircuit (child : Subcircuit)

def Op.localLength : Op → Nat
  | .witness batch => batch.outputLength
  | .assertZero _ => 0
  | .subcircuit child => child.localLength

def localLength (ops : List Op) : Nat := (ops.map Op.localLength).sum

def Op.holds (env : Env) : Op → Prop
  | .witness batch => ConstraintsHold env (recipeConstraints batch.start batch.recipes)
  | .assertZero e => e.eval env = 0
  | .subcircuit child => child.spec env

def holds (env : Env) (ops : List Op) : Prop := ∀ op ∈ ops, op.holds env

def Op.flatConstraints : Op → List Expr
  | .witness batch => recipeConstraints batch.start batch.recipes
  | .assertZero e => [e]
  | .subcircuit child => child.constraints

def flatConstraints (ops : List Op) : List Expr := ops.flatMap Op.flatConstraints

def Op.rowCount : Op → Nat
  | .witness batch => batch.recipes.length
  | .assertZero _ => 1
  | .subcircuit child => child.rowCount

def rowCount (ops : List Op) : Nat := (ops.map Op.rowCount).sum

theorem Op.flatConstraints_length_eq_rowCount (operation : Op) :
    operation.flatConstraints.length = operation.rowCount := by
  cases operation with
  | witness batch =>
      exact recipeConstraints_length batch.start batch.recipes
  | assertZero _ =>
      rfl
  | subcircuit child =>
      exact child.rowCount_eq

theorem flatConstraints_length_eq_rowCount (ops : List Op) :
    (flatConstraints ops).length = rowCount ops := by
  induction ops with
  | nil =>
      rfl
  | cons operation rest inductionHypothesis =>
      change (operation.flatConstraints ++ flatConstraints rest).length =
        operation.rowCount + rowCount rest
      rw [List.length_append, operation.flatConstraints_length_eq_rowCount,
        inductionHypothesis]

@[simp] theorem rowCount_append (left right : List Op) :
    rowCount (left ++ right) = rowCount left + rowCount right := by
  simp [rowCount, List.sum_append]

@[simp] theorem flatConstraints_append (left right : List Op) :
    flatConstraints (left ++ right) =
      flatConstraints left ++ flatConstraints right := by
  simp [flatConstraints]

@[simp] theorem flatConstraints_singleton (operation : Op) :
    flatConstraints [operation] = operation.flatConstraints := by
  simp [flatConstraints]

def Op.witnesses : Op → List WitnessBatch
  | .witness batch => [batch]
  | .assertZero _ => []
  | .subcircuit child => child.witnesses

def witnesses (ops : List Op) : List WitnessBatch := ops.flatMap Op.witnesses

def Op.holdsFlat (env : Env) (op : Op) : Prop :=
  ConstraintsHold env op.flatConstraints

def holdsFlat (env : Env) (ops : List Op) : Prop :=
  ConstraintsHold env (flatConstraints ops)

def Circuit (α : Type) := Nat → α × Nat × List Op

namespace Circuit

variable {α β : Type}

def pure' (a : α) : Circuit α := fun n => (a, n, [])

def bind' (c : Circuit α) (f : α → Circuit β) : Circuit β := fun n =>
  let (a, n₁, ops₁) := c n
  let (b, n₂, ops₂) := f a n₁
  (b, n₂, ops₁ ++ ops₂)

instance : Monad Circuit where
  pure := pure'
  bind := bind'

def witness (recipe : Expr) : Circuit Expr := fun n =>
  (Expr.var n, n + 1,
    [Op.witness (WitnessBatch.arithmetic n [recipe])])

def hint (instruction : Hint) : Circuit Expr := fun n =>
  (Expr.var n, n + 1,
    [Op.witness (WitnessBatch.hinted n [instruction])])

def assertZero (e : Expr) : Circuit Unit := fun n => ((), n, [Op.assertZero e])

def run (c : Circuit α) (offset : Nat) : α × Nat × List Op := c offset

def ops (c : Circuit α) (offset : Nat) : List Op := (c offset).2.2
def output (c : Circuit α) (offset : Nat) : α := (c offset).1
def finalOffset (c : Circuit α) (offset : Nat) : Nat := (c offset).2.1

@[simp] theorem run_pure (a : α) (n : Nat) : (pure a : Circuit α) n = (a, n, []) := rfl
@[simp] theorem run_bind (c : Circuit α) (f : α → Circuit β) (n : Nat) :
    (c >>= f) n =
      ((f (c n).1 (c n).2.1).1, (f (c n).1 (c n).2.1).2.1,
        (c n).2.2 ++ (f (c n).1 (c n).2.1).2.2) := rfl
@[simp] theorem run_witness (recipe : Expr) (n : Nat) :
    witness recipe n =
      (Expr.var n, n + 1,
        [Op.witness (WitnessBatch.arithmetic n [recipe])]) := rfl
@[simp] theorem run_hint (instruction : Hint) (n : Nat) :
    hint instruction n =
      (Expr.var n, n + 1,
        [Op.witness (WitnessBatch.hinted n [instruction])]) := rfl
@[simp] theorem run_assertZero (e : Expr) (n : Nat) : assertZero e n = ((), n, [Op.assertZero e]) := rfl

end Circuit

@[simp] theorem holds_nil (env : Env) : holds env [] := by simp [holds]
@[simp] theorem holds_cons (env : Env) (op : Op) (ops : List Op) :
    holds env (op :: ops) ↔ op.holds env ∧ holds env ops := by
  simp [holds]
@[simp] theorem holds_append (env : Env) (a b : List Op) :
    holds env (a ++ b) ↔ holds env a ∧ holds env b := by
  unfold holds
  constructor
  · intro h; exact ⟨fun op m => h op (List.mem_append_left _ m), fun op m => h op (List.mem_append_right _ m)⟩
  · rintro ⟨ha, hb⟩ op m
    rcases List.mem_append.mp m with m | m
    · exact ha op m
    · exact hb op m
@[simp] theorem Op.holds_witness (env : Env) (batch : WitnessBatch) :
    (Op.witness batch).holds env ↔
      ConstraintsHold env (recipeConstraints batch.start batch.recipes) := Iff.rfl
@[simp] theorem Op.holds_assertZero (env : Env) (e : Expr) :
    (Op.assertZero e).holds env ↔ e.eval env = 0 := Iff.rfl
@[simp] theorem Op.holds_subcircuit (env : Env) (child : Subcircuit) :
    (Op.subcircuit child).holds env ↔ child.spec env := Iff.rfl

theorem Op.holds_of_holdsFlat (env : Env) (op : Op) :
    op.holdsFlat env → op.holds env := by
  cases op with
  | witness batch =>
      exact id
  | assertZero e =>
      intro h
      exact h e (by simp [Op.flatConstraints])
  | subcircuit child =>
      exact child.soundness env

theorem holdsFlat_implies_holds (env : Env) (ops : List Op) :
    holdsFlat env ops → holds env ops := by
  intro h op hop
  apply op.holds_of_holdsFlat env
  intro e he
  apply h e
  simp only [flatConstraints, List.mem_flatMap]
  exact ⟨op, hop, he⟩

def AgreesOutside (before after : Env) (offset count : Nat) : Prop :=
  ∀ index, index < offset ∨ offset + count ≤ index → after index = before index

theorem AgreesOutside.append {before middle after : Env}
    {offset firstCount secondCount : Nat}
    (first : AgreesOutside before middle offset firstCount)
    (second : AgreesOutside middle after (offset + firstCount) secondCount) :
    AgreesOutside before after offset (firstCount + secondCount) := by
  intro index outside
  rcases outside with below | above
  · calc
      after index = middle index := second index (Or.inl (by omega))
      _ = before index := first index (Or.inl below)
  · calc
      after index = middle index := second index (Or.inr (by omega))
      _ = before index := first index (Or.inr (by omega))

theorem logicalCompleteness_of_flat
    {env : Env} {offset : Nat} {ops : List Op}
    (flat : ∃ completed,
      AgreesOutside env completed offset (localLength ops) ∧
      holdsFlat completed ops) :
    ∃ completed,
      AgreesOutside env completed offset (localLength ops) ∧
      holds completed ops := by
  rcases flat with ⟨completed, agrees, rows⟩
  exact ⟨completed, agrees, holdsFlat_implies_holds completed ops rows⟩

structure FormalCircuit where
  main : Circuit Unit
  assumptions : Nat → Env → Prop := fun _ _ => True
  spec : Nat → Env → Prop
  privateCount : Nat → Nat := fun offset =>
    localLength (Circuit.ops main offset)
  rowCount : Nat → Nat := fun offset =>
    (flatConstraints (Circuit.ops main offset)).length
  privateCount_eq : ∀ offset,
    localLength (Circuit.ops main offset) = privateCount offset := by
      intro _
      rfl
  rowCount_eq : ∀ offset,
    (flatConstraints (Circuit.ops main offset)).length = rowCount offset := by
      intro _
      rfl
  soundness : ∀ env offset, assumptions offset env →
    holds env (Circuit.ops main offset) → spec offset env
  completeness : ∀ env offset, assumptions offset env → spec offset env →
    ∃ completed,
      AgreesOutside env completed offset (localLength (Circuit.ops main offset)) ∧
      holdsFlat completed (Circuit.ops main offset)

namespace FormalCircuit

def withConstantFootprint (circuit : FormalCircuit)
    (privateCount rowCount : Nat)
    (privateCount_eq : ∀ offset,
      localLength (Circuit.ops circuit.main offset) = privateCount)
    (rowCount_eq : ∀ offset,
      (flatConstraints (Circuit.ops circuit.main offset)).length = rowCount) :
    FormalCircuit :=
  { circuit with
    privateCount := fun _ => privateCount
    rowCount := fun _ => rowCount
    privateCount_eq := privateCount_eq
    rowCount_eq := rowCount_eq }

@[simp] theorem withConstantFootprint_main (circuit : FormalCircuit)
    (privateCount rowCount : Nat)
    (privateCount_eq : ∀ offset,
      localLength (Circuit.ops circuit.main offset) = privateCount)
    (rowCount_eq : ∀ offset,
      (flatConstraints (Circuit.ops circuit.main offset)).length = rowCount) :
    (circuit.withConstantFootprint privateCount rowCount privateCount_eq
      rowCount_eq).main = circuit.main := by
  rfl

@[simp] theorem withConstantFootprint_privateCount (circuit : FormalCircuit)
    (privateCount rowCount : Nat)
    (privateCount_eq : ∀ offset,
      localLength (Circuit.ops circuit.main offset) = privateCount)
    (rowCount_eq : ∀ offset,
      (flatConstraints (Circuit.ops circuit.main offset)).length = rowCount)
    (offset : Nat) :
    (circuit.withConstantFootprint privateCount rowCount privateCount_eq
      rowCount_eq).privateCount offset = privateCount := by
  rfl

@[simp] theorem withConstantFootprint_rowCount (circuit : FormalCircuit)
    (privateCount rowCount : Nat)
    (privateCount_eq : ∀ offset,
      localLength (Circuit.ops circuit.main offset) = privateCount)
    (rowCount_eq : ∀ offset,
      (flatConstraints (Circuit.ops circuit.main offset)).length = rowCount)
    (offset : Nat) :
    (circuit.withConstantFootprint privateCount rowCount privateCount_eq
      rowCount_eq).rowCount offset = rowCount := by
  rfl

def asSubcircuit (circuit : FormalCircuit) (name : String) (offset : Nat) : Subcircuit where
  name := name
  localLength := circuit.privateCount offset
  witnesses := witnesses (Circuit.ops circuit.main offset)
  constraints := flatConstraints (Circuit.ops circuit.main offset)
  rowCount := circuit.rowCount offset
  rowCount_eq := circuit.rowCount_eq offset
  spec := fun env => circuit.assumptions offset env → circuit.spec offset env
  soundness := by
    intro env hflat hassumptions
    apply circuit.soundness env offset hassumptions
    exact holdsFlat_implies_holds env _ hflat

@[simp] theorem asSubcircuit_constraints (circuit : FormalCircuit)
    (name : String) (offset : Nat) :
    (circuit.asSubcircuit name offset).constraints =
      flatConstraints (Circuit.ops circuit.main offset) := by
  rfl

@[simp] theorem asSubcircuit_localLength (circuit : FormalCircuit)
    (name : String) (offset : Nat) :
    (circuit.asSubcircuit name offset).localLength =
      localLength (Circuit.ops circuit.main offset) := by
  exact (circuit.privateCount_eq offset).symm

@[simp] theorem asSubcircuit_rowCount (circuit : FormalCircuit)
    (name : String) (offset : Nat) :
    (circuit.asSubcircuit name offset).rowCount = circuit.rowCount offset := by
  rfl

theorem asSubcircuit_constraints_length (circuit : FormalCircuit)
    (name : String) (offset : Nat) :
    (circuit.asSubcircuit name offset).constraints.length =
      circuit.rowCount offset := by
  exact circuit.rowCount_eq offset

end FormalCircuit

namespace Circuit

def call (name : String) (circuit : FormalCircuit) : Circuit Unit := fun offset =>
  let child := circuit.asSubcircuit name offset
  ((), offset + child.localLength, [Op.subcircuit child])

end Circuit

end NightstreamFPrime.Circuit
