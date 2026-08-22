import Mathlib.Data.ZMod.Defs
import NightstreamFPrime.Spec.Algebra

/-!
Owns the circuit DSL: expressions over offset-indexed variables, the three
operations (`witness`, `assertZero`, `subcircuit`), the offset-state circuit
monad, evaluation under an environment, and the meaning of "all constraints
hold". No column numbers, no rows, no artifact data: those belong to `Layout/`.
-/

namespace NightstreamFPrime.Circuit

open NightstreamFPrime.Spec

/-- Arithmetic expressions over variables addressed by absolute offset. -/
inductive Expr where
  | var (index : Nat)
  | const (value : F)
  | add (left right : Expr)
  | mul (left right : Expr)
deriving Repr, DecidableEq, Inhabited

/-- An environment assigns every variable a field value. -/
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
  simp [eval, neg_one_mul]
@[simp] theorem eval_sub (env : Env) (a b : Expr) : (a - b).eval env = a.eval env - b.eval env := by
  show (add a (neg b)).eval env = _
  simp [eval, neg, sub_eq_add_neg, neg_one_mul]

end Expr

/-- A flat constraint list is the physical relation that `Layout/` lowers.
Witness allocation has no constraint row and is tracked separately. -/
def ConstraintsHold (env : Env) (constraints : List Expr) : Prop :=
  ∀ e ∈ constraints, e.eval env = 0

/-- Canonical equations for a straight-line witness batch. Recipe `i` fills
variable `start + i`; the emitted row checks that value against the recipe. -/
def recipeConstraints (start : Nat) : List Expr → List Expr
  | [] => []
  | recipe :: rest =>
      (Expr.var start - recipe) :: recipeConstraints (start + 1) rest

@[simp] theorem recipeConstraints_length (start : Nat) (recipes : List Expr) :
    (recipeConstraints start recipes).length = recipes.length := by
  induction recipes generalizing start with
  | nil => rfl
  | cons recipe rest ih => simp [recipeConstraints, ih]

/-- Exportable witness-program instruction. `recipe` expressions are data,
not native closures, and their results remain non-authoritative until the
corresponding rows pass. -/
structure WitnessBatch where
  start : Nat
  recipes : List Expr

/-- A proof-carrying opaque child. Parents use only `spec`; `Layout/` uses
`constraints`. The proof is the only authority that connects the two. -/
structure Subcircuit where
  name : String
  localLength : Nat
  witnesses : List WitnessBatch
  constraints : List Expr
  spec : Env → Prop
  soundness : ∀ env, ConstraintsHold env constraints → spec env

/-- One circuit operation. A `witness` allocates `count` fresh variables
starting at the current offset; the exported witness program fills them. An
`assertZero` contributes one flat constraint. A `subcircuit` is opaque to its
parent and carries its own soundness proof. -/
inductive Op where
  | witness (batch : WitnessBatch)
  | assertZero (e : Expr)
  | subcircuit (child : Subcircuit)

/-- Number of variables an operation list allocates. -/
def Op.localLength : Op → Nat
  | .witness batch => batch.recipes.length
  | .assertZero _ => 0
  | .subcircuit child => child.localLength

def localLength (ops : List Op) : Nat := (ops.map Op.localLength).sum

/-- The relation an operation list imposes on an environment. A subcircuit
contributes its spec, not its operations: parents never see inside. -/
def Op.holds (env : Env) : Op → Prop
  | .witness batch => ConstraintsHold env (recipeConstraints batch.start batch.recipes)
  | .assertZero e => e.eval env = 0
  | .subcircuit child => child.spec env

def holds (env : Env) (ops : List Op) : Prop := ∀ op ∈ ops, op.holds env

/-- The exact constraints below one operation. -/
def Op.flatConstraints : Op → List Expr
  | .witness batch => recipeConstraints batch.start batch.recipes
  | .assertZero e => [e]
  | .subcircuit child => child.constraints

def flatConstraints (ops : List Op) : List Expr := ops.flatMap Op.flatConstraints

def Op.witnesses : Op → List WitnessBatch
  | .witness batch => [batch]
  | .assertZero _ => []
  | .subcircuit child => child.witnesses

def witnesses (ops : List Op) : List WitnessBatch := ops.flatMap Op.witnesses

/-- The exact flattened relation, used only by `Layout/` to connect emitted
rows to opaque logical specifications. -/
def Op.holdsFlat (env : Env) (op : Op) : Prop :=
  ConstraintsHold env op.flatConstraints

def holdsFlat (env : Env) (ops : List Op) : Prop :=
  ConstraintsHold env (flatConstraints ops)

/-- Offset-state writer monad: a circuit consumes an offset and returns a
result, the new offset, and the operations it emitted. -/
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

/-- Allocate one fresh variable with a canonical witness recipe. -/
def witness (recipe : Expr) : Circuit Expr := fun n =>
  (Expr.var n, n + 1, [Op.witness ⟨n, [recipe]⟩])

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
    witness recipe n = (Expr.var n, n + 1, [Op.witness ⟨n, [recipe]⟩]) := rfl
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
      exact h e (by simp [Op.holdsFlat, Op.flatConstraints, ConstraintsHold])
  | subcircuit child =>
      exact child.soundness env

/-- Physical satisfaction implies the opaque logical meaning of every child.
This is the generic layout-to-builder soundness bridge. -/
theorem holdsFlat_implies_holds (env : Env) (ops : List Op) :
    holdsFlat env ops → holds env ops := by
  intro h op hop
  apply op.holds_of_holdsFlat env
  intro e he
  apply h e
  simp only [flatConstraints, List.mem_flatMap]
  exact ⟨op, hop, he⟩

/-- Environments may change only variables allocated by a circuit call. -/
def AgreesOutside (before after : Env) (offset count : Nat) : Prop :=
  ∀ index, index < offset ∨ offset + count ≤ index → after index = before index

/-- Adjacent witness ranges compose into one exact changed interval. -/
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

/-- The one proved-circuit object. Soundness quantifies over arbitrary witness
values. Completeness supplies an environment for the circuit's local witness
range while preserving every external variable. -/
structure FormalCircuit where
  main : Circuit Unit
  assumptions : Nat → Env → Prop := fun _ _ => True
  spec : Nat → Env → Prop
  soundness : ∀ env offset, assumptions offset env →
    holds env (Circuit.ops main offset) → spec offset env
  completeness : ∀ env offset, assumptions offset env → spec offset env →
    ∃ completed,
      AgreesOutside env completed offset (localLength (Circuit.ops main offset)) ∧
      holdsFlat completed (Circuit.ops main offset)

namespace FormalCircuit

/-- Package a proved circuit as one opaque operation. Its visible meaning is
`assumptions → spec`; its physical constraints are the flattened child rows. -/
def asSubcircuit (circuit : FormalCircuit) (name : String) (offset : Nat) : Subcircuit where
  name := name
  localLength := localLength (Circuit.ops circuit.main offset)
  witnesses := witnesses (Circuit.ops circuit.main offset)
  constraints := flatConstraints (Circuit.ops circuit.main offset)
  spec := fun env => circuit.assumptions offset env → circuit.spec offset env
  soundness := by
    intro env hflat hassumptions
    apply circuit.soundness env offset hassumptions
    exact holdsFlat_implies_holds env _ hflat

end FormalCircuit

namespace Circuit

/-- Call a proved child. The parent operation contains the child's flattened
rows but exposes only the child's proved interface. -/
def call (name : String) (circuit : FormalCircuit) : Circuit Unit := fun offset =>
  let child := circuit.asSubcircuit name offset
  ((), offset + child.localLength, [Op.subcircuit child])

end Circuit

end NightstreamFPrime.Circuit
