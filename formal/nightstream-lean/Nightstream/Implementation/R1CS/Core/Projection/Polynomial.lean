import Nightstream.Implementation.R1CS.Core.Projection.Trace

/-! Concrete Goldilocks-quadratic polynomial semantics for projection traces. -/

namespace Nightstream.Implementation.R1CS.ProjectionProgram

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
/-! ## Concrete Goldilocks-quadratic semantics -/
instance : NeZero goldilocksP := ⟨by decide⟩

abbrev F := Fin goldilocksP
structure K where
  c0 : F
  c1 : F
deriving DecidableEq, Repr, Inhabited
def K.zero : K := ⟨0, 0⟩
def K.one : K := ⟨1, 0⟩
def K.ofBase (value : F) : K := ⟨value, 0⟩
def K.add (left right : K) : K :=
  ⟨left.c0 + right.c0, left.c1 + right.c1⟩

/-- Goldilocks quadratic extension multiplication, `X² = 7`. -/
def K.mul (left right : K) : K :=
  ⟨left.c0 * right.c0 + 7 * (left.c1 * right.c1),
   left.c0 * right.c1 + left.c1 * right.c0⟩
def K.pow (point : K) : Nat → K
  | 0 => K.one
  | exponent + 1 => K.mul (K.pow point exponent) point
def K.powersFrom (point current : K) : Nat → List K
  | 0 => []
  | count + 1 => current :: K.powersFrom point (K.mul current point) count
theorem K.take_powersFrom (point current : K) {count total : Nat}
    (within : count ≤ total) :
    (K.powersFrom point current total).take count =
      K.powersFrom point current count := by
  induction count generalizing total current with
  | zero => rfl
  | succ count inductionHypothesis =>
      cases total with
      | zero => omega
      | succ total =>
          simp only [K.powersFrom, List.take_succ_cons, List.cons.injEq, true_and]
          exact inductionHypothesis (current := K.mul current point)
            (Nat.le_of_succ_le_succ within)
def K.ops : Nightstream.SuperNeo.ProjectionCheck.Ops K where
  zero := K.zero
  add := K.add
  mul := K.mul
def residue (value : Nat) : F :=
  ⟨value % goldilocksP, Nat.mod_lt _ (by decide)⟩

@[simp] theorem residue_one : residue 1 = (1 : F) := rfl

@[simp] theorem residue_seven : residue 7 = (7 : F) := rfl

@[simp] theorem residue_zero : residue 0 = (0 : F) := rfl
def baseAt (assignment : Nat → Nat) (column : Nat) : F :=
  residue (assignment column)
def KColumns.value (columns : KColumns) (assignment : Nat → Nat) : K :=
  ⟨baseAt assignment columns.c0, baseAt assignment columns.c1⟩
def KTerms.value (terms : KTerms) (assignment : Nat → Nat) : K :=
  ⟨residue (lcEval assignment terms.c0),
   residue (lcEval assignment terms.c1)⟩
def ProjectionTrace.pairProductValues (trace : ProjectionTrace)
    (assignment : Nat → Nat) : List K :=
  trace.pairs.map fun pair => pair.product.output.value assignment

@[simp] theorem KTerms.ofColumns_value (columns : KColumns)
    (assignment : Nat → Nat) :
    (KTerms.ofColumns columns).value assignment = columns.value assignment := by
  rcases columns with ⟨c0, c1⟩
  simp [KTerms.ofColumns, KTerms.value, KColumns.value, baseAt, residue,
    lcEval]
def EvalTrace.ExpectedProducts (trace : EvalTrace)
    (assignment : Nat → Nat) : List K :=
  trace.entries.map fun entry =>
    K.mul (K.ofBase (baseAt assignment entry.1.1))
      (entry.1.2.value assignment)
def EvalTrace.PowersValid (trace : EvalTrace) (assignment : Nat → Nat)
    (point : K) : Prop :=
  trace.powers.map (fun power => power.value assignment) =
    K.powersFrom point K.one trace.coefficients.length
def basePolynomial (assignment : Nat → Nat)
    (columns : List Nat) : List K :=
  columns.map fun column => ⟨baseAt assignment column, 0⟩
namespace Polynomial
def add : List K → List K → List K
  | [], right => right
  | left, [] => left
  | leftHead :: leftTail, rightHead :: rightTail =>
      K.add leftHead rightHead :: add leftTail rightTail
def scale (scalar : K) : List K → List K
  | [] => []
  | head :: tail => K.mul scalar head :: scale scalar tail

/-- Constant-first schoolbook multiplication, retaining the fixed-width
zero padding of its inputs. -/
def mul : List K → List K → List K
  | [], _ => []
  | head :: tail, right =>
      add (scale head right) (K.zero :: mul tail right)
def sum : List (List K) → List K
  | [] => []
  | head :: tail => add head (sum tail)
def padRight (width : Nat) (coefficients : List K) : List K :=
  coefficients ++ List.replicate (width - coefficients.length) K.zero
def dot (assignment : Nat → Nat) (columns : List Nat)
    (powers : List K) : K :=
  (List.zip columns powers).foldr (fun entry suffix =>
    K.add (K.mul (K.ofBase (baseAt assignment entry.1)) entry.2) suffix)
    K.zero
def phi81 : List K :=
  [K.one] ++ List.replicate 26 K.zero ++
    [K.one] ++ List.replicate 26 K.zero ++ [K.one]
end Polynomial
def PairTrace.productPolynomial (trace : PairTrace)
    (assignment : Nat → Nat) : List K :=
  Polynomial.mul
    (basePolynomial assignment trace.rhoColumns)
    (basePolynomial assignment trace.inputColumns)

/-- The actual bounded coefficient identity checked by one projection trace.
It is not a projection-value surrogate: both sides contain the complete
coefficient vectors whose equality is required by the native ring action. -/
def ProjectionTrace.identity (trace : ProjectionTrace)
    (assignment : Nat → Nat) :
    Nightstream.SuperNeo.ProjectionCheck.Identity K where
  lhs := Polynomial.sum (trace.pairs.map fun pair =>
    pair.productPolynomial assignment)
  rhs := Polynomial.add
    (Polynomial.mul
      (basePolynomial assignment trace.quotientColumns) Polynomial.phi81)
    (Polynomial.padRight (trace.maxDegree + 1)
      (basePolynomial assignment trace.outputColumns))
  beta := trace.ladder.beta.value assignment
  maxDegree := trace.maxDegree
def BatchIdentity (traces : List ProjectionTrace)
    (assignment : Nat → Nat) :
    List (Nightstream.SuperNeo.ProjectionCheck.Identity K) :=
  traces.map fun trace => trace.identity assignment

/-! ## Algebra used by the polynomial interpreter -/
theorem fadd_assoc (a b c : F) : (a + b) + c = a + (b + c) := by
  apply Fin.ext
  simp only [Fin.val_add]
  rw [Nat.mod_add_mod, Nat.add_mod_mod, Nat.add_assoc]
theorem fadd_comm (a b : F) : a + b = b + a := by
  apply Fin.ext
  simp only [Fin.val_add, Nat.add_comm]
theorem fmul_assoc (a b c : F) : (a * b) * c = a * (b * c) :=
  Fin.mul_assoc _ _ _
theorem fmul_comm (a b : F) : a * b = b * a :=
  Fin.mul_comm _ _
theorem fmul_add (a b c : F) : a * (b + c) = a * b + a * c := by
  apply Fin.ext
  simp only [Fin.val_add, Fin.val_mul]
  rw [Nat.mul_mod_mod, Nat.mul_add, ← Nat.add_mod]
theorem fadd_mul (a b c : F) : (a + b) * c = a * c + b * c := by
  apply Fin.ext
  simp only [Fin.val_add, Fin.val_mul]
  rw [Nat.mod_mul_mod, Nat.add_mul, ← Nat.add_mod]

local instance : Std.Associative (fun (a b : F) => a + b) := ⟨fadd_assoc⟩
local instance : Std.Commutative (fun (a b : F) => a + b) := ⟨fadd_comm⟩
local instance : Std.Associative (fun (a b : F) => a * b) := ⟨fmul_assoc⟩
local instance : Std.Commutative (fun (a b : F) => a * b) := ⟨fmul_comm⟩
theorem K.add_assoc (a b c : K) :
    K.add (K.add a b) c = K.add a (K.add b c) := by
  rcases a with ⟨a0, a1⟩
  rcases b with ⟨b0, b1⟩
  rcases c with ⟨c0, c1⟩
  simp only [K.add, K.mk.injEq]
  exact ⟨fadd_assoc _ _ _, fadd_assoc _ _ _⟩
theorem K.add_comm (a b : K) : K.add a b = K.add b a := by
  rcases a with ⟨a0, a1⟩
  rcases b with ⟨b0, b1⟩
  simp only [K.add, K.mk.injEq]
  exact ⟨fadd_comm _ _, fadd_comm _ _⟩

@[simp] theorem K.add_zero (a : K) : K.add a K.zero = a := by
  rcases a with ⟨a0, a1⟩
  simp [K.add, K.zero]

@[simp] theorem K.zero_add (a : K) : K.add K.zero a = a := by
  rw [K.add_comm, K.add_zero]
theorem K.mul_assoc (a b c : K) :
    K.mul (K.mul a b) c = K.mul a (K.mul b c) := by
  rcases a with ⟨a0, a1⟩
  rcases b with ⟨b0, b1⟩
  rcases c with ⟨c0, c1⟩
  simp only [K.mul, K.mk.injEq]
  constructor <;> simp only [fmul_add, fadd_mul, fmul_assoc] <;> ac_rfl
theorem K.mul_comm (a b : K) : K.mul a b = K.mul b a := by
  rcases a with ⟨a0, a1⟩
  rcases b with ⟨b0, b1⟩
  simp only [K.mul, K.mk.injEq]
  constructor <;> ac_rfl
theorem K.mul_add (a b c : K) :
    K.mul a (K.add b c) = K.add (K.mul a b) (K.mul a c) := by
  rcases a with ⟨a0, a1⟩
  rcases b with ⟨b0, b1⟩
  rcases c with ⟨c0, c1⟩
  simp only [K.mul, K.add, K.mk.injEq]
  constructor <;> simp only [fmul_add] <;> ac_rfl
theorem K.add_mul (a b c : K) :
    K.mul (K.add a b) c = K.add (K.mul a c) (K.mul b c) := by
  rw [K.mul_comm, K.mul_add]
  congr 1 <;> rw [K.mul_comm]

@[simp] theorem K.mul_zero (a : K) : K.mul a K.zero = K.zero := by
  rcases a with ⟨a0, a1⟩
  simp only [K.mul, K.zero, Fin.mul_zero, Fin.add_zero]

@[simp] theorem K.zero_mul (a : K) : K.mul K.zero a = K.zero := by
  rw [K.mul_comm, K.mul_zero]

@[simp] theorem K.mul_one (a : K) : K.mul a K.one = a := by
  rcases a with ⟨a0, a1⟩
  simp only [K.mul, K.one, Fin.mul_one, Fin.mul_zero, Fin.add_zero,
    Fin.zero_add]

@[simp] theorem K.one_mul (a : K) : K.mul K.one a = a := by
  rcases a with ⟨a0, a1⟩
  simp only [K.mul, K.one, Fin.one_mul, Fin.zero_mul, Fin.mul_zero,
    Fin.add_zero]
theorem K.pow_add (point : K) (left right : Nat) :
    K.pow point (left + right) =
      K.mul (K.pow point left) (K.pow point right) := by
  induction right with
  | zero => simp [K.pow]
  | succ right inductionHypothesis =>
      rw [Nat.add_succ]
      simp only [K.pow, inductionHypothesis, K.mul_assoc]

local instance : Std.Associative K.add := ⟨K.add_assoc⟩
local instance : Std.Commutative K.add := ⟨K.add_comm⟩
local instance : Std.Associative K.mul := ⟨K.mul_assoc⟩
local instance : Std.Commutative K.mul := ⟨K.mul_comm⟩
namespace Polynomial
def eval (coefficients : List K) (point : K) : K :=
  Nightstream.SuperNeo.ProjectionCheck.eval K.ops coefficients point

@[simp] theorem eval_nil (point : K) : eval [] point = K.zero := rfl

@[simp] theorem eval_cons (head : K) (tail : List K) (point : K) :
    eval (head :: tail) point = K.add head (K.mul point (eval tail point)) := rfl
theorem eval_add (left right : List K) (point : K) :
    eval (add left right) point = K.add (eval left point) (eval right point) := by
  induction left generalizing right with
  | nil => simp [add]
  | cons leftHead leftTail inductionHypothesis =>
      cases right with
      | nil => simp [add]
      | cons rightHead rightTail =>
          simp only [add, eval_cons, inductionHypothesis, K.mul_add]
          ac_rfl
theorem eval_scale (scalar : K) (coefficients : List K) (point : K) :
    eval (scale scalar coefficients) point =
      K.mul scalar (eval coefficients point) := by
  induction coefficients with
  | nil => simp [scale]
  | cons head tail inductionHypothesis =>
      simp only [scale, eval_cons, inductionHypothesis, K.mul_add]
      ac_rfl
theorem eval_mul (left right : List K) (point : K) :
    eval (mul left right) point = K.mul (eval left point) (eval right point) := by
  induction left with
  | nil => simp [mul]
  | cons head tail inductionHypothesis =>
      simp only [mul, eval_add, eval_scale, eval_cons,
        inductionHypothesis, K.add_mul, K.mul_assoc, K.zero_add]
theorem eval_sum (polynomials : List (List K)) (point : K) :
    eval (sum polynomials) point =
      polynomials.foldr (fun polynomial suffix =>
        K.add (eval polynomial point) suffix) K.zero := by
  induction polynomials with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [sum, eval_add, List.foldr_cons, inductionHypothesis]
private theorem eval_replicate_zero (count : Nat) (point : K) :
    eval (List.replicate count K.zero) point = K.zero := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [List.replicate_succ, eval_cons, inductionHypothesis,
        K.mul_zero, K.add_zero]
theorem eval_append_zeros (coefficients : List K) (count : Nat) (point : K) :
    eval (coefficients ++ List.replicate count K.zero) point =
      eval coefficients point := by
  induction coefficients with
  | nil => exact eval_replicate_zero count point
  | cons head tail inductionHypothesis =>
      simp only [List.cons_append, eval_cons, inductionHypothesis]
theorem eval_padRight (width : Nat) (coefficients : List K) (point : K) :
    eval (padRight width coefficients) point = eval coefficients point := by
  exact eval_append_zeros coefficients (width - coefficients.length) point
private theorem eval_zero_prefix (point : K) (count : Nat)
    (tail : List K) :
    eval (List.replicate count K.zero ++ tail) point =
      K.mul (K.pow point count) (eval tail point) := by
  induction count with
  | zero => simp [K.pow]
  | succ count inductionHypothesis =>
      simp only [List.replicate_succ, List.cons_append, eval_cons,
        K.zero_add, inductionHypothesis, K.pow]
      rw [← K.mul_assoc, K.mul_comm point (K.pow point count), K.mul_assoc]
theorem dot_powersFrom (assignment : Nat → Nat) (columns : List Nat)
    (point current : K) :
    dot assignment columns (K.powersFrom point current columns.length) =
      K.mul current (eval (basePolynomial assignment columns) point) := by
  induction columns generalizing current with
  | nil =>
      change K.zero = K.mul current K.zero
      exact (K.mul_zero current).symm
  | cons head tail inductionHypothesis =>
      change K.add
          (K.mul (K.ofBase (baseAt assignment head)) current)
          (dot assignment tail
            (K.powersFrom point (K.mul current point) tail.length)) =
        K.mul current
          (K.add (K.ofBase (baseAt assignment head))
            (K.mul point
              (eval (basePolynomial assignment tail) point)))
      rw [inductionHypothesis, K.mul_add]
      ac_rfl
theorem dot_powers (assignment : Nat → Nat) (columns : List Nat)
    (point : K) :
    dot assignment columns (K.powersFrom point K.one columns.length) =
      eval (basePolynomial assignment columns) point := by
  rw [dot_powersFrom, K.one_mul]
theorem phi81_eval (point : K) :
    eval phi81 point =
      K.add K.one (K.add (K.pow point 27) (K.pow point 54)) := by
  unfold phi81
  simp only [List.cons_append, List.nil_append, eval_cons,
    List.append_assoc]
  rw [eval_zero_prefix point 26
    (K.one :: (List.replicate 26 K.zero ++ [K.one]))]
  simp only [eval_cons]
  rw [eval_zero_prefix point 26 [K.one]]
  simp only [eval_cons, eval_nil, K.mul_zero, K.add_zero, K.mul_one]
  have power27 : K.mul point (K.pow point 26) = K.pow point 27 :=
    K.mul_comm _ _
  rw [← K.mul_assoc, power27]
  rw [K.mul_add, K.mul_one]
  have square : K.mul (K.pow point 27) (K.pow point 27) =
      K.pow point 54 := by
    rw [← K.pow_add]
  rw [square]
end Polynomial


end Nightstream.Implementation.R1CS.ProjectionProgram
