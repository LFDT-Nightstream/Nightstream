import Nightstream.Implementation.R1CS.CheckedProgram
import Nightstream.SuperNeo.ProjectionCheck

/-!
Contract: semantic trace language for the exact PiRLC projection R1CS rows.

This module names the low-level straight-line programs emitted by
`enforce_k_mul`, `enforce_eval_at_beta`, and the final batched projection
equality.  A generated artifact may certify that these definitions and checks
occur in an exact Rust row program.  The soundness module then interprets the
trace; no trace metadata replaces or authorizes an R1CS row.
-/
namespace Nightstream.Implementation.R1CS.ProjectionProgram

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
structure KColumns where
  c0 : Nat
  c1 : Nat
deriving DecidableEq, Repr, Inhabited
structure KTerms where
  c0 : List (Nat × Nat)
  c1 : List (Nat × Nat)
deriving DecidableEq, Repr
def KTerms.ofColumns (columns : KColumns) : KTerms :=
  ⟨[(columns.c0, 1)], [(columns.c1, 1)]⟩
structure KMulTrace where
  left : KTerms
  right : KTerms
  sumLeft : List (Nat × Nat)
  sumRight : List (Nat × Nat)
  productC0 : Nat
  productC1 : Nat
  productSum : Nat
  output : KColumns
deriving DecidableEq, Repr

/-- Column-layout constructor for the exact five-column Karatsuba gadget used
by production projection checks. The artifact exporter separately verifies
the reconstructed rows against Rust; this constructor only removes repetitive
metadata from generated Lean files. -/
def KMulTrace.ofColumns (left right output : KColumns) : KMulTrace where
  left := KTerms.ofColumns left
  right := KTerms.ofColumns right
  sumLeft := [(left.c0, 1), (left.c1, 1)]
  sumRight := [(right.c0, 1), (right.c1, 1)]
  productC0 := output.c0 - 3
  productC1 := output.c0 - 2
  productSum := output.c0 - 1
  output := output

/-- The exact five SSA definitions emitted by `enforce_k_mul`: three
Karatsuba products followed by the two extension-limb outputs. -/
def KMulTrace.definitions (trace : KMulTrace) : List Definition :=
  [⟨trace.productC0, .product trace.left.c0 trace.right.c0⟩,
   ⟨trace.productC1, .product trace.left.c1 trace.right.c1⟩,
   ⟨trace.productSum,
      .product trace.sumLeft trace.sumRight⟩,
   ⟨trace.output.c0,
      .linear [(trace.productC0, 1), (trace.productC1, 7)]⟩,
   ⟨trace.output.c1,
      .linear [(trace.productSum, 1),
        (trace.productC0, goldilocksP - 1),
        (trace.productC1, goldilocksP - 1)]⟩]
def KMulTrace.SumLayoutValid (trace : KMulTrace) : Prop :=
  trace.sumLeft.Perm (trace.left.c0 ++ trace.left.c1) ∧
  trace.sumRight.Perm (trace.right.c0 ++ trace.right.c1)
instance (left right : List (Nat × Nat)) : Decidable (left.Perm right) :=
  List.decidablePerm left right
instance (trace : KMulTrace) : Decidable trace.SumLayoutValid := by
  unfold KMulTrace.SumLayoutValid
  infer_instance
structure EvalTrace where
  coefficients : List Nat
  powers : List KColumns
  products : List KColumns
  output : KColumns
deriving DecidableEq, Repr

/-- Layout constructor for `enforce_eval_at_beta`: two product columns for
every nonconstant coefficient followed by two output columns. -/
def EvalTrace.ofColumns (coefficients : List Nat) (powers : List KColumns)
    (output : KColumns) : EvalTrace where
  coefficients := coefficients
  powers := powers.take coefficients.length
  products := (List.range (coefficients.length - 1)).map fun index =>
    let start := output.c0 - 2 * (coefficients.length - 1)
    ⟨start + 2 * index, start + 2 * index + 1⟩
  output := output
private def EvalTrace.productDefinitionsFor
    (coefficients : List Nat) (powers products : List KColumns) :
    List Definition :=
  (List.zip (List.zip coefficients powers) products).flatMap
    fun entry =>
      let coefficient := entry.1.1
      let power := entry.1.2
      let product := entry.2
      [⟨product.c0,
          .product [(coefficient, 1)] [(power.c0, 1)]⟩,
       ⟨product.c1,
          .product [(coefficient, 1)] [(power.c1, 1)]⟩]
private def EvalTrace.productDefinitions
    (coefficients : List Nat) (powers products : List KColumns) :
    List Definition :=
  match coefficients, powers with
  | _ :: coefficientTail, _ :: powerTail =>
      EvalTrace.productDefinitionsFor coefficientTail powerTail products
  | _, _ => []

/-- Exact SSA definitions emitted by `enforce_eval_at_beta`.  The constant
coefficient is added directly; every later coefficient has two product wires,
then the two accumulated limbs are allocated by linear definitions. -/
def EvalTrace.definitions (trace : EvalTrace) : List Definition :=
  let constantTerms :=
    match trace.coefficients with
    | [] => []
    | coefficient :: _ => [(coefficient, 1)]
  EvalTrace.productDefinitions trace.coefficients trace.powers trace.products ++
    [⟨trace.output.c0,
        .linear (constantTerms ++ trace.products.map fun product =>
          (product.c0, 1))⟩,
     ⟨trace.output.c1,
        .linear (trace.products.map fun product => (product.c1, 1))⟩]
def EvalTrace.entries (trace : EvalTrace) :
    List ((Nat × KColumns) × KColumns) :=
  match trace.coefficients, trace.powers with
  | _ :: coefficientTail, _ :: powerTail =>
      List.zip (List.zip coefficientTail powerTail) trace.products
  | _, _ => []
def EvalTrace.LayoutValid (trace : EvalTrace) : Prop :=
  trace.coefficients ≠ [] ∧
  trace.coefficients.length = trace.powers.length ∧
  trace.products.length + 1 = trace.coefficients.length
instance (trace : EvalTrace) : Decidable trace.LayoutValid := by
  unfold EvalTrace.LayoutValid
  infer_instance
structure LadderTrace where
  beta : KColumns
  powers : List KColumns
  multiplications : List KMulTrace
deriving DecidableEq, Repr

/-- Layout constructor for the shared `β^0 .. β^top` ladder. -/
def LadderTrace.ofColumns (beta : KColumns)
    (powers : List KColumns) : LadderTrace where
  beta := beta
  powers := powers
  multiplications := (List.range (powers.length - 1)).map fun index =>
    KMulTrace.ofColumns (powers.getD index default) beta
      (powers.getD (index + 1) default)
def LadderTrace.definitions (trace : LadderTrace) : List Definition :=
  match trace.powers with
  | [] => []
  | base :: _ =>
      [⟨base.c0, .linear [(0, 1)]⟩,
       ⟨base.c1, .linear []⟩] ++
      trace.multiplications.flatMap KMulTrace.definitions
def LadderLinked (beta : KColumns) :
    List KColumns → List KMulTrace → Prop
  | [], _ => False
  | [_], [] => True
  | current :: next :: rest, multiplication :: multiplications =>
      multiplication.left = KTerms.ofColumns current ∧
      multiplication.right = KTerms.ofColumns beta ∧
      multiplication.output = next ∧
      multiplication.SumLayoutValid ∧
      LadderLinked beta (next :: rest) multiplications
  | _, _ => False
private def ladderLinkedDecidable (beta : KColumns) :
    (powers : List KColumns) → (multiplications : List KMulTrace) →
      Decidable (LadderLinked beta powers multiplications)
  | [], _ => isFalse id
  | [_], [] => isTrue trivial
  | current :: next :: rest, multiplication :: multiplications => by
      letI : Decidable (LadderLinked beta (next :: rest) multiplications) :=
        ladderLinkedDecidable beta (next :: rest) multiplications
      exact inferInstanceAs (Decidable
        (multiplication.left = KTerms.ofColumns current ∧
         multiplication.right = KTerms.ofColumns beta ∧
         multiplication.output = next ∧
         multiplication.SumLayoutValid ∧
         LadderLinked beta (next :: rest) multiplications))
  | [_], _ :: _ => isFalse id
  | _ :: _ :: _, [] => isFalse id
instance (beta : KColumns) (powers : List KColumns)
    (multiplications : List KMulTrace) :
    Decidable (LadderLinked beta powers multiplications) :=
  ladderLinkedDecidable beta powers multiplications
def LadderTrace.LayoutValid (trace : LadderTrace) : Prop :=
  LadderLinked trace.beta trace.powers trace.multiplications
instance (trace : LadderTrace) : Decidable trace.LayoutValid := by
  unfold LadderTrace.LayoutValid
  infer_instance
structure PairTrace where
  rhoColumns : List Nat
  inputColumns : List Nat
  rhoEvaluation : EvalTrace
  inputEvaluation : EvalTrace
  product : KMulTrace
deriving DecidableEq, Repr

/-- One production projection pair reconstructed from its retained source and
output columns. -/
def PairTrace.ofColumns (powers : List KColumns)
    (rhoColumns inputColumns : List Nat)
    (rhoOutput inputOutput productOutput : KColumns) : PairTrace where
  rhoColumns := rhoColumns
  inputColumns := inputColumns
  rhoEvaluation := EvalTrace.ofColumns rhoColumns powers rhoOutput
  inputEvaluation := EvalTrace.ofColumns inputColumns powers inputOutput
  product := KMulTrace.ofColumns rhoOutput inputOutput productOutput
def PairTrace.definitions (trace : PairTrace) : List Definition :=
  trace.rhoEvaluation.definitions ++
    trace.inputEvaluation.definitions ++
    trace.product.definitions
def PairTrace.LayoutValid (trace : PairTrace)
    (ladderPowers : List KColumns) : Prop :=
  trace.rhoEvaluation.LayoutValid ∧
  trace.inputEvaluation.LayoutValid ∧
  trace.rhoEvaluation.coefficients = trace.rhoColumns ∧
  trace.inputEvaluation.coefficients = trace.inputColumns ∧
  trace.rhoEvaluation.powers =
    ladderPowers.take trace.rhoColumns.length ∧
  trace.inputEvaluation.powers =
    ladderPowers.take trace.inputColumns.length ∧
  trace.product.left = KTerms.ofColumns trace.rhoEvaluation.output ∧
  trace.product.right = KTerms.ofColumns trace.inputEvaluation.output ∧
  trace.product.SumLayoutValid
instance (trace : PairTrace) (ladderPowers : List KColumns) :
    Decidable (trace.LayoutValid ladderPowers) := by
  unfold PairTrace.LayoutValid EvalTrace.LayoutValid
  infer_instance
structure ProjectionTrace where
  ladder : LadderTrace
  pairs : List PairTrace
  outputColumns : List Nat
  quotientColumns : List Nat
  outputEvaluation : EvalTrace
  quotientEvaluation : EvalTrace
  quotientPhiProduct : KMulTrace
  maxDegree : Nat
deriving DecidableEq, Repr
def phiTerms (powers : List KColumns) : KTerms :=
  let power54 := powers.getD 54 default
  let power27 := powers.getD 27 default
  ⟨[(power54.c0, 1), (power27.c0, 1), (0, 1)],
   [(power54.c1, 1), (power27.c1, 1)]⟩
def KMulTrace.quotientPhi (quotientOutput : KColumns)
    (powers : List KColumns) (output : KColumns) : KMulTrace where
  left := KTerms.ofColumns quotientOutput
  right := phiTerms powers
  sumLeft := [(quotientOutput.c0, 1), (quotientOutput.c1, 1)]
  sumRight := (phiTerms powers).c0.dropLast ++
    (phiTerms powers).c1 ++ [(0, 1)]
  productC0 := output.c0 - 3
  productC1 := output.c0 - 2
  productSum := output.c0 - 1
  output := output
def ProjectionTrace.LayoutValid (trace : ProjectionTrace) : Prop :=
  trace.ladder.LayoutValid ∧
  trace.ladder.powers.length = 55 ∧
  (∀ pair ∈ trace.pairs, pair.LayoutValid trace.ladder.powers) ∧
  trace.outputEvaluation.LayoutValid ∧
  trace.quotientEvaluation.LayoutValid ∧
  trace.outputEvaluation.coefficients = trace.outputColumns ∧
  trace.quotientEvaluation.coefficients = trace.quotientColumns ∧
  trace.outputEvaluation.powers =
    trace.ladder.powers.take trace.outputColumns.length ∧
  trace.quotientEvaluation.powers =
    trace.ladder.powers.take trace.quotientColumns.length ∧
  trace.quotientPhiProduct.left =
    KTerms.ofColumns trace.quotientEvaluation.output ∧
  trace.quotientPhiProduct.right = phiTerms trace.ladder.powers ∧
  trace.quotientPhiProduct.SumLayoutValid ∧
  trace.outputColumns.length = 54 ∧
  trace.quotientColumns.length = 53 ∧
  trace.maxDegree = 106
instance (trace : ProjectionTrace) : Decidable trace.LayoutValid := by
  unfold ProjectionTrace.LayoutValid
  infer_instance
def ProjectionTrace.definitions (trace : ProjectionTrace) : List Definition :=
  trace.ladder.definitions ++
    trace.pairs.flatMap (fun pair => pair.rhoEvaluation.definitions) ++
    trace.pairs.flatMap (fun pair =>
      pair.inputEvaluation.definitions ++ pair.product.definitions) ++
    trace.outputEvaluation.definitions ++
    trace.quotientEvaluation.definitions ++
    trace.quotientPhiProduct.definitions
private def negatedColumns (columns : List Nat) : List (Nat × Nat) :=
  columns.map fun column => (column, goldilocksP - 1)

/-- The two exact assertion rows at the end of a projection identity. -/
def ProjectionTrace.checks (trace : ProjectionTrace) : List Row :=
  let lhsC0 := trace.pairs.map fun pair => pair.product.output.c0
  let lhsC1 := trace.pairs.map fun pair => pair.product.output.c1
  [⟨lhsC0.map (fun column => (column, 1)) ++
        negatedColumns
          [trace.quotientPhiProduct.output.c0,
           trace.outputEvaluation.output.c0],
      [(0, 1)], []⟩,
   ⟨lhsC1.map (fun column => (column, 1)) ++
        negatedColumns
          [trace.quotientPhiProduct.output.c1,
           trace.outputEvaluation.output.c1],
      [(0, 1)], []⟩]

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
private theorem fadd_assoc (a b c : F) : (a + b) + c = a + (b + c) := by
  apply Fin.ext
  simp only [Fin.val_add]
  rw [Nat.mod_add_mod, Nat.add_mod_mod, Nat.add_assoc]
private theorem fadd_comm (a b : F) : a + b = b + a := by
  apply Fin.ext
  simp only [Fin.val_add, Nat.add_comm]
private theorem fmul_assoc (a b c : F) : (a * b) * c = a * (b * c) :=
  Fin.mul_assoc _ _ _
private theorem fmul_comm (a b : F) : a * b = b * a :=
  Fin.mul_comm _ _
private theorem fmul_add (a b c : F) : a * (b + c) = a * b + a * c := by
  apply Fin.ext
  simp only [Fin.val_add, Fin.val_mul]
  rw [Nat.mul_mod_mod, Nat.mul_add, ← Nat.add_mod]
private theorem fadd_mul (a b c : F) : (a + b) * c = a * c + b * c := by
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

/-! ## Exact-definition interpretation -/
def DefinitionsHold (assignment : Nat → Nat)
    (definitions : List Definition) : Prop :=
  ∀ definition ∈ definitions, definition.Holds assignment
private theorem rawLcEval_append (assignment : Nat → Nat)
    (left right : List (Nat × Nat)) :
    rawLcEval assignment (left ++ right) =
      rawLcEval assignment left + rawLcEval assignment right := by
  induction left with
  | nil => simp [rawLcEval]
  | cons head tail inductionHypothesis =>
      simp [rawLcEval, inductionHypothesis, Nat.add_assoc]
private theorem rawLcEval_perm (assignment : Nat → Nat)
    {left right : List (Nat × Nat)} (permutation : left.Perm right) :
    rawLcEval assignment left = rawLcEval assignment right := by
  induction permutation with
  | nil => rfl
  | cons _ _ inductionHypothesis => simp [rawLcEval, inductionHypothesis]
  | swap _ _ _ => simp [rawLcEval]; omega
  | trans _ _ leftHypothesis rightHypothesis =>
      exact leftHypothesis.trans rightHypothesis
private theorem termsValue_perm (assignment : Nat → Nat)
    {left right : List (Nat × Nat)} (permutation : left.Perm right) :
    residue (lcEval assignment left) = residue (lcEval assignment right) := by
  apply Fin.ext
  simp only [residue]
  rw [lcEval_eq_raw_mod, lcEval_eq_raw_mod,
    rawLcEval_perm assignment permutation]
private theorem termsValue_append (assignment : Nat → Nat)
    (left right : List (Nat × Nat)) :
    residue (lcEval assignment (left ++ right)) =
      residue (lcEval assignment left) + residue (lcEval assignment right) := by
  apply Fin.ext
  simp only [residue, Fin.val_add]
  rw [lcEval_eq_raw_mod, lcEval_eq_raw_mod, lcEval_eq_raw_mod,
    rawLcEval_append]
  simp [Nat.add_mod]
private theorem karatsuba_cross (a0 a1 b0 b1 : F) :
    ((a0 + a1) * (b0 + b1) +
        residue (goldilocksP - 1) * (a0 * b0)) +
      residue (goldilocksP - 1) * (a1 * b1) =
        a0 * b1 + a1 * b0 := by
  let rawLeft := (a0.val + a1.val) * (b0.val + b1.val) +
    (goldilocksP - 1) * (a0.val * b0.val) +
    (goldilocksP - 1) * (a1.val * b1.val)
  let rawRight := a0.val * b1.val + a1.val * b0.val
  have rawEquality : rawLeft = rawRight + goldilocksP *
      ((a0.val * b0.val) + (a1.val * b1.val)) := by
    dsimp [rawLeft, rawRight]
    simp only [Nat.add_mul, Nat.mul_add]
    unfold goldilocksP
    omega
  apply Fin.ext
  simp only [Fin.val_add, Fin.val_mul, residue]
  have modularEquality : rawLeft % goldilocksP =
      rawRight % goldilocksP := by
    rw [rawEquality, Nat.add_mul_mod_self_left]
  dsimp [rawLeft, rawRight] at modularEquality
  simpa only [Nat.add_mod, Nat.mul_mod, Nat.mod_mod] using modularEquality
private theorem negOne_mul_add_self (value : F) :
    residue (goldilocksP - 1) * value + value = 0 := by
  apply Fin.ext
  simp only [Fin.val_add, Fin.val_mul, residue]
  have raw : (goldilocksP - 1) * value.val + value.val =
      goldilocksP * value.val := by
    unfold goldilocksP
    omega
  have modular : ((goldilocksP - 1) * value.val + value.val) %
      goldilocksP = 0 := by
    rw [raw, Nat.mul_mod_right]
  simpa only [Nat.add_mod, Nat.mul_mod, Nat.mod_mod,
    Nat.mod_eq_of_lt value.isLt] using modular
private theorem solve_two_negatives (left right0 right1 : F)
    (zero : (left + residue (goldilocksP - 1) * right0) +
      residue (goldilocksP - 1) * right1 = 0) :
    left = right0 + right1 := by
  have added := congrArg (fun value => (value + right0) + right1) zero
  dsimp at added
  have rearrange :
      ((((left + residue (goldilocksP - 1) * right0) +
          residue (goldilocksP - 1) * right1) + right0) + right1) =
        (left + (residue (goldilocksP - 1) * right0 + right0)) +
          (residue (goldilocksP - 1) * right1 + right1) := by
    ac_rfl
  rw [rearrange, negOne_mul_add_self, negOne_mul_add_self,
    Fin.add_zero, Fin.zero_add] at added
  simpa only [Fin.zero_add, Fin.add_zero] using added
private theorem productDefinition_value (assignment : Nat → Nat)
    (output : Nat) (left right : List (Nat × Nat))
    (holds : assignment output =
      lcEval assignment left * lcEval assignment right % goldilocksP) :
    baseAt assignment output =
      residue (lcEval assignment left) * residue (lcEval assignment right) := by
  apply Fin.ext
  simp only [baseAt, residue, Fin.val_mul]
  rw [holds]
  simp [Nat.mul_mod]
private theorem evalProducts_sound (assignment : Nat → Nat) :
    ∀ (coefficients : List Nat) (powers products : List KColumns),
      coefficients.length = powers.length →
      coefficients.length = products.length →
      DefinitionsHold assignment
        (EvalTrace.productDefinitionsFor coefficients powers products) →
      products.map (fun product => product.value assignment) =
        (List.zip (List.zip coefficients powers) products).map fun entry =>
          K.mul (K.ofBase (baseAt assignment entry.1.1))
            (entry.1.2.value assignment) := by
  intro coefficients
  induction coefficients with
  | nil =>
      intro powers products powersLength productsLength _
      cases powers with
      | cons _ _ => simp at powersLength
      | nil =>
          cases products with
          | cons _ _ => simp at productsLength
          | nil => rfl
  | cons coefficient coefficientTail inductionHypothesis =>
      intro powers products powersLength productsLength definitionsHold
      cases powers with
      | nil => simp at powersLength
      | cons power powerTail =>
          cases products with
          | nil => simp at productsLength
          | cons product productTail =>
              have tailPowersLength : coefficientTail.length = powerTail.length := by
                simpa using powersLength
              have tailProductsLength : coefficientTail.length = productTail.length := by
                simpa using productsLength
              have productC0Holds : assignment product.c0 =
                  lcEval assignment [(coefficient, 1)] *
                    lcEval assignment [(power.c0, 1)] % goldilocksP := by
                simpa [Definition.Holds, Rhs.eval] using
                  definitionsHold
                    ⟨product.c0,
                      .product [(coefficient, 1)] [(power.c0, 1)]⟩
                    (by simp [EvalTrace.productDefinitionsFor])
              have productC1Holds : assignment product.c1 =
                  lcEval assignment [(coefficient, 1)] *
                    lcEval assignment [(power.c1, 1)] % goldilocksP := by
                simpa [Definition.Holds, Rhs.eval] using
                  definitionsHold
                    ⟨product.c1,
                      .product [(coefficient, 1)] [(power.c1, 1)]⟩
                    (by simp [EvalTrace.productDefinitionsFor])
              have productC0Value := productDefinition_value assignment product.c0
                [(coefficient, 1)] [(power.c0, 1)] productC0Holds
              have productC1Value := productDefinition_value assignment product.c1
                [(coefficient, 1)] [(power.c1, 1)] productC1Holds
              have headValue : product.value assignment =
                  K.mul (K.ofBase (baseAt assignment coefficient))
                    (power.value assignment) := by
                rcases power with ⟨powerC0, powerC1⟩
                simp only [KColumns.value, K.ofBase, K.mul, K.mk.injEq,
                  Fin.zero_mul, Fin.mul_zero, Fin.add_zero]
                constructor
                · simpa [lcEval, residue, baseAt] using productC0Value
                · simpa [lcEval, residue, baseAt] using productC1Value
              have tailDefinitionsHold : DefinitionsHold assignment
                  (EvalTrace.productDefinitionsFor coefficientTail powerTail
                    productTail) := by
                intro definition member
                apply definitionsHold definition
                simp only [EvalTrace.productDefinitionsFor, List.zip_cons_cons,
                  List.flatMap_cons, List.mem_append]
                exact Or.inr member
              simp only [List.map_cons, List.zip_cons_cons]
              rw [headValue]
              congr 1
              exact inductionHypothesis powerTail productTail
                tailPowersLength tailProductsLength tailDefinitionsHold
private theorem expectedProducts_fold_eq_dot (assignment : Nat → Nat) :
    ∀ (coefficients : List Nat) (powers products : List KColumns),
      coefficients.length = powers.length →
      coefficients.length = products.length →
      ((List.zip (List.zip coefficients powers) products).map fun entry =>
        K.mul (K.ofBase (baseAt assignment entry.1.1))
          (entry.1.2.value assignment)).foldr K.add K.zero =
        Polynomial.dot assignment coefficients
          (powers.map fun power => power.value assignment) := by
  intro coefficients
  induction coefficients with
  | nil =>
      intro powers products powersLength productsLength
      cases powers with
      | cons _ _ => simp at powersLength
      | nil =>
          cases products with
          | cons _ _ => simp at productsLength
          | nil => rfl
  | cons coefficient coefficientTail inductionHypothesis =>
      intro powers products powersLength productsLength
      cases powers with
      | nil => simp at powersLength
      | cons power powerTail =>
          cases products with
          | nil => simp at productsLength
          | cons product productTail =>
              have tailPowersLength : coefficientTail.length = powerTail.length := by
                simpa using powersLength
              have tailProductsLength : coefficientTail.length = productTail.length := by
                simpa using productsLength
              simp only [List.zip_cons_cons, List.map_cons, List.foldr_cons,
                Polynomial.dot]
              rw [inductionHypothesis powerTail productTail
                tailPowersLength tailProductsLength]
              rfl
private theorem linearDefinition2_value (assignment : Nat → Nat)
    (output left right leftCoefficient rightCoefficient : Nat)
    (holds : assignment output = lcEval assignment
      [(left, leftCoefficient), (right, rightCoefficient)]) :
    baseAt assignment output =
      residue leftCoefficient * baseAt assignment left +
      residue rightCoefficient * baseAt assignment right := by
  apply Fin.ext
  simp [baseAt, residue, lcEval, Fin.val_add, Fin.val_mul, holds]
private theorem linearDefinition3_value (assignment : Nat → Nat)
    (output first second third firstCoefficient secondCoefficient
      thirdCoefficient : Nat)
    (holds : assignment output = lcEval assignment
      [(first, firstCoefficient), (second, secondCoefficient),
       (third, thirdCoefficient)]) :
    baseAt assignment output =
      (residue firstCoefficient * baseAt assignment first +
       residue secondCoefficient * baseAt assignment second) +
      residue thirdCoefficient * baseAt assignment third := by
  apply Fin.ext
  simp [baseAt, residue, lcEval, Fin.val_add, Fin.val_mul, holds]
private theorem termsValue_columns (assignment : Nat → Nat)
    (columns : List Nat) :
    residue (lcEval assignment (columns.map fun column => (column, 1))) =
      columns.foldr (fun column suffix =>
        baseAt assignment column + suffix) 0 := by
  induction columns with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      apply Fin.ext
      simp only [List.map_cons, List.foldr_cons, Fin.val_add, residue]
      rw [lcEval_eq_raw_mod]
      simp only [rawLcEval, Nat.one_mul, Nat.mod_mod]
      have valueHypothesis := congrArg Fin.val inductionHypothesis
      simp only [residue] at valueHypothesis
      rw [lcEval_eq_raw_mod] at valueHypothesis
      simp only [Nat.mod_mod] at valueHypothesis
      rw [← valueHypothesis]
      simp only [baseAt, residue]
      rw [← Nat.add_mod]
private theorem linearColumns_value (assignment : Nat → Nat)
    (output : Nat) (columns : List Nat)
    (holds : assignment output =
      lcEval assignment (columns.map fun column => (column, 1))) :
    baseAt assignment output =
      columns.foldr (fun column suffix =>
        baseAt assignment column + suffix) 0 := by
  calc
    baseAt assignment output =
        residue (lcEval assignment (columns.map fun column => (column, 1))) := by
      apply Fin.ext
      simp only [baseAt, residue]
      rw [holds]
    _ = _ := termsValue_columns assignment columns
private theorem projectionCheckLimb_sound (assignment : Nat → Nat)
    (constantOne : assignment 0 = 1) (outputs : List Nat)
    (quotientPhi output : Nat)
    (holds : RowHolds assignment
      ⟨outputs.map (fun column => (column, 1)) ++
          [(quotientPhi, goldilocksP - 1),
           (output, goldilocksP - 1)],
       [(0, 1)], []⟩) :
    outputs.foldr (fun column suffix =>
      baseAt assignment column + suffix) 0 =
      baseAt assignment quotientPhi + baseAt assignment output := by
  have linearZero : lcEval assignment
      (outputs.map (fun column => (column, 1)) ++
        [(quotientPhi, goldilocksP - 1),
         (output, goldilocksP - 1)]) = 0 := by
    simpa [RowHolds, lcEval, constantOne] using holds
  have split := termsValue_append assignment
    (outputs.map fun column => (column, 1))
    [(quotientPhi, goldilocksP - 1),
     (output, goldilocksP - 1)]
  rw [linearZero] at split
  have outputTerms := termsValue_columns assignment outputs
  have negativeTerms : residue (lcEval assignment
      [(quotientPhi, goldilocksP - 1),
       (output, goldilocksP - 1)]) =
      residue (goldilocksP - 1) * baseAt assignment quotientPhi +
      residue (goldilocksP - 1) * baseAt assignment output := by
    apply Fin.ext
    simp [baseAt, residue, lcEval, Fin.val_add, Fin.val_mul]
  rw [outputTerms, negativeTerms] at split
  apply solve_two_negatives
  rw [fadd_assoc]
  simpa only [residue_zero] using split.symm
private theorem foldKValues (values : List K) :
    values.foldr K.add K.zero =
      ⟨(values.map K.c0).foldr (· + ·) 0,
       (values.map K.c1).foldr (· + ·) 0⟩ := by
  induction values with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.foldr_cons, List.map_cons]
      rw [inductionHypothesis]
      rfl
theorem ProjectionTrace.checks_sound (trace : ProjectionTrace)
    (assignment : Nat → Nat) (constantOne : assignment 0 = 1)
    (checksHold : Satisfies trace.checks assignment) :
    (trace.pairProductValues assignment).foldr K.add K.zero =
      K.add (trace.quotientPhiProduct.output.value assignment)
        (trace.outputEvaluation.output.value assignment) := by
  let outputC0 := trace.pairs.map fun pair => pair.product.output.c0
  let outputC1 := trace.pairs.map fun pair => pair.product.output.c1
  have rowC0 : RowHolds assignment
      ⟨outputC0.map (fun column => (column, 1)) ++
          [(trace.quotientPhiProduct.output.c0, goldilocksP - 1),
           (trace.outputEvaluation.output.c0, goldilocksP - 1)],
       [(0, 1)], []⟩ := by
    apply checksHold
    simp [ProjectionTrace.checks, negatedColumns, outputC0]
  have rowC1 : RowHolds assignment
      ⟨outputC1.map (fun column => (column, 1)) ++
          [(trace.quotientPhiProduct.output.c1, goldilocksP - 1),
           (trace.outputEvaluation.output.c1, goldilocksP - 1)],
       [(0, 1)], []⟩ := by
    apply checksHold
    simp [ProjectionTrace.checks, negatedColumns, outputC1]
  have c0 := projectionCheckLimb_sound assignment constantOne outputC0
    trace.quotientPhiProduct.output.c0 trace.outputEvaluation.output.c0 rowC0
  have c1 := projectionCheckLimb_sound assignment constantOne outputC1
    trace.quotientPhiProduct.output.c1 trace.outputEvaluation.output.c1 rowC1
  dsimp [outputC0] at c0
  dsimp [outputC1] at c1
  rw [List.foldr_map] at c0 c1
  unfold ProjectionTrace.pairProductValues
  rw [foldKValues]
  simp only [KColumns.value, K.add, K.mk.injEq, List.map_map]
  constructor
  · simpa only [List.foldr_map, Function.comp_apply] using c0
  · simpa only [List.foldr_map, Function.comp_apply] using c1
theorem EvalTrace.products_sound (trace : EvalTrace)
    (assignment : Nat → Nat) (layout : trace.LayoutValid)
    (definitionsHold : DefinitionsHold assignment trace.definitions) :
    trace.products.map (fun product => product.value assignment) =
      trace.ExpectedProducts assignment := by
  rcases trace with ⟨coefficients, powers, products, output⟩
  cases coefficients with
  | nil => exact False.elim (layout.1 rfl)
  | cons coefficient coefficientTail =>
      cases powers with
      | nil => simp [EvalTrace.LayoutValid] at layout
      | cons power powerTail =>
          have powersLength : coefficientTail.length = powerTail.length := by
            simpa [EvalTrace.LayoutValid] using layout.2.1
          have productsLength : coefficientTail.length = products.length := by
            exact (Nat.add_right_cancel layout.2.2).symm
          have productDefinitionsHold : DefinitionsHold assignment
              (EvalTrace.productDefinitionsFor coefficientTail powerTail products) := by
            intro definition member
            apply definitionsHold definition
            apply List.mem_append_left
            simpa [EvalTrace.definitions, EvalTrace.productDefinitions] using member
          exact evalProducts_sound assignment coefficientTail powerTail products
            powersLength productsLength productDefinitionsHold
theorem EvalTrace.output_value (trace : EvalTrace)
    (assignment : Nat → Nat) (layout : trace.LayoutValid)
    (definitionsHold : DefinitionsHold assignment trace.definitions) :
    trace.output.value assignment =
      K.add (K.ofBase
        (baseAt assignment (trace.coefficients.head layout.1)))
        ((trace.products.map fun product => product.value assignment).foldr
          K.add K.zero) := by
  rcases trace with ⟨coefficients, powers, products, output⟩
  cases coefficients with
  | nil => exact False.elim (layout.1 rfl)
  | cons coefficient coefficientTail =>
      have outputC0Holds : assignment output.c0 = lcEval assignment
          ((coefficient :: products.map KColumns.c0).map fun column =>
            (column, 1)) := by
        simpa [Definition.Holds, Rhs.eval, EvalTrace.definitions] using
          definitionsHold
            ⟨output.c0,
              .linear ((coefficient, 1) ::
                (products.map fun product => (product.c0, 1)))⟩
            (by simp [EvalTrace.definitions])
      have outputC1Holds : assignment output.c1 = lcEval assignment
          ((products.map KColumns.c1).map fun column => (column, 1)) := by
        simpa [Definition.Holds, Rhs.eval, EvalTrace.definitions] using
          definitionsHold
            ⟨output.c1,
              .linear (products.map fun product => (product.c1, 1))⟩
            (by simp [EvalTrace.definitions])
      have outputC0Value := linearColumns_value assignment output.c0
        (coefficient :: products.map KColumns.c0) outputC0Holds
      have outputC1Value := linearColumns_value assignment output.c1
        (products.map KColumns.c1) outputC1Holds
      change output.value assignment =
        K.add (K.ofBase (baseAt assignment coefficient))
          ((products.map fun product => product.value assignment).foldr
            K.add K.zero)
      rw [foldKValues]
      simp only [KColumns.value, K.ofBase, K.add, K.mk.injEq,
        List.map_map]
      constructor
      · simpa only [List.foldr_cons, List.foldr_map,
          Function.comp_apply] using outputC0Value
      · simpa only [List.foldr_map, Function.comp_apply,
          Fin.zero_add] using outputC1Value

/-- Exact evaluation rows compute the Horner evaluation of every committed
coefficient at the supplied power ladder. -/
theorem EvalTrace.sound (trace : EvalTrace) (assignment : Nat → Nat)
    (point : K) (layout : trace.LayoutValid)
    (powersValid : trace.PowersValid assignment point)
    (definitionsHold : DefinitionsHold assignment trace.definitions) :
    trace.output.value assignment =
      Polynomial.eval (basePolynomial assignment trace.coefficients) point := by
  have productValues := trace.products_sound assignment layout definitionsHold
  have outputValue := trace.output_value assignment layout definitionsHold
  rcases trace with ⟨coefficients, powers, products, output⟩
  cases coefficients with
  | nil => exact False.elim (layout.1 rfl)
  | cons coefficient coefficientTail =>
      cases powers with
      | nil => simp [EvalTrace.LayoutValid] at layout
      | cons power powerTail =>
          have tailPowersLength : coefficientTail.length = powerTail.length := by
            simpa using layout.2.1
          have tailProductsLength : coefficientTail.length = products.length := by
            exact (Nat.add_right_cancel layout.2.2).symm
          have powerSequence := powersValid
          simp only [EvalTrace.PowersValid, List.map_cons, List.length_cons,
            K.powersFrom, K.one_mul, List.cons.injEq] at powerSequence
          have expectedFold := expectedProducts_fold_eq_dot assignment
            coefficientTail powerTail products tailPowersLength
            tailProductsLength
          rw [outputValue, productValues]
          change K.add (K.ofBase (baseAt assignment coefficient))
              (((List.zip (List.zip coefficientTail powerTail) products).map
                fun entry =>
                  K.mul (K.ofBase (baseAt assignment entry.1.1))
                    (entry.1.2.value assignment)).foldr K.add K.zero) =
            Polynomial.eval
              (basePolynomial assignment (coefficient :: coefficientTail)) point
          rw [expectedFold, powerSequence.2, Polynomial.dot_powersFrom]
          rfl
theorem EvalTrace.powersValid_of_ladderPrefix (trace : EvalTrace)
    (assignment : Nat → Nat) (point : K) (ladder : List KColumns)
    (prefixShape : trace.powers = ladder.take trace.coefficients.length)
    (within : trace.coefficients.length ≤ ladder.length)
    (ladderValues : ladder.map (fun power => power.value assignment) =
      K.powersFrom point K.one ladder.length) :
    trace.PowersValid assignment point := by
  unfold EvalTrace.PowersValid
  rw [prefixShape, List.map_take]
  calc
    (ladder.map fun power => power.value assignment).take
        trace.coefficients.length =
        (K.powersFrom point K.one ladder.length).take
          trace.coefficients.length := by rw [ladderValues]
    _ = K.powersFrom point K.one trace.coefficients.length :=
      K.take_powersFrom point K.one within
theorem EvalTrace.coefficientLength_le_ladder (trace : EvalTrace)
    (ladder : List KColumns) (layout : trace.LayoutValid)
    (prefixShape : trace.powers = ladder.take trace.coefficients.length) :
    trace.coefficients.length ≤ ladder.length := by
  have lengths := layout.2.1.trans (congrArg List.length prefixShape)
  rw [List.length_take] at lengths
  omega

/-- The five exact Karatsuba definitions determine extension-field
multiplication.  The only layout premise says that the two sum rows contain
the same LC terms as their components; sparse ordering may differ. -/
theorem KMulTrace.sound (trace : KMulTrace) (assignment : Nat → Nat)
    (layout : trace.SumLayoutValid)
    (definitionsHold : DefinitionsHold assignment trace.definitions) :
    trace.output.value assignment =
      K.mul (trace.left.value assignment) (trace.right.value assignment) := by
  have productC0Holds : assignment trace.productC0 =
      lcEval assignment trace.left.c0 * lcEval assignment trace.right.c0 %
        goldilocksP := by
    simpa [Definition.Holds, Rhs.eval] using
      definitionsHold
        ⟨trace.productC0, .product trace.left.c0 trace.right.c0⟩
        (by simp [KMulTrace.definitions])
  have productC1Holds : assignment trace.productC1 =
      lcEval assignment trace.left.c1 * lcEval assignment trace.right.c1 %
        goldilocksP := by
    simpa [Definition.Holds, Rhs.eval] using
      definitionsHold
        ⟨trace.productC1, .product trace.left.c1 trace.right.c1⟩
        (by simp [KMulTrace.definitions])
  have productSumHolds : assignment trace.productSum =
      lcEval assignment trace.sumLeft * lcEval assignment trace.sumRight %
        goldilocksP := by
    simpa [Definition.Holds, Rhs.eval] using
      definitionsHold
        ⟨trace.productSum, .product trace.sumLeft trace.sumRight⟩
        (by simp [KMulTrace.definitions])
  have outputC0Holds : assignment trace.output.c0 = lcEval assignment
      [(trace.productC0, 1), (trace.productC1, 7)] := by
    simpa [Definition.Holds, Rhs.eval] using
      definitionsHold
        ⟨trace.output.c0,
          .linear [(trace.productC0, 1), (trace.productC1, 7)]⟩
        (by simp [KMulTrace.definitions])
  have outputC1Holds : assignment trace.output.c1 = lcEval assignment
      [(trace.productSum, 1),
       (trace.productC0, goldilocksP - 1),
       (trace.productC1, goldilocksP - 1)] := by
    simpa [Definition.Holds, Rhs.eval] using
      definitionsHold
        ⟨trace.output.c1,
          .linear [(trace.productSum, 1),
            (trace.productC0, goldilocksP - 1),
            (trace.productC1, goldilocksP - 1)]⟩
        (by simp [KMulTrace.definitions])
  have productC0Value := productDefinition_value assignment
    trace.productC0 trace.left.c0 trace.right.c0 productC0Holds
  have productC1Value := productDefinition_value assignment
    trace.productC1 trace.left.c1 trace.right.c1 productC1Holds
  have productSumValue := productDefinition_value assignment
    trace.productSum trace.sumLeft trace.sumRight productSumHolds
  have sumLeftValue : residue (lcEval assignment trace.sumLeft) =
      residue (lcEval assignment trace.left.c0) +
      residue (lcEval assignment trace.left.c1) := by
    rw [termsValue_perm assignment layout.1, termsValue_append]
  have sumRightValue : residue (lcEval assignment trace.sumRight) =
      residue (lcEval assignment trace.right.c0) +
      residue (lcEval assignment trace.right.c1) := by
    rw [termsValue_perm assignment layout.2, termsValue_append]
  have outputC0Value := linearDefinition2_value assignment
    trace.output.c0 trace.productC0 trace.productC1 1 7 outputC0Holds
  have outputC1Value := linearDefinition3_value assignment
    trace.output.c1 trace.productSum trace.productC0 trace.productC1
    1 (goldilocksP - 1) (goldilocksP - 1) outputC1Holds
  simp only [KColumns.value, KTerms.value, K.mul, K.mk.injEq]
  constructor
  · rw [productC0Value, productC1Value] at outputC0Value
    simpa only [residue_one, residue_seven, Fin.one_mul] using outputC0Value
  · rw [productSumValue, productC0Value, productC1Value,
      sumLeftValue, sumRightValue] at outputC1Value
    simp only [residue_one, Fin.one_mul] at outputC1Value
    rw [karatsuba_cross] at outputC1Value
    exact outputC1Value
theorem PairTrace.sound (trace : PairTrace) (assignment : Nat → Nat)
    (point : K) (ladder : List KColumns)
    (ladderValues : ladder.map (fun power => power.value assignment) =
      K.powersFrom point K.one ladder.length)
    (layout : trace.LayoutValid ladder)
    (definitionsHold : DefinitionsHold assignment trace.definitions) :
    trace.product.output.value assignment =
      K.mul
        (Polynomial.eval (basePolynomial assignment trace.rhoColumns) point)
        (Polynomial.eval (basePolynomial assignment trace.inputColumns) point) := by
  rcases layout with
    ⟨rhoLayout, inputLayout, rhoCoefficients, inputCoefficients,
     rhoPrefix, inputPrefix, productLeft, productRight, productLayout⟩
  have rhoDefinitionsHold : DefinitionsHold assignment
      trace.rhoEvaluation.definitions := by
    intro definition member
    apply definitionsHold definition
    simp [PairTrace.definitions, member]
  have inputDefinitionsHold : DefinitionsHold assignment
      trace.inputEvaluation.definitions := by
    intro definition member
    apply definitionsHold definition
    simp [PairTrace.definitions, member]
  have productDefinitionsHold : DefinitionsHold assignment
      trace.product.definitions := by
    intro definition member
    apply definitionsHold definition
    simp [PairTrace.definitions, member]
  have rhoPrefix' :
      trace.rhoEvaluation.powers =
        ladder.take trace.rhoEvaluation.coefficients.length := by
    rw [rhoCoefficients]
    exact rhoPrefix
  have inputPrefix' :
      trace.inputEvaluation.powers =
        ladder.take trace.inputEvaluation.coefficients.length := by
    rw [inputCoefficients]
    exact inputPrefix
  have rhoWithin := trace.rhoEvaluation.coefficientLength_le_ladder
    ladder rhoLayout rhoPrefix'
  have inputWithin := trace.inputEvaluation.coefficientLength_le_ladder
    ladder inputLayout inputPrefix'
  have rhoPowers := trace.rhoEvaluation.powersValid_of_ladderPrefix
    assignment point ladder rhoPrefix' rhoWithin ladderValues
  have inputPowers := trace.inputEvaluation.powersValid_of_ladderPrefix
    assignment point ladder inputPrefix' inputWithin ladderValues
  have rhoValue := trace.rhoEvaluation.sound assignment point rhoLayout
    rhoPowers rhoDefinitionsHold
  have inputValue := trace.inputEvaluation.sound assignment point inputLayout
    inputPowers inputDefinitionsHold
  have productValue := trace.product.sound assignment productLayout
    productDefinitionsHold
  rw [productLeft, productRight, KTerms.ofColumns_value,
    KTerms.ofColumns_value, rhoValue, inputValue,
    rhoCoefficients, inputCoefficients] at productValue
  exact productValue
private theorem ladderLinked_values (assignment : Nat → Nat)
    (beta : KColumns) :
    ∀ (current : KColumns) (rest : List KColumns)
      (multiplications : List KMulTrace) (expected : K),
      LadderLinked beta (current :: rest) multiplications →
      current.value assignment = expected →
      DefinitionsHold assignment
        (multiplications.flatMap KMulTrace.definitions) →
      (current :: rest).map (fun power => power.value assignment) =
        K.powersFrom (beta.value assignment) expected
          (current :: rest).length := by
  intro current rest
  induction rest generalizing current with
  | nil =>
      intro multiplications expected linked currentValue _
      cases multiplications with
      | nil =>
          simp only [List.map_cons, List.map_nil, List.length_cons,
            List.length_nil, K.powersFrom]
          exact congrArg (fun value => [value]) currentValue
      | cons multiplication multiplications =>
          simp [LadderLinked] at linked
  | cons next rest inductionHypothesis =>
      intro multiplications expected linked currentValue definitionsHold
      cases multiplications with
      | nil => simp [LadderLinked] at linked
      | cons multiplication multiplications =>
          simp only [LadderLinked] at linked
          rcases linked with
            ⟨leftShape, rightShape, outputShape, sumLayout, tailLinked⟩
          have multiplicationDefinitionsHold : DefinitionsHold assignment
              multiplication.definitions := by
            intro definition member
            apply definitionsHold definition
            simp [member]
          have multiplicationValue := multiplication.sound assignment sumLayout
            multiplicationDefinitionsHold
          rw [leftShape, rightShape, outputShape,
            KTerms.ofColumns_value, KTerms.ofColumns_value] at multiplicationValue
          have nextValue : next.value assignment =
              K.mul expected (beta.value assignment) := by
            rw [← currentValue]
            exact multiplicationValue
          have tailDefinitionsHold : DefinitionsHold assignment
              (multiplications.flatMap KMulTrace.definitions) := by
            intro definition member
            apply definitionsHold definition
            simp [member]
          have tailValues := inductionHypothesis next multiplications
            (K.mul expected (beta.value assignment)) tailLinked nextValue
            tailDefinitionsHold
          simp only [List.map_cons, List.length_cons, K.powersFrom]
          rw [currentValue]
          exact congrArg (List.cons expected) tailValues

/-- The exact base rows and linked K-multiplication blocks force the shared
ladder to be `1, beta, ..., beta^D`. -/
theorem LadderTrace.sound (trace : LadderTrace) (assignment : Nat → Nat)
    (constantOne : assignment 0 = 1) (layout : trace.LayoutValid)
    (definitionsHold : DefinitionsHold assignment trace.definitions) :
    trace.powers.map (fun power => power.value assignment) =
      K.powersFrom (trace.beta.value assignment) K.one trace.powers.length := by
  rcases trace with ⟨beta, powers, multiplications⟩
  cases powers with
  | nil => simp [LadderTrace.LayoutValid, LadderLinked] at layout
  | cons base rest =>
      have baseC0Holds : assignment base.c0 = lcEval assignment [(0, 1)] := by
        simpa [Definition.Holds, Rhs.eval] using
          definitionsHold ⟨base.c0, .linear [(0, 1)]⟩
            (by simp [LadderTrace.definitions])
      have baseC1Holds : assignment base.c1 = lcEval assignment [] := by
        simpa [Definition.Holds, Rhs.eval] using
          definitionsHold ⟨base.c1, .linear []⟩
            (by simp [LadderTrace.definitions])
      have baseValue : base.value assignment = K.one := by
        simp only [KColumns.value, K.one, K.mk.injEq]
        constructor
        · apply Fin.ext
          simp [baseAt, residue, lcEval, baseC0Holds, constantOne]
          decide
        · apply Fin.ext
          simp [baseAt, residue, lcEval, baseC1Holds]
      have multiplicationDefinitionsHold : DefinitionsHold assignment
          (multiplications.flatMap KMulTrace.definitions) := by
        intro definition member
        apply definitionsHold definition
        simp [LadderTrace.definitions, member]
      exact ladderLinked_values assignment beta base rest multiplications K.one
        layout baseValue multiplicationDefinitionsHold
private theorem ladderPower27 (assignment : Nat → Nat)
    (powers : List KColumns) (length : powers.length = 55) (point : K)
    (values : powers.map (fun power => power.value assignment) =
      K.powersFrom point K.one 55) :
    (powers.getD 27 default).value assignment = K.pow point 27 := by
  have bound : 27 < powers.length := by omega
  have selected := congrArg (fun values : List K => values[27]?) values
  dsimp at selected
  rw [List.getElem?_map, List.getElem?_eq_getElem bound] at selected
  have expected : (K.powersFrom point K.one 55)[27]? =
      some (K.pow point 27) := by
    simp [K.powersFrom, K.pow, K.one_mul]
  rw [expected] at selected
  have selectedValue : powers[27].value assignment = K.pow point 27 :=
    Option.some.inj selected
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem bound]
  exact selectedValue
private theorem ladderPower54 (assignment : Nat → Nat)
    (powers : List KColumns) (length : powers.length = 55) (point : K)
    (values : powers.map (fun power => power.value assignment) =
      K.powersFrom point K.one 55) :
    (powers.getD 54 default).value assignment = K.pow point 54 := by
  have bound : 54 < powers.length := by omega
  have selected := congrArg (fun values : List K => values[54]?) values
  dsimp at selected
  rw [List.getElem?_map, List.getElem?_eq_getElem bound] at selected
  have expected : (K.powersFrom point K.one 55)[54]? =
      some (K.pow point 54) := by
    simp [K.powersFrom, K.pow, K.one_mul]
  rw [expected] at selected
  have selectedValue : powers[54].value assignment = K.pow point 54 :=
    Option.some.inj selected
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem bound]
  exact selectedValue
theorem phiTerms_value (assignment : Nat → Nat) (constantOne : assignment 0 = 1)
    (powers : List KColumns) (length : powers.length = 55) (point : K)
    (values : powers.map (fun power => power.value assignment) =
      K.powersFrom point K.one 55) :
    (phiTerms powers).value assignment = Polynomial.eval Polynomial.phi81 point := by
  let power54 := powers.getD 54 default
  let power27 := powers.getD 27 default
  have termsValue : (phiTerms powers).value assignment =
      K.add (power54.value assignment)
        (K.add (power27.value assignment) K.one) := by
    change (⟨[(power54.c0, 1), (power27.c0, 1), (0, 1)],
      [(power54.c1, 1), (power27.c1, 1)]⟩ : KTerms).value assignment = _
    simp only [KTerms.value, KColumns.value, K.add, K.one, K.mk.injEq]
    constructor
    · have coefficientValues := termsValue_columns assignment
        [power54.c0, power27.c0, 0]
      simp only [List.map_cons, List.map_nil, List.foldr_cons,
        List.foldr_nil, Fin.add_zero] at coefficientValues
      rw [coefficientValues]
      have oneValue : baseAt assignment 0 = (1 : F) := by
        apply Fin.ext
        simp [baseAt, residue, constantOne]
        decide
      rw [oneValue]
    · have coefficientValues := termsValue_columns assignment
        [power54.c1, power27.c1]
      simpa only [List.map_cons, List.map_nil, List.foldr_cons,
        List.foldr_nil, Fin.add_zero] using coefficientValues
  rw [termsValue, ladderPower54 assignment powers length point values,
    ladderPower27 assignment powers length point values,
    Polynomial.phi81_eval]
  ac_rfl
end Nightstream.Implementation.R1CS.ProjectionProgram
