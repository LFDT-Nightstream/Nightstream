import Nightstream.Implementation.R1CS.Core.CheckedProgram
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
def EvalTrace.productDefinitionsFor
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
def EvalTrace.productDefinitions
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
def negatedColumns (columns : List Nat) : List (Nat × Nat) :=
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


end Nightstream.Implementation.R1CS.ProjectionProgram
