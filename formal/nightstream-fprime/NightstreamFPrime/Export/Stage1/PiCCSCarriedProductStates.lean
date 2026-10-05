import NightstreamFPrime.Export.Stage1.PiCCSCarriedReadCache

/-!
Carried rows of one Phi81 product invocation from its state values. Every row
evaluates the same five state polynomials (left, right, output, prior and
quotient; 54 coefficients each) at its own node, so each state coefficient is
read once per invocation instead of once for each of the 108 rows, whose forms
have 8,911 entries each. Every row and port equals the existing direct product
row under the carried read.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSCarriedProductStates

open NightstreamFPrime.Spec
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace
open NightstreamFPrime.Export.Stage1.PiCCSCarriedReadCache

/-- Columns of the five product states and of the selector column. -/
def stateColumnKeys {columns : Nat} (interface : Phi81ProductPlan.Interface columns) :
    List Nat :=
  interface.oneColumn.val ::
    ([interface.left, interface.right, interface.output, interface.prior,
      interface.quotient].flatMap fun state =>
      (List.ofFn state).flatMap fun form => form.entries.map fun entry => entry.column.val)

/-- The carried value of every coefficient of one state. -/
def stateValues {columns : Nat} (state : Phi81ProductPlan.State columns)
    (read : Fin columns → K) : Vector K ringDegree :=
  Vector.ofFn fun lane => PiCCSSparseEvaluation.evaluateK (state lane) read

/-- One state polynomial at an evaluation node, per field coordinate. -/
def evaluateAt (values : Vector K ringDegree) (point : F) : K :=
  ⟨Phi81Relation.QuotientProduct.evaluate (fun lane => (values.get lane).c0) point,
    Phi81Relation.QuotientProduct.evaluate (fun lane => (values.get lane).c1) point⟩

/-- `output - prior + Phi81(point) * quotient` at an evaluation node. -/
def outputAt (output prior quotient : Vector K ringDegree) (point : F) : K :=
  let modulus := Phi81Relation.QuotientProduct.modulusValue point
  ⟨(Phi81Relation.QuotientProduct.evaluate (fun lane => (output.get lane).c0) point +
      -1 * Phi81Relation.QuotientProduct.evaluate (fun lane => (prior.get lane).c0) point) +
      modulus * Phi81Relation.QuotientProduct.evaluate (fun lane => (quotient.get lane).c0) point,
    (Phi81Relation.QuotientProduct.evaluate (fun lane => (output.get lane).c1) point +
      -1 * Phi81Relation.QuotientProduct.evaluate (fun lane => (prior.get lane).c1) point) +
      modulus * Phi81Relation.QuotientProduct.evaluate (fun lane => (quotient.get lane).c1) point⟩

/-- The fourteen carried port values of one product row from its state values. -/
def rowValues (left right output prior quotient : Vector K ringDegree) (selector : K)
    (row : Fin 108) : Vector K Spec.ProductionRelation.matrixCount :=
  let point := Phi81Relation.QuotientProduct.node row
  Vector.ofFn fun port =>
    match meaningfulPort? port with
    | some meaningful =>
        match meaningful.val with
        | 0 => evaluateAt left point
        | 2 => evaluateAt right point
        | 4 => outputAt output prior quotient point
        | 7 => selector
        | _ => ⟨0, 0⟩
    | none => ⟨0, 0⟩

/-- Read each state coefficient once, then evaluate every row at its node. -/
def invocation {columns : Nat}
    (basis : FixedArray (Vector K ringDegree) ringDegree)
    (blocks : Nat → Vector K ringDegree)
    (interface : Phi81ProductPlan.Interface columns) :
    Vector (Option (Vector K Spec.ProductionRelation.matrixCount)) 108 :=
  let blockCache := prepareCache ((stateColumnKeys interface).map (· / ringDegree)) blocks
  let read : Fin columns → K := PiCCSCarriedRead.read basis (cachedRead blockCache blocks)
  let left := stateValues interface.left read
  let right := stateValues interface.right read
  let output := stateValues interface.output read
  let prior := stateValues interface.prior read
  let quotient := stateValues interface.quotient read
  let selector := PiCCSSparseEvaluation.evaluateK (SparseForm.singleton interface.oneColumn 1) read
  Vector.ofFn fun row => some (rowValues left right output prior quotient selector row)

private theorem get_ofFn {Alpha : Type} {size : Nat}
    (values : Fin size → Alpha) (index : Fin size) :
    (Vector.ofFn values).get index = values index := by
  change (Vector.ofFn values)[index.val] = values index
  rw [Vector.getElem_ofFn]

private theorem evaluateK_evaluateForm {columns : Nat}
    (state : Phi81ProductPlan.State columns) (point : F) (read : Fin columns → K) :
    PiCCSSparseEvaluation.evaluateK (Phi81ProductPlan.evaluateForm state point) read =
      evaluateAt (stateValues state read) point := by
  simp only [PiCCSSparseEvaluation.evaluateK, evaluateAt, stateValues, get_ofFn,
    SparseForm.evalSparse_eq_eval, Phi81ProductPlan.evaluateForm_eval]
  rfl

private theorem evaluateK_outputForm {columns : Nat}
    (interface : Phi81ProductPlan.Interface columns) (point : F) (read : Fin columns → K) :
    PiCCSSparseEvaluation.evaluateK (Phi81ProductPlan.outputForm interface point) read =
      outputAt (stateValues interface.output read) (stateValues interface.prior read)
        (stateValues interface.quotient read) point := by
  simp only [PiCCSSparseEvaluation.evaluateK, outputAt, stateValues, get_ofFn,
    Phi81ProductPlan.outputForm, SparseForm.evalSparse_eq_eval, SparseForm.add_eval,
    SparseForm.scale_eval, Phi81ProductPlan.evaluateForm_eval]
  rfl

/-- One row's values from state values are the row's carried port evaluations. -/
private theorem rowValues_eq {columns : Nat} (interface : Phi81ProductPlan.Interface columns)
    (read : Fin columns → K) (row : Fin 108) :
    rowValues (stateValues interface.left read) (stateValues interface.right read)
        (stateValues interface.output read) (stateValues interface.prior read)
        (stateValues interface.quotient read)
        (PiCCSSparseEvaluation.evaluateK (SparseForm.singleton interface.oneColumn 1) read) row =
      Vector.ofFn fun port : Fin Spec.ProductionRelation.matrixCount =>
        PiCCSSparseEvaluation.evaluateK ((Phi81ProductPlan.rowAt interface row).portForm port)
          read := by
  apply Vector.ext
  intro index bounded
  simp only [rowValues, Vector.getElem_ofFn]
  match index, bounded with
  | 0, _ => exact (evaluateK_evaluateForm _ _ _).symm
  | 2, _ => exact (evaluateK_evaluateForm _ _ _).symm
  | 4, _ => exact (evaluateK_outputForm _ _ _).symm
  | 1, _ | 3, _ | 5, _ | 6, _ | 7, _ | 8, _ | 9, _ | 10, _ | 11, _ | 12, _ | 13, _ => rfl
  | _ + 14, bounded => exact absurd bounded (by change ¬ _ < 14; omega)

/-- Every row and port equals the existing direct product row under the original
carried read. No cache, interface or row premise is used. -/
theorem invocation_value {columns : Nat}
    (basis : FixedArray (Vector K ringDegree) ringDegree)
    (blocks : Nat → Vector K ringDegree)
    (interface : Phi81ProductPlan.Interface columns) (row : Fin 108) :
    (invocation basis blocks interface).get row =
      (PiDECProductRow.row? interface row.val).map (fun selected =>
        Vector.ofFn fun port : Fin Spec.ProductionRelation.matrixCount =>
          PiCCSSparseEvaluation.evaluateK (selected.portForm port)
            (PiCCSCarriedRead.read basis blocks)) := by
  have reads : cachedRead (prepareCache ((stateColumnKeys interface).map (· / ringDegree))
      blocks) blocks = blocks :=
    funext (cachedRead_prepareCache _ blocks)
  have rowSome : PiDECProductRow.row? interface row.val =
      some (Phi81ProductPlan.rowAt interface row) := by
    unfold PiDECProductRow.row?
    rw [dif_pos row.isLt]
  rw [rowSome, Option.map_some, ← rowValues_eq]
  dsimp only [invocation]
  rw [reads]
  exact get_ofFn _ _

end NightstreamFPrime.Export.Stage1.PiCCSCarriedProductStates
