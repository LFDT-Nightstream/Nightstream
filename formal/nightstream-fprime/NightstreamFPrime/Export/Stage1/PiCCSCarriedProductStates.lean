import NightstreamFPrime.Export.Stage1.PiCCSCarriedReadCache

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSCarriedProductStates

open NightstreamFPrime.Spec
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace
open NightstreamFPrime.Export.Stage1.PiCCSCarriedReadCache

def stateColumnKeys {columns : Nat} (interface : Phi81ProductPlan.Interface columns) :
    List Nat :=
  [interface.left, interface.right, interface.output, interface.prior,
    interface.quotient].flatMap fun state =>
    (List.ofFn state).flatMap fun form => form.entries.map fun entry => entry.column.val

def stateValues {columns : Nat} (state : Phi81ProductPlan.State columns)
    (read : Fin columns → K) : Vector K ringDegree :=
  Vector.ofFn fun lane => PiCCSSparseEvaluation.evaluateK (state lane) read

def evaluateAt (values : Vector K ringDegree) (point : F) : K :=
  ⟨Phi81Relation.QuotientProduct.evaluate (fun lane => (values.get lane).c0) point,
    Phi81Relation.QuotientProduct.evaluate (fun lane => (values.get lane).c1) point⟩

def outputAt (output prior quotient : Vector K ringDegree) (point : F) : K :=
  let modulus := Phi81Relation.QuotientProduct.modulusValue point
  ⟨(Phi81Relation.QuotientProduct.evaluate (fun lane => (output.get lane).c0) point +
      -1 * Phi81Relation.QuotientProduct.evaluate (fun lane => (prior.get lane).c0) point) +
      modulus * Phi81Relation.QuotientProduct.evaluate (fun lane => (quotient.get lane).c0) point,
    (Phi81Relation.QuotientProduct.evaluate (fun lane => (output.get lane).c1) point +
      -1 * Phi81Relation.QuotientProduct.evaluate (fun lane => (prior.get lane).c1) point) +
      modulus * Phi81Relation.QuotientProduct.evaluate (fun lane => (quotient.get lane).c1) point⟩

def rowValues (left right output prior quotient : Vector K ringDegree)
    (row : Fin 108) : Vector K Spec.ProductionRelation.matrixCount :=
  let point := Phi81Relation.QuotientProduct.node row
  Vector.ofFn fun port =>
    match meaningfulPort? port with
    | some meaningful =>
        match meaningful.val with
        | 0 => evaluateAt left point
        | 1 => evaluateAt right point
        | 2 => outputAt output prior quotient point
        | _ => ⟨0, 0⟩
    | none => ⟨0, 0⟩

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
  Vector.ofFn fun row => some (rowValues left right output prior quotient row)

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

private theorem rowValues_eq {columns : Nat} (interface : Phi81ProductPlan.Interface columns)
    (read : Fin columns → K) (row : Fin 108) :
    rowValues (stateValues interface.left read) (stateValues interface.right read)
        (stateValues interface.output read) (stateValues interface.prior read)
        (stateValues interface.quotient read) row =
      Vector.ofFn fun port : Fin Spec.ProductionRelation.matrixCount =>
        PiCCSSparseEvaluation.evaluateK ((Phi81ProductPlan.rowAt interface row).portForm port)
          read := by
  apply Vector.ext
  intro index bounded
  simp only [rowValues, Vector.getElem_ofFn]
  match index, bounded with
  | 0, _ => exact (evaluateK_evaluateForm _ _ _).symm
  | 1, _ => exact (evaluateK_evaluateForm _ _ _).symm
  | 2, _ => exact (evaluateK_outputForm _ _ _).symm
  | 3, _ => rfl
  | _ + 4, bounded => exact absurd bounded (by change ¬ _ < 4; omega)

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
    rw [dite_eq_left row.isLt]
  rw [rowSome, Option.map_some, ← rowValues_eq]
  dsimp only [invocation]
  rw [reads]
  exact get_ofFn _ _

end NightstreamFPrime.Export.Stage1.PiCCSCarriedProductStates
