import NightstreamFPrime.Export.Stage1.PiDECMatrixSparseRange
import NightstreamFPrime.Export.Stage1.PiDECNativeSparseEvaluation

/-!
Evaluate rows that share their columns by pushing the row weights through the
columns. The 108 rows of one Phi81 product invocation list the same columns in
the same order with different coefficients, so for each port and lane
`Σ_r w_r Σ_j c_rj x_j = Σ_j (Σ_r w_r c_rj) x_j`: one pass over the columns
replaces one pass per row. The column weights do not depend on the read, so
`prepare` computes them once for every source or child, as one native-word
sparse form per extension coordinate. Rows that do not share their columns use
the existing range sum. Every port and lane equals
`PiDECMatrixSparseRange.sum` for every read; no row, read or alignment premise
is needed.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECMatrixWeightedRange

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.ProductionRelation
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace (MaterializedRingK)

/-- The stored form of one port; the zero port is empty. -/
def portForm {columns : Nat} (selected : MatrixProgram.RowForms columns)
    (port : Fin matrixCount) : SparseForm columns :=
  match meaningfulPort? port with
  | some meaningful => selected meaningful
  | none => SparseForm.empty

/-- `weight · embed coefficient` without the embedded zero terms. -/
def scale (weight : K) (coefficient : F) : K :=
  ⟨weight.c0 * coefficient, weight.c1 * coefficient⟩

/-- Add each weighted coefficient of one row to the weight of its position. -/
def addRow (weight : K) : List K → List F → List K
  | total :: totals, coefficient :: coefficients =>
      K.add total (scale weight coefficient) :: addRow weight totals coefficients
  | _, _ => []

/-- The sum of position-wise products. -/
def dot : List K → List K → K
  | weight :: weights, value :: values =>
      extensionOps.add (extensionOps.mul weight value) (dot weights values)
  | _, _ => extensionOps.zero

/-- One sparse form per extension coordinate of the column weights. -/
def coordinateForms {columns : Nat} (selected : List (Fin columns)) (weights : List K) :
    SparseForm columns × SparseForm columns :=
  (⟨List.zipWith (fun column weight => ⟨column, weight.c0⟩) selected weights⟩,
    ⟨List.zipWith (fun column weight => ⟨column, weight.c1⟩) selected weights⟩)

/-- The coefficients of one row and port, or none past the rows. -/
def rowCoefficients {columns count : Nat}
    (forms : Vector (MatrixProgram.RowForms columns) count) (port : Fin matrixCount)
    (index : Nat) : List F :=
  if live : index < count then
    (portForm (forms.get ⟨index, live⟩) port).entries.map (·.coefficient)
  else []

/-- The columns of one port in the first row. -/
def columnsOf {columns count : Nat}
    (forms : Vector (MatrixProgram.RowForms columns) count) (port : Fin matrixCount) :
    List (Fin columns) :=
  if nonempty : 0 < count then
    (portForm (forms.get ⟨0, nonempty⟩) port).entries.map (·.column)
  else []

/-- Every row lists the columns of the first row, in order, in every port. -/
def aligned {columns count : Nat}
    (forms : Vector (MatrixProgram.RowForms columns) count) : Bool :=
  (List.finRange matrixCount).all fun port =>
    forms.toList.all fun row =>
      (portForm row port).entries.map (·.column) == columnsOf forms port

/-- The weighted coefficient sum of every column position of one port. -/
def portWeights {columns arity count : Nat} (firstRow : Nat) (point : CubePoint K arity)
    (forms : Vector (MatrixProgram.RowForms columns) count) (port : Fin matrixCount)
    (positions : Nat) : List K :=
  Nat.fold count (fun index _ totals =>
      addRow (PiDECEvaluationWeights.weight point (firstRow + index)) totals
        (rowCoefficients forms port index))
    (List.replicate positions extensionOps.zero)

/-- Coordinate forms of every port, shared by all reads of the same rows. -/
structure Prepared (columns : Nat) where
  aligned : Bool
  ports : Vector (SparseForm columns × SparseForm columns) matrixCount

/-- Compute the column weights once; unaligned rows keep no weights. -/
def prepare {columns arity count : Nat} (firstRow : Nat) (point : CubePoint K arity)
    (forms : Vector (MatrixProgram.RowForms columns) count) : Prepared columns :=
  if aligned forms then
    { aligned := true
      ports := Vector.ofFn fun port =>
        let selected := columnsOf forms port
        coordinateForms selected (portWeights firstRow point forms port selected.length) }
  else { aligned := false, ports := Vector.replicate matrixCount (SparseForm.empty, SparseForm.empty) }

/-- The range sum of one read from prepared column weights. -/
@[specialize] def evaluate {columns arity count : Nat} (prepared : Prepared columns) (firstRow : Nat)
    (point : CubePoint K arity) (read : Fin ringDegree → Fin columns → F)
    (forms : Vector (MatrixProgram.RowForms columns) count) :
    Vector MaterializedRingK matrixCount :=
  if prepared.aligned then
    Vector.ofFn fun port =>
      let (low, high) := prepared.ports.get port
      MaterializedRingK.ofRing fun output =>
        ⟨PiDECNativeSparseEvaluation.nativeEvalSparse low (read output),
          PiDECNativeSparseEvaluation.nativeEvalSparse high (read output)⟩
  else PiDECMatrixSparseRange.sum firstRow point read forms

private theorem get_ofFn {Alpha : Type} {size : Nat}
    (values : Fin size → Alpha) (index : Fin size) :
    (Vector.ofFn values).get index = values index := by
  change (Vector.ofFn values)[index.val] = values index
  rw [Vector.getElem_ofFn]

private theorem zero_mul (value : K) :
    extensionOps.mul extensionOps.zero value = extensionOps.zero := by
  rw [extensionLaws.mul_comm, extensionLaws.mul_zero]

private theorem add_add_add_comm (first second third fourth : K) :
    extensionOps.add (extensionOps.add first second) (extensionOps.add third fourth) =
      extensionOps.add (extensionOps.add first third) (extensionOps.add second fourth) := by
  rw [extensionLaws.add_assoc, ← extensionLaws.add_assoc second,
    extensionLaws.add_comm second third, extensionLaws.add_assoc third,
    ← extensionLaws.add_assoc first]

private theorem mul_embed (weight : K) (coefficient : F) :
    extensionOps.mul weight (K.embed coefficient) = scale weight coefficient := by
  simp only [extensionOps, K.mul, K.embed, scale, Fin.mul_zero, Fin.add_zero, Fin.zero_add]

private theorem foldl_from {columns : Nat} (read : Fin columns → F) :
    ∀ (entries : List (SparseEntry columns)) (initial : F),
      entries.foldl (fun total entry => total + entry.coefficient * read entry.column)
          initial =
        initial + entries.foldl (fun total entry =>
          total + entry.coefficient * read entry.column) 0
  | [], initial => (Fin.add_zero initial).symm
  | entry :: entries, initial => by
      simp only [List.foldl_cons]
      rw [foldl_from read entries (initial + _), foldl_from read entries (0 + _),
        Fin.zero_add]
      exact baseLaws.add_assoc _ _ _

private theorem evalSparse_cons {columns : Nat} (entry : SparseEntry columns)
    (entries : List (SparseEntry columns)) (read : Fin columns → F) :
    SparseForm.evalSparse ⟨entry :: entries⟩ read =
      entry.coefficient * read entry.column + SparseForm.evalSparse ⟨entries⟩ read := by
  unfold SparseForm.evalSparse
  rw [List.foldl_cons, foldl_from, Fin.zero_add]

/-- The two coordinate forms evaluate to the weighted dot product. -/
private theorem coordinateForms_eval {columns : Nat} (read : Fin columns → F) :
    ∀ (selected : List (Fin columns)) (weights : List K),
      selected.length = weights.length →
      (⟨(coordinateForms selected weights).1.evalSparse read,
          (coordinateForms selected weights).2.evalSparse read⟩ : K) =
        dot weights (selected.map fun column => K.embed (read column))
  | [], [], _ => rfl
  | column :: selected, weight :: weights, same => by
      have rest := coordinateForms_eval read selected weights (by simpa using same)
      simp only [coordinateForms, List.zipWith_cons_cons, List.map_cons, dot] at rest ⊢
      rw [evalSparse_cons, evalSparse_cons, ← rest, mul_embed]
      rfl
  | [], _ :: _, same => by simp at same
  | _ :: _, [], same => by simp at same

private theorem dot_replicate_zero :
    ∀ (values : List K),
      dot (List.replicate values.length extensionOps.zero) values = extensionOps.zero
  | [] => rfl
  | _ :: values => by
      simp only [List.length_cons, List.replicate_succ, dot, zero_mul,
        dot_replicate_zero values, extensionLaws.add_zero]

private theorem length_addRow (weight : K) :
    ∀ (totals : List K) (coefficients : List F), totals.length = coefficients.length →
      (addRow weight totals coefficients).length = totals.length
  | [], [], _ => rfl
  | _ :: totals, _ :: coefficients, same => by
      simp only [addRow, List.length_cons]
      rw [length_addRow weight totals coefficients (by simpa using same)]
  | [], _ :: _, same => by simp at same
  | _ :: _, [], same => by simp at same

/-- Adding one weighted row adds its weighted dot product. -/
private theorem dot_addRow (weight : K) :
    ∀ (totals : List K) (coefficients : List F) (values : List K),
      totals.length = values.length → coefficients.length = values.length →
      dot (addRow weight totals coefficients) values =
        extensionOps.add (dot totals values)
          (extensionOps.mul weight (dot (coefficients.map K.embed) values))
  | [], [], [], _, _ => by
      simp only [addRow, dot, List.map_nil, extensionLaws.mul_zero, extensionLaws.add_zero]
  | total :: totals, coefficient :: coefficients, value :: values, sameTotals,
      sameCoefficients => by
      simp only [addRow, dot, List.map_cons, ← mul_embed]
      change extensionOps.add (extensionOps.mul (extensionOps.add total
          (extensionOps.mul weight (K.embed coefficient))) value)
          (dot (addRow weight totals coefficients) values) = _
      rw [dot_addRow weight totals coefficients values (by simpa using sameTotals)
          (by simpa using sameCoefficients),
        extensionLaws.right_distrib, extensionLaws.mul_assoc,
        extensionLaws.left_distrib weight, add_add_add_comm]
  | [], _, _ :: _, same, _ => by simp at same
  | _ :: _, _, [], same, _ => by simp at same
  | _, [], _ :: _, _, same => by simp at same
  | _, _ :: _, [], _, same => by simp at same

private theorem embed_foldl {columns : Nat} (read : Fin columns → F) :
    ∀ (entries : List (SparseEntry columns)) (initial : F),
      K.embed (entries.foldl (fun total entry =>
          total + entry.coefficient * read entry.column) initial) =
        extensionOps.add (K.embed initial)
          (dot (entries.map fun entry => K.embed entry.coefficient)
            (entries.map fun entry => K.embed (read entry.column)))
  | [], initial => (extensionLaws.add_zero _).symm
  | entry :: entries, initial => by
      simp only [List.foldl_cons, List.map_cons, dot]
      rw [embed_foldl read entries, ← extensionLaws.add_assoc]
      congr 1
      exact (embed_add _ _).trans (congrArg (extensionOps.add (K.embed initial))
        (embed_mul _ _))

/-- The embedded sparse evaluation is the dot product of its embedded
coefficients and embedded column reads. -/
private theorem embed_evalSparse {columns : Nat} (form : SparseForm columns)
    (read : Fin columns → F) :
    K.embed (form.evalSparse read) =
      dot (form.entries.map fun entry => K.embed entry.coefficient)
        (form.entries.map fun entry => K.embed (read entry.column)) := by
  unfold SparseForm.evalSparse
  rw [embed_foldl, show K.embed 0 = extensionOps.zero from embed_zero, extensionLaws.zero_add]

private theorem numericSum_succ (count : Nat) (term : Nat → K) :
    NumericCompletionSum.numericSum extensionOps (count + 1) term =
      extensionOps.add (NumericCompletionSum.numericSum extensionOps count term)
        (term count) := by
  simp only [NumericCompletionSum.numericSum, Nat.fold_succ]

/-- Row-by-row weight sums give the weighted sum of the row dot products. -/
private theorem fold_dot (weight : Nat → K) (coefficients : Nat → List F)
    (values : List K) (term : Nat → K) :
    ∀ (count : Nat),
      (∀ index, index < count → (coefficients index).length = values.length) →
      (∀ index, index < count → term index = extensionOps.mul (weight index)
        (dot ((coefficients index).map K.embed) values)) →
      (Nat.fold count (fun index _ totals => addRow (weight index) totals
          (coefficients index)) (List.replicate values.length extensionOps.zero)).length =
          values.length ∧
        dot (Nat.fold count (fun index _ totals => addRow (weight index) totals
            (coefficients index)) (List.replicate values.length extensionOps.zero)) values =
          NumericCompletionSum.numericSum extensionOps count term
  | 0, _, _ => ⟨List.length_replicate, dot_replicate_zero values⟩
  | count + 1, lengths, terms => by
      obtain ⟨length, sum⟩ := fold_dot weight coefficients values term count
        (fun index bound => lengths index (by omega))
        (fun index bound => terms index (by omega))
      rw [Nat.fold_succ, numericSum_succ]
      refine ⟨?_, ?_⟩
      · rw [length_addRow _ _ _ (length.trans (lengths count (by omega)).symm), length]
      · rw [dot_addRow _ _ _ _ length (lengths count (by omega)), sum,
          terms count (by omega)]

private theorem aligned_rows {columns count : Nat}
    (forms : Vector (MatrixProgram.RowForms columns) count)
    (same : aligned forms = true) (port : Fin matrixCount) (index : Nat)
    (live : index < count) :
    (portForm (forms.get ⟨index, live⟩) port).entries.map (·.column) = columnsOf forms port := by
  unfold aligned at same
  rw [List.all_eq_true] at same
  have rows := same port (List.mem_finRange port)
  rw [List.all_eq_true] at rows
  have row := rows (forms.get ⟨index, live⟩) (by
    rw [Vector.mem_toList_iff]
    exact Vector.getElem_mem live)
  exact beq_iff_eq.mp row

/-- Prepared weights give the existing sparse range sum for every read, port,
lane and both field coordinates, whether or not the rows are aligned. -/
theorem evaluate_prepare_toRing {columns arity count : Nat} (firstRow : Nat)
    (point : CubePoint K arity) (read : Fin ringDegree → Fin columns → F)
    (forms : Vector (MatrixProgram.RowForms columns) count) (port : Fin matrixCount) :
    ((evaluate (prepare firstRow point forms) firstRow point read forms).get port).toRing =
      ((PiDECMatrixSparseRange.sum firstRow point read forms).get port).toRing := by
  by_cases same : aligned forms = true
  · simp only [evaluate, prepare, if_pos same, ↓reduceIte, get_ofFn]
    funext output
    rw [MaterializedRingK.toRing_ofRing, PiDECMatrixSparseRange.sum_value,
      PiDECNativeSparseEvaluation.nativeEvalSparse_eq_spec,
      PiDECNativeSparseEvaluation.nativeEvalSparse_eq_spec]
    have lengths : ∀ index, index < count →
        (rowCoefficients forms port index).length =
          ((columnsOf forms port).map fun column => K.embed (read output column)).length := by
      intro index live
      rw [List.length_map, ← aligned_rows forms same port index live]
      simp only [rowCoefficients, dif_pos live, List.length_map]
    have terms : ∀ index, index < count →
        extensionOps.mul (PiDECEvaluationWeights.weight point (firstRow + index))
            (K.embed (if live : index < count then
              (portForm (forms.get ⟨index, live⟩) port).evalSparse (read output) else 0)) =
          extensionOps.mul (PiDECEvaluationWeights.weight point (firstRow + index))
            (dot ((rowCoefficients forms port index).map K.embed)
              ((columnsOf forms port).map fun column => K.embed (read output column))) := by
      intro index live
      rw [dif_pos live, embed_evalSparse, ← aligned_rows forms same port index live]
      simp only [rowCoefficients, dif_pos live, List.map_map]
      rfl
    have result := fold_dot (fun index => PiDECEvaluationWeights.weight point (firstRow + index))
      (rowCoefficients forms port)
      ((columnsOf forms port).map fun column => K.embed (read output column)) _ count
      lengths terms
    rw [List.length_map] at result
    unfold portWeights
    rw [coordinateForms_eval (read output) _ _ result.1.symm]
    exact result.2
  · simp only [evaluate, prepare, if_neg same, Bool.false_eq_true, ↓reduceIte]

end NightstreamFPrime.Export.Stage1.PiDECMatrixWeightedRange
