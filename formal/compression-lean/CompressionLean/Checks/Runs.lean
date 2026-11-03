import CompressionLean.Checks.PoseidonRows
import CompressionLean.Spec.Runs

namespace CompressionLean.Checks

open CompressionLean.Spec CompressionLean.Spec.PoseidonRows CompressionLean.Setup
open NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Layout NightstreamFPrime.Layout.ProductionRelation NightstreamFPrime.Layout.MatrixProgram

def entrySum {n : ℕ} (es : List (SparseEntry n)) (x : ℕ) : Gl :=
  (es.map fun e => if e.column.val = x then gl e.coefficient else 0).sum

theorem entrySum_cons {n : ℕ} (e : SparseEntry n) (es : List (SparseEntry n)) (x : ℕ) :
    entrySum (e :: es) x = (if e.column.val = x then gl e.coefficient else 0) + entrySum es x := by
  rw [entrySum, List.map_cons, List.sum_cons, entrySum]

theorem matchRun_sum {n : ℕ} (r : Gl) : ∀ (k c : ℕ) (v : Gl) (es rest : List (SparseEntry n)),
    matchRun r c k v es = some rest → ∀ x,
      entrySum es x = (if c ≤ x ∧ x < c + k then v * r ^ (x - c) else 0) + entrySum rest x
  | 0, c, v, es, rest, found, x => by
      rw [matchRun, Option.some.injEq] at found
      subst found
      rw [ite_eq_right (by omega), zero_add]
  | k + 1, c, v, [], rest, found, x => by simp [matchRun] at found
  | k + 1, c, v, e :: es, rest, found, x => by
      rw [matchRun] at found
      split_ifs at found with hit
      rw [entrySum_cons, matchRun_sum r k (c + 1) (v * r) es rest found x, hit.1, hit.2, ← add_assoc]
      congr 1
      by_cases first : c = x
      · subst first
        rw [ite_eq_left rfl, ite_eq_right (by omega), ite_eq_left (by omega), Nat.sub_self, pow_zero, mul_one, add_zero]
      · rw [ite_eq_right first, zero_add]
        by_cases inside : c + 1 ≤ x ∧ x < c + 1 + k
        · rw [ite_eq_left inside, ite_eq_left (by omega), show x - c = x - (c + 1) + 1 by omega, pow_succ]
          ring
        · rw [ite_eq_right inside, ite_eq_right (by omega)]

theorem matchRuns_sum {n : ℕ} : ∀ (ps : List Push) (es : List (SparseEntry n)), matchRuns ps es = true →
    ∀ x, (ps.map (·.value x)).sum = entrySum es x
  | [], es, found, x => by
      rw [matchRuns, List.isEmpty_iff] at found
      subst found
      rfl
  | p :: ps, es, found, x => by
      rw [matchRuns] at found
      split at found
      · rename_i rest first
        rw [List.map_cons, List.sum_cons, matchRuns_sum ps rest found x,
          matchRun_sum p.ratio p.len p.start p.initial es rest first x, Push.value]
      · simp at found

theorem entrySum_eq {n : ℕ} (es : List (SparseEntry n)) (x : ℕ) :
    entrySum es x = if h : x < n then gl ((⟨es⟩ : SparseForm n).coefficient ⟨x, h⟩) else 0 := by
  split_ifs with below
  · induction es with
    | nil => rfl
    | cons e es ih =>
        rw [entrySum_cons, ih, coefficient_cons]
        by_cases same : e.column.val = x
        · rw [ite_eq_left same, ite_eq_left (Fin.ext same)]
          rfl
        · rw [ite_eq_right same, ite_eq_right fun equal => same (congrArg Fin.val equal)]
          rfl
  · induction es with
    | nil => rfl
    | cons e es ih =>
        rw [entrySum_cons, ih, ite_eq_right (by have := e.column.isLt; omega), zero_add]

theorem normalEntries_coefficient {n : ℕ} (form : SparseForm n) (c : Fin n) :
    (⟨normalEntries form⟩ : SparseForm n).coefficient c = form.coefficient c := by
  unfold normalEntries
  split
  · rfl
  · exact normalize_coefficient form c

theorem rowFailure_sum {n : ℕ} {forms : RowForms n} {runs : List Push} (pass : rowFailure forms runs = none)
    (j : Fin Spec.ProductionRelation.meaningfulPortCount) (x : ℕ) :
    ((runs.filter (·.matrix = j.val)).map (·.value x)).sum =
      if h : x < n then gl ((forms j).coefficient ⟨x, h⟩) else 0 := by
  have matched : matchRuns (runs.filter (·.matrix = j.val)) (normalEntries (forms j)) = true := by
    by_contra fails
    have none := List.findSome?_eq_none_iff.mp pass j (List.mem_finRange j)
    rw [ite_eq_right fails] at none
    exact Option.some_ne_none _ none
  rw [matchRuns_sum _ _ matched x, entrySum_eq]
  split_ifs with below
  · rw [normalEntries_coefficient]
  · rfl

theorem pushedEntry_eq (pushes : List Push) (j i x : ℕ) :
    pushedEntry pushes j i x = (((pushes.filter (·.row = i)).filter (·.matrix = j)).map (·.value x)).sum := by
  induction pushes with
  | nil => rfl
  | cons p ps ih =>
      unfold pushedEntry at ih ⊢
      rw [List.map_cons, List.sum_cons, ih]
      by_cases row : p.row = i <;> by_cases matrix : p.matrix = j <;> simp [row, matrix]

theorem foldl_modify_getD (i : ℕ) : ∀ (pushes : List Push) (byRow : Array (List Push)), i < byRow.size →
    (pushes.foldl (fun byRow p => byRow.modify p.row (p :: ·)) byRow).getD i [] =
      (pushes.filter (·.row = i)).reverse ++ byRow.getD i []
  | [], _, _ => rfl
  | p :: ps, byRow, inside => by
      rw [List.foldl_cons, foldl_modify_getD i ps _ (by rwa [Array.size_modify])]
      rw [Array.getD_eq_getD_getElem?, Array.getElem?_modify, Array.getD_eq_getD_getElem?, List.filter_cons]
      by_cases row : p.row = i
      · simp [row, Array.getElem?_eq_getElem inside]
      · simp [row]

theorem runsByRow_getD {rows i : ℕ} (pushes : List Push) (inside : i < rows) :
    ((runsByRow rows pushes).getD i []).reverse = pushes.filter (·.row = i) := by
  rw [runsByRow, foldl_modify_getD i pushes _ (by rwa [Array.size_replicate]), Array.getD_eq_getD_getElem?,
    Array.getElem?_eq_getElem (by rwa [Array.size_replicate]), Array.getElem_replicate]
  simp

theorem source_entry {cv fresh running count lw : ℕ}
    (matrices : Fin count → PaperLinearAlgebra.BooleanMatrix F cv lw)
    (polynomial : CCSResidualTable.ConstraintPolynomial F count) (j : Fin count)
    (vertex : BooleanVertex cv) (c : Fin (Phi81CarrierLayout.carrierWidth lw)) (below : c.val < lw) :
    (Phi81MatrixSource.source cv fresh running count lw matrices polynomial).matrices j vertex c =
      matrices j vertex ⟨c.val, below⟩ := by
  show Phi81CarrierLayout.extendMatrix 0 (matrices j) vertex c = _
  unfold Phi81CarrierLayout.extendMatrix Phi81CarrierLayout.logicalColumn?
  rw [dite_eq_left below]

theorem plan_matrix {lw : ℕ} (plan : ProductionRelation.Plan lw) (row : Fin plan.rowCount)
    (port : Fin Spec.ProductionRelation.matrixCount) (below : port.val < Spec.ProductionRelation.meaningfulPortCount)
    (column : Fin lw) :
    plan.matrix port (plan.rowLayout.toVertex row) column = (plan.forms row ⟨port.val, below⟩).coefficient column := by
  unfold ProductionRelation.Plan.matrix ProductionRelation.Plan.portForm ProductionRelation.meaningfulPort?
  rw [plan.rowLayout.toColumn_toVertex, dite_eq_left below]

theorem plan_matrix_padding {lw : ℕ} (plan : ProductionRelation.Plan lw) {i : ℕ}
    (cube : i < 2 ^ Lifecycle.cubeVariables) (past : plan.rowCount ≤ i)
    (port : Fin Spec.ProductionRelation.matrixCount) (column : Fin lw) :
    plan.matrix port (NumericBooleanDomain.vertex Lifecycle.cubeVariables ⟨i, cube⟩) column = 0 := by
  have decoded : plan.rowLayout.toColumn? (NumericBooleanDomain.vertex Lifecycle.cubeVariables ⟨i, cube⟩) = none := by
    apply (NightstreamFPrime.Spec.Folding.PiCCS.CanonicalRowLayout.toColumn?_eq_none_iff Lifecycle.cubeVariables
      plan.rowCount plan.rowCount_le _).2
    simpa only [NumericBooleanDomain.index_vertex] using past
  unfold ProductionRelation.Plan.matrix
  rw [decoded]

end CompressionLean.Checks
