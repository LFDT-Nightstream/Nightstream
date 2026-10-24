import CompressionLean.Checks.PoseidonRows
import CompressionLean.Spec.Runs

/-!
Owns the proof of the run check (`runsFailure`, `Spec/Runs.lean`):
`runsFailure_sound` proves that runs that pass it give F′'s matrices of the
Poseidon2 hash chain (`RunsGive`).

- `matchRuns_sum`: runs that give a list of entries give its sum at each
  column.
- `unitRows_row`: each row of a unit of `unitRows` has the coefficients of
  that row of F′'s matrix program.
- `matrixSource_row`, `matrixSource_padding`: a row of F′'s matrix program
  gives the entries of F′'s completed CCS matrices at its vertex, and a
  vertex past the rows has none.
-/

namespace CompressionLean.Checks

open CompressionLean.Spec CompressionLean.Spec.PoseidonRows CompressionLean.Setup
open NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Layout NightstreamFPrime.Layout.ProductionRelation NightstreamFPrime.Layout.MatrixProgram

/-! ## Runs against entries -/

/-- The sum of the coefficients of the entries at column `x`. -/
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
      rw [if_neg (by omega), zero_add]
  | k + 1, c, v, [], rest, found, x => by simp [matchRun] at found
  | k + 1, c, v, e :: es, rest, found, x => by
      rw [matchRun] at found
      split_ifs at found with hit
      rw [entrySum_cons, matchRun_sum r k (c + 1) (v * r) es rest found x, hit.1, hit.2, ← add_assoc]
      congr 1
      by_cases first : c = x
      · subst first
        rw [if_pos rfl, if_neg (by omega), if_pos (by omega), Nat.sub_self, pow_zero, mul_one, add_zero]
      · rw [if_neg first, zero_add]
        by_cases inside : c + 1 ≤ x ∧ x < c + 1 + k
        · rw [if_pos inside, if_pos (by omega), show x - c = x - (c + 1) + 1 by omega, pow_succ]
          ring
        · rw [if_neg inside, if_neg (by omega)]

/-- Runs that give the entries `es`, one after another, give at each column
the sum of the entries there. -/
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

/-- The entries' sum at column `x` is the form's coefficient below the width,
and zero past it. -/
theorem entrySum_eq {n : ℕ} (es : List (SparseEntry n)) (x : ℕ) :
    entrySum es x = if h : x < n then gl ((⟨es⟩ : SparseForm n).coefficient ⟨x, h⟩) else 0 := by
  split_ifs with below
  · induction es with
    | nil => rfl
    | cons e es ih =>
        rw [entrySum_cons, ih, coefficient_cons]
        by_cases same : e.column.val = x
        · rw [if_pos same, if_pos (Fin.ext same)]
          rfl
        · rw [if_neg same, if_neg fun equal => same (congrArg Fin.val equal)]
          rfl
  · induction es with
    | nil => rfl
    | cons e es ih =>
        rw [entrySum_cons, ih, if_neg (by have := e.column.isLt; omega), zero_add]

theorem normalEntries_coefficient {n : ℕ} (form : SparseForm n) (c : Fin n) :
    (⟨normalEntries form⟩ : SparseForm n).coefficient c = form.coefficient c := by
  unfold normalEntries
  split
  · rfl
  · exact normalize_coefficient form c

/-- A row that passes `rowFailure`: the runs of each matrix give the row's
entries. -/
theorem rowFailure_sum {n : ℕ} {forms : RowForms n} {runs : List Push} (pass : rowFailure forms runs = none)
    (j : Fin Spec.ProductionRelation.meaningfulPortCount) (x : ℕ) :
    ((runs.filter (·.matrix = j.val)).map (·.value x)).sum =
      if h : x < n then gl ((forms j).coefficient ⟨x, h⟩) else 0 := by
  have matched : matchRuns (runs.filter (·.matrix = j.val)) (normalEntries (forms j)) = true := by
    by_contra fails
    have none := List.findSome?_eq_none_iff.mp pass j (List.mem_finRange j)
    rw [if_neg fails] at none
    exact Option.some_ne_none _ none
  rw [matchRuns_sum _ _ matched x, entrySum_eq]
  split_ifs with below
  · rw [normalEntries_coefficient]
  · rfl

/-! ## The runs of a row -/

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

/-- The runs of row `i`, in push order. -/
theorem runsByRow_getD {rows i : ℕ} (pushes : List Push) (inside : i < rows) :
    ((runsByRow rows pushes).getD i []).reverse = pushes.filter (·.row = i) := by
  rw [runsByRow, foldl_modify_getD i pushes _ (by rwa [Array.size_replicate]), Array.getD_eq_getD_getElem?,
    Array.getElem?_eq_getElem (by rwa [Array.size_replicate]), Array.getElem_replicate]
  simp

/-! ## Units and blocks -/

/-- The first row of block `k` of a matrix program. -/
def blockStart (blocks : List MatrixProgram.Block) (k : ℕ) : ℕ :=
  ((blocks.take k).map MatrixProgram.Block.rowCount).sum

theorem select_blockStart (lw : ℕ) (sourceRow : ℕ → Option R1CS.Row) :
    ∀ (blocks : List MatrixProgram.Block) (k : ℕ) (block : MatrixProgram.Block) (o : ℕ),
      blocks[k]? = some block → o < block.rowCount →
        (MatrixProgram.Program.mk blocks).row? lw sourceRow (blockStart blocks k + o) = block.row? lw sourceRow o
  | [], _, _, _, found, _ => by simp at found
  | b :: rest, 0, block, o, found, small => by
      simp only [List.getElem?_cons_zero, Option.some.injEq] at found
      subst found
      rw [blockStart, List.take_zero, List.map_nil, List.sum_nil, Nat.zero_add]
      exact MatrixProgram.Program.cons_first_row? _ rest lw sourceRow o small
  | b :: rest, k + 1, block, o, found, small => by
      rw [List.getElem?_cons_succ] at found
      rw [blockStart, List.take_succ_cons, List.map_cons, List.sum_cons,
        MatrixProgram.Program.cons_rest_row? b rest lw sourceRow _ (by omega),
        show b.rowCount + ((rest.take k).map MatrixProgram.Block.rowCount).sum + o - b.rowCount =
          blockStart rest k + o by rw [blockStart]; omega]
      exact select_blockStart lw sourceRow rest k block o found small

/-- Row `blockStart k + o` of a matrix program is row `o` of its block `k`
(`Program.row?`). -/
theorem row?_blockStart (lw : ℕ) (sourceRow : ℕ → Option R1CS.Row)
    (program : MatrixProgram.Program) (k : ℕ) (block : MatrixProgram.Block) (o : ℕ)
    (found : program.blocks[k]? = some block) (small : o < block.rowCount) :
    program.row? lw sourceRow (blockStart program.blocks k + o) = block.row? lw sourceRow o := by
  cases program with
  | mk blocks => exact select_blockStart lw sourceRow blocks k block o found small

/-- A row of a matrix program is a row of one of its blocks. -/
theorem exists_blockStart : ∀ (blocks : List MatrixProgram.Block) (i : ℕ),
    i < (blocks.map MatrixProgram.Block.rowCount).sum →
      ∃ k block o, blocks[k]? = some block ∧ o < block.rowCount ∧ i = blockStart blocks k + o
  | [], i, inside => by simp at inside
  | b :: rest, i, inside => by
      by_cases first : i < b.rowCount
      · exact ⟨0, b, i, rfl, first, by simp [blockStart]⟩
      · obtain ⟨k, block, o, found, small, eq⟩ :=
          exists_blockStart rest (i - b.rowCount) (by simp only [List.map_cons, List.sum_cons] at inside; omega)
        refine ⟨k + 1, block, o, found, small, ?_⟩
        rw [blockStart, List.take_succ_cons, List.map_cons, List.sum_cons, ← blockStart]
        omega

/-- Every unit of every block has a job. -/
theorem mem_unitJobs : ∀ (blocks : List MatrixProgram.Block) (start k u : ℕ) (block : MatrixProgram.Block),
    blocks[k]? = some block → u < unitCount block → (block, start + blockStart blocks k, u) ∈ unitJobs start blocks
  | [], _, _, _, _, found, _ => by simp at found
  | b :: rest, start, 0, u, block, found, small => by
      simp only [List.getElem?_cons_zero, Option.some.injEq] at found
      subst found
      simp [unitJobs, blockStart, small]
  | b :: rest, start, k + 1, u, block, found, small => by
      rw [List.getElem?_cons_succ] at found
      have later := mem_unitJobs rest (start + b.rowCount) k u block found small
      rw [unitJobs]
      refine List.mem_append_right _ ?_
      rwa [blockStart, List.take_succ_cons, List.map_cons, List.sum_cons, ← blockStart, ← Nat.add_assoc]

theorem unitSize_pos (block : MatrixProgram.Block) : 0 < unitSize block := by
  cases block <;> simp [unitSize]

theorem unitSize_mul_unitCount (block : MatrixProgram.Block) : unitSize block * unitCount block = block.rowCount := by
  cases block <;> simp [unitSize, unitCount, MatrixProgram.Block.rowCount, Poseidon.Block.rowCount,
    Phi81Product.Block.rowCount, Nat.mul_comm]

/-- A unit that passes `unitFailure`: its rows, one per row of the unit, each
pass `rowFailure` against the runs of its row. -/
theorem unitFailure_rows {n : ℕ} {sourceRow : ℕ → Option R1CS.Row} {byRow : Array (List Push)}
    {block : MatrixProgram.Block} {start u : ℕ} (pass : unitFailure n sourceRow byRow (block, start, u) = none) :
    ∃ rows, unitRows n sourceRow block u = some rows ∧ rows.size = unitSize block ∧
      ∀ r, r < unitSize block →
        rowFailure (rows.getD r fun _ => .empty) (byRow.getD (start + unitSize block * u + r) []).reverse = none := by
  simp only [unitFailure] at pass
  split at pass
  · simp at pass
  · rename_i rows computed
    split_ifs at pass with size
    refine ⟨rows, computed, size, fun r small => ?_⟩
    exact Option.map_eq_none_iff.mp (List.findSome?_eq_none_iff.mp pass r (List.mem_range.mpr small))

/-! ## F′'s rows -/

section application

variable (application : Lifecycle.Stage1.Application.Program) (fits : PerApplicationFixedPoint.FitsTwoPow28 application)

/-- **The rows of a unit.** Row `r` of unit `u` of block `k` of an
application's matrix program, as `unitRows` gives it over the stored source
rows, has the coefficients of row `blockStart k + size·u + r` of F′'s
`Program.row?`. -/
theorem unitRows_row {k : ℕ} {block : MatrixProgram.Block}
    (found : (PerApplicationMatrixProgram.matrixProgram application).blocks[k]? = some block) {u : ℕ}
    {rows : Array (RowForms (PerApplicationFixedPoint.logicalWidth application))}
    (computed : unitRows (PerApplicationFixedPoint.logicalWidth application)
      ((PiDECCanonicalSourceCache.stored application)[·]?) block u = some rows)
    (size : rows.size = unitSize block) {r : ℕ} (small : r < unitSize block)
    (inside : unitSize block * u + r < block.rowCount) :
    ∃ forms, (PerApplicationMatrixProgram.matrixProgram application).row? (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationCanonicalPackage.sourceRow application fits)
        (blockStart (PerApplicationMatrixProgram.matrixProgram application).blocks k + (unitSize block * u + r)) =
          some forms ∧
      ∀ port c, ((rows.getD r fun _ => .empty) port).coefficient c = (forms port).coefficient c := by
  rw [row?_blockStart _ _ _ k block _ found inside]
  have cache : (fun s => (PiDECCanonicalSourceCache.stored application)[s]?) =
      PerApplicationCanonicalPackage.sourceRow application fits :=
    funext (PiDECCanonicalSourceCache.stored_value application fits)
  have single : (block.row? (PerApplicationFixedPoint.logicalWidth application)
      ((PiDECCanonicalSourceCache.stored application)[·]?) u).map (fun forms => #[forms]) = some rows →
      unitSize block = 1 → ∃ forms, block.row? (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationCanonicalPackage.sourceRow application fits) (unitSize block * u + r) = some forms ∧
      ∀ port c, ((rows.getD r fun _ => .empty) port).coefficient c = (forms port).coefficient c := by
    intro one unit
    obtain ⟨forms, row, rfl⟩ := Option.map_eq_some_iff.mp one
    obtain rfl : r = 0 := by omega
    rw [unit, Nat.one_mul, Nat.add_zero, ← cache]
    exact ⟨forms, row, fun _ _ => rfl⟩
  cases block with
  | ordinary b => exact single computed rfl
  | multiplicationGrid b => exact single computed rfl
  | pin b => exact single computed rfl
  | poseidon b =>
      obtain ⟨invocation, invoked, rfl⟩ := Option.map_eq_some_iff.mp computed
      obtain ⟨length, spec⟩ := invocationRows?_spec invoked
      have small' : r < invocation.size := by rw [length]; exact small
      obtain ⟨forms, row, same⟩ := spec r small'
      refine ⟨forms, row, fun port c => ?_⟩
      rw [Array.getD_eq_getD_getElem?, Array.getElem?_map, Array.getElem?_eq_getElem small']
      exact same port c
  | phi81Product b =>
      unfold unitRows productRows at computed
      simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at computed
      obtain ⟨descriptor, selected, interface, loaded, rfl⟩ := computed
      have quotient : (108 * u + r) / 108 = u := by simp [unitSize] at small; omega
      have remainder : (108 * u + r) % 108 = r := by simp [unitSize] at small; omega
      have small' : r < 108 := by simpa [unitSize] using small
      refine ⟨(Phi81ProductPlan.rowAt interface ⟨r, small'⟩).meaningfulForm, ?_, fun port c => ?_⟩
      · refine Phi81Product.Block.row?_of_loaded b _ (108 * u + r) inside descriptor (by rw [quotient]; exact selected)
          interface (by rw [← PiDECProductInterface.interface?_value]; exact loaded) _ ?_
        rw [remainder, Phi81ProductPlan.rows, List.getElem?_ofFn, dif_pos small']
      · rw [Array.getD_eq_getD_getElem?, Array.getElem?_map, Array.getElem?_ofFn, dif_pos small']
        rfl

theorem select_lt_rowCount (lw : ℕ) (sourceRow : ℕ → Option R1CS.Row) :
    ∀ (blocks : List MatrixProgram.Block) (i : ℕ) (forms : RowForms lw),
      MatrixProgram.Program.row?.select lw sourceRow blocks i = some forms →
        i < (blocks.map MatrixProgram.Block.rowCount).sum
  | [], _, _, found => by simp [MatrixProgram.Program.row?.select] at found
  | b :: rest, i, forms, found => by
      rw [MatrixProgram.Program.row?.select] at found
      simp only [List.map_cons, List.sum_cons]
      split_ifs at found with first
      · omega
      · have := select_lt_rowCount lw sourceRow rest _ forms found
        omega

theorem source_entry {cv fresh running count lw : ℕ}
    (matrices : Fin count → PaperLinearAlgebra.BooleanMatrix F cv lw)
    (polynomial : CCSResidualTable.ConstraintPolynomial F count) (j : Fin count)
    (vertex : BooleanVertex cv) (c : Fin (Phi81CarrierLayout.carrierWidth lw)) (below : c.val < lw) :
    (Phi81MatrixSource.source cv fresh running count lw matrices polynomial).matrices j vertex c =
      matrices j vertex ⟨c.val, below⟩ := by
  show Phi81CarrierLayout.extendMatrix 0 (matrices j) vertex c = _
  unfold Phi81CarrierLayout.extendMatrix Phi81CarrierLayout.logicalColumn?
  rw [dif_pos below]

theorem plan_matrix {lw : ℕ} (plan : ProductionRelation.Plan lw) (row : Fin plan.rowCount)
    (port : Fin Spec.ProductionRelation.matrixCount) (below : port.val < Spec.ProductionRelation.meaningfulPortCount)
    (column : Fin lw) :
    plan.matrix port (plan.rowLayout.toVertex row) column = (plan.forms row ⟨port.val, below⟩).coefficient column := by
  unfold ProductionRelation.Plan.matrix ProductionRelation.Plan.portForm ProductionRelation.meaningfulPort?
  rw [plan.rowLayout.toColumn_toVertex, dif_pos below]

theorem system_matrices :
    (PerApplicationFixedPoint.relation application fits).system.matrices =
      (PerApplicationFixedPoint.structuralPlan application fits).matrix := by
  rw [← PerApplicationFixedPoint.relation_matrices]
  rfl

/-- **A row of F′'s matrices.** A row of an application's matrix program
(`Program.row?`) gives row `i` of every completed CCS matrix of F′'s relation
(`PaperAlgebra.matrixSource`): matrix `j` has at column `c` the coefficient
of port `j` below the logical width, and zero past it. -/
theorem matrixSource_row {i : ℕ} {forms : RowForms (PerApplicationFixedPoint.logicalWidth application)}
    (row : (PerApplicationMatrixProgram.matrixProgram application).row? (PerApplicationFixedPoint.logicalWidth application)
      (PerApplicationCanonicalPackage.sourceRow application fits) i = some forms)
    (j : Fin productionProfile.ccsMatrices)
    (c : Fin (Phi81CarrierLayout.carrierWidth (PerApplicationFixedPoint.logicalWidth application))) :
    (Lifecycle.PaperAlgebra.matrixSource (PerApplicationFixedPoint.relation application fits).system).matrices j
        (vertexAt Lifecycle.cubeVariables i) c =
      if h : c.val < PerApplicationFixedPoint.logicalWidth application then
        (forms ⟨j.val, j.isLt⟩).coefficient ⟨c.val, h⟩
      else 0 := by
  have total : i < (PerApplicationFixedPoint.structuralPlan application fits).rowCount := by
    rw [← PerApplicationMatrixProgram.matrixProgram_rowCount_eq_structuralPlan application fits]
    exact select_lt_rowCount _ _ _ i forms row
  have formsEq : (PerApplicationFixedPoint.structuralPlan application fits).forms ⟨i, total⟩ = forms :=
    Option.some.inj ((PerApplicationCanonicalPackage.matrixProgram_row? application fits ⟨i, total⟩).symm.trans row)
  have vertex : vertexAt Lifecycle.cubeVariables i =
      (PerApplicationFixedPoint.structuralPlan application fits).rowLayout.toVertex ⟨i, total⟩ := by
    unfold vertexAt
    congr 1
    exact Fin.ext (Nat.mod_eq_of_lt (Nat.lt_of_lt_of_le total
      (PerApplicationFixedPoint.structuralPlan application fits).rowCount_le))
  rw [Lifecycle.PaperAlgebra.matrixSource, vertex]
  split_ifs with below
  · rw [source_entry _ _ _ _ _ below, system_matrices,
      plan_matrix _ _ _ (show j.val < Spec.ProductionRelation.meaningfulPortCount from j.isLt), formsEq]
  · exact Phi81MatrixSource.source_matrix_tail_zero _ _ _ _ _ _ _ _ _ _ (by omega)

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

/-- A vertex past F′'s rows has no entries. -/
theorem matrixSource_padding {i : ℕ} (past : (PerApplicationFixedPoint.structuralPlan application fits).rowCount ≤ i)
    (cube : i < 2 ^ Lifecycle.cubeVariables) (j : Fin productionProfile.ccsMatrices)
    (c : Fin (Phi81CarrierLayout.carrierWidth (PerApplicationFixedPoint.logicalWidth application))) :
    (Lifecycle.PaperAlgebra.matrixSource (PerApplicationFixedPoint.relation application fits).system).matrices j
        (vertexAt Lifecycle.cubeVariables i) c = 0 := by
  have vertex : vertexAt Lifecycle.cubeVariables i = NumericBooleanDomain.vertex Lifecycle.cubeVariables ⟨i, cube⟩ := by
    unfold vertexAt
    congr 1
    exact Fin.ext (Nat.mod_eq_of_lt cube)
  rw [Lifecycle.PaperAlgebra.matrixSource, vertex]
  by_cases below : c.val < PerApplicationFixedPoint.logicalWidth application
  · rw [source_entry _ _ _ _ _ below, system_matrices, plan_matrix_padding _ cube past]
  · exact Phi81MatrixSource.source_matrix_tail_zero _ _ _ _ _ _ _ _ _ _ (by omega)

end application

/-! ## The check -/

/-- **The run check.** Runs that pass `runsFailure` give F′'s matrices of
the Poseidon2 hash chain: they name rows of the cube, and they give every
entry of every completed CCS matrix. -/
theorem runsFailure_sound {pushes : List Push} (pass : runsFailure pushes = none) :
    RunsGive pushes Production.poseidon2Relation := by
  unfold runsFailure at pass
  dsimp only at pass
  split at pass
  · simp at pass
  · rename_i none_past
    have rows : (PerApplicationMatrixProgram.matrixProgram Poseidon2HashChainV1Package.application).rowCount =
        (PerApplicationFixedPoint.structuralPlan Poseidon2HashChainV1Package.application
          Poseidon2HashChainV1Package.fits).rowCount :=
      PerApplicationMatrixProgram.matrixProgram_rowCount_eq_structuralPlan _ _
    have cube := (PerApplicationFixedPoint.structuralPlan Poseidon2HashChainV1Package.application
      Poseidon2HashChainV1Package.fits).rowCount_le
    have below : ∀ p ∈ pushes,
        p.row < (PerApplicationMatrixProgram.matrixProgram Poseidon2HashChainV1Package.application).rowCount := by
      intro p member
      simpa using List.find?_eq_none.mp none_past p member
    rw [List.findSome?_map, List.findSome?_eq_none_iff] at pass
    refine ⟨fun p member => by have := below p member; omega, fun j i c cubeRow => ?_⟩
    change _ = gl ((Lifecycle.PaperAlgebra.matrixSource (PerApplicationFixedPoint.relation
      Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits).system).matrices j _ c)
    by_cases inside : i < (PerApplicationMatrixProgram.matrixProgram Poseidon2HashChainV1Package.application).rowCount
    · obtain ⟨k, block, o, found, small, rfl⟩ := exists_blockStart _ i inside
      have unit := unitSize_pos block
      have split := Nat.div_add_mod o (unitSize block)
      have job := mem_unitJobs _ 0 k (o / unitSize block) block found
        (Nat.div_lt_of_lt_mul (by rw [unitSize_mul_unitCount]; exact small))
      rw [Nat.zero_add] at job
      obtain ⟨units, computed, size, each⟩ := unitFailure_rows (pass _ job)
      have remainder := Nat.mod_lt o unit
      obtain ⟨forms, row, same⟩ := unitRows_row _ Poseidon2HashChainV1Package.fits found computed size remainder
        (by rw [split]; exact small)
      rw [split] at row
      have checked := each _ remainder
      rw [Nat.add_assoc, split] at checked
      rw [pushedEntry_eq, ← runsByRow_getD pushes inside, rowFailure_sum checked ⟨j.val, j.isLt⟩ c,
        matrixSource_row _ _ row j c]
      split_ifs with below
      · rw [same]
      · rfl
    · have empty : pushes.filter (·.row = i) = [] := List.filter_eq_nil_iff.mpr fun p member => by
        have := below p member
        rw [decide_eq_true_eq]
        omega
      rw [pushedEntry_eq, empty, matrixSource_padding _ _ (by omega) cubeRow j c]
      rfl

end CompressionLean.Checks
