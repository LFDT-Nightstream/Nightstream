import CompressionLean.Checks.OptionBForms
import CompressionLean.Checks.Runs

/-! `Spec.OptionB.keyRunsFailure` gives `RunsGive` for an option-B circuit key's relation (CW-6.5,
`keyRunsFailure_sound`). The relation's plan is the package's ordinary rows, then the S-box rows of
its blocks (`CircuitRelation.plan`). An ordinary row's forms are `compileRow` through the key
layout, which the check's fast forms give (`OptionBForms`). An S-box row's forms are the Poseidon2
trace through the block's interface, which the shared atom rows give (`rep_trace`). -/

namespace CompressionLean.Checks.OptionB

open CompressionLean.Spec CompressionLean.Spec.OptionB CompressionLean.Spec.PoseidonRows CompressionLean.Setup
open CompressionLean.Checks
open NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Layout NightstreamFPrime.Layout.ProductionRelation NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Export NightstreamFPrime.Export.Package NightstreamFPrime.Export.Generic NightstreamFPrime.Export.OptionB

/-! ### A plan's relation -/

theorem planSource_row {lw : ℕ}
    {pf : ringDegree * Lifecycle.PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth lw}
    (plan : ProductionRelation.Plan lw) (cubeFits : Phi81CarrierLayout.carrierWidth lw ≤ 2 ^ Lifecycle.cubeVariables)
    {i : ℕ} (inside : i < plan.rowCount) (j : Fin productionProfile.ccsMatrices)
    (c : Fin (Phi81CarrierLayout.carrierWidth lw)) :
    (Lifecycle.PaperAlgebra.matrixSource (plan.logicalRelation (publicFits := pf) cubeFits).system).matrices j
        (vertexAt Lifecycle.cubeVariables i) c =
      if h : c.val < lw then (plan.forms ⟨i, inside⟩ ⟨j.val, j.isLt⟩).coefficient ⟨c.val, h⟩ else 0 := by
  have vertex : vertexAt Lifecycle.cubeVariables i = plan.rowLayout.toVertex ⟨i, inside⟩ := by
    unfold vertexAt
    congr 1
    exact Fin.ext (Nat.mod_eq_of_lt (Nat.lt_of_lt_of_le inside plan.rowCount_le))
  rw [Lifecycle.PaperAlgebra.matrixSource, vertex]
  split_ifs with below
  · rw [source_entry _ _ _ _ _ below, Plan.logicalRelation_system_matrices,
      plan_matrix _ _ _ (show j.val < Spec.ProductionRelation.meaningfulPortCount from j.isLt)]
  · exact Phi81MatrixSource.source_matrix_tail_zero _ _ _ _ _ _ _ _ _ _ (by omega)

theorem planSource_padding {lw : ℕ}
    {pf : ringDegree * Lifecycle.PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth lw}
    (plan : ProductionRelation.Plan lw) (cubeFits : Phi81CarrierLayout.carrierWidth lw ≤ 2 ^ Lifecycle.cubeVariables)
    {i : ℕ} (past : plan.rowCount ≤ i) (cube : i < 2 ^ Lifecycle.cubeVariables)
    (j : Fin productionProfile.ccsMatrices) (c : Fin (Phi81CarrierLayout.carrierWidth lw)) :
    (Lifecycle.PaperAlgebra.matrixSource (plan.logicalRelation (publicFits := pf) cubeFits).system).matrices j
        (vertexAt Lifecycle.cubeVariables i) c = 0 := by
  have vertex : vertexAt Lifecycle.cubeVariables i = NumericBooleanDomain.vertex Lifecycle.cubeVariables ⟨i, cube⟩ := by
    unfold vertexAt
    congr 1
    exact Fin.ext (Nat.mod_eq_of_lt cube)
  rw [Lifecycle.PaperAlgebra.matrixSource, vertex]
  by_cases below : c.val < lw
  · rw [source_entry _ _ _ _ _ below, Plan.logicalRelation_system_matrices, plan_matrix_padding _ cube past]
  · exact Phi81MatrixSource.source_matrix_tail_zero _ _ _ _ _ _ _ _ _ _ (by omega)

theorem appendSource_left {lw : ℕ}
    {pf : ringDegree * Lifecycle.PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth lw}
    (left right : ProductionRelation.Plan lw) (le : left.rowCount + right.rowCount ≤ 2 ^ Lifecycle.cubeVariables)
    (cubeFits : Phi81CarrierLayout.carrierWidth lw ≤ 2 ^ Lifecycle.cubeVariables)
    {i : ℕ} (inside : i < left.rowCount) (j : Fin productionProfile.ccsMatrices)
    (c : Fin (Phi81CarrierLayout.carrierWidth lw)) :
    (Lifecycle.PaperAlgebra.matrixSource ((Plan.append left right le).logicalRelation (publicFits := pf)
        cubeFits).system).matrices j (vertexAt Lifecycle.cubeVariables i) c =
      if h : c.val < lw then (left.forms ⟨i, inside⟩ ⟨j.val, j.isLt⟩).coefficient ⟨c.val, h⟩ else 0 := by
  rw [planSource_row _ _ (show i < (Plan.append left right le).rowCount by rw [Plan.append_rowCount]; omega)]
  split_ifs with h
  · rw [show (⟨i, _⟩ : Fin (Plan.append left right le).rowCount) =
      Plan.leftIndex left.rowCount right.rowCount ⟨i, inside⟩ from rfl, Plan.append_forms_left]
  · rfl

theorem appendSource_right {lw : ℕ}
    {pf : ringDegree * Lifecycle.PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth lw}
    (left right : ProductionRelation.Plan lw) (le : left.rowCount + right.rowCount ≤ 2 ^ Lifecycle.cubeVariables)
    (cubeFits : Phi81CarrierLayout.carrierWidth lw ≤ 2 ^ Lifecycle.cubeVariables)
    {i : ℕ} (inside : i < right.rowCount) (j : Fin productionProfile.ccsMatrices)
    (c : Fin (Phi81CarrierLayout.carrierWidth lw)) :
    (Lifecycle.PaperAlgebra.matrixSource ((Plan.append left right le).logicalRelation (publicFits := pf)
        cubeFits).system).matrices j (vertexAt Lifecycle.cubeVariables (left.rowCount + i)) c =
      if h : c.val < lw then (right.forms ⟨i, inside⟩ ⟨j.val, j.isLt⟩).coefficient ⟨c.val, h⟩ else 0 := by
  rw [planSource_row _ _ (show left.rowCount + i < (Plan.append left right le).rowCount by
    rw [Plan.append_rowCount]; omega)]
  split_ifs with h
  · rw [show (⟨left.rowCount + i, _⟩ : Fin (Plan.append left right le).rowCount) =
      Plan.rightIndex left.rowCount right.rowCount ⟨i, inside⟩ from rfl, Plan.append_forms_right]
  · rfl

theorem indexed_forms_at {lw blockCount : ℕ}
    (forms : Fin blockCount → Fin 150 → Fin Spec.ProductionRelation.meaningfulPortCount → SparseForm lw)
    (le : blockCount * 150 ≤ 2 ^ Lifecycle.cubeVariables) {k r : ℕ} (block : k < blockCount) (row : r < 150)
    (inside : 150 * k + r < (Plan.indexed forms le).rowCount) (port : Fin Spec.ProductionRelation.meaningfulPortCount) :
    (Plan.indexed forms le).forms ⟨150 * k + r, inside⟩ port = forms ⟨k, block⟩ ⟨r, row⟩ port := by
  have decoded : Fin.decodeProd (⟨150 * k + r, inside⟩ : Fin (blockCount * 150)) = (⟨k, block⟩, ⟨r, row⟩) := by
    ext
    · show (150 * k + r) / 150 = k
      omega
    · show (150 * k + r) % 150 = r
      omega
  simp only [Plan.indexed, decoded]

/-! ### The check's rows -/

section package

variable {p : StepPackage.Package} (logical : ProductionAssignment.logicalWidth (slotsOf p) ≤ StepRelation.keyWidth)
  (increase : sourcesIncrease (tableOf (slotsOf p)).slots = true)

include increase

theorem fastMap_form {c : ℕ} (inside : c < p.width) (confirmedAt : (fastForm (tableOf (slotsOf p)) c).isSome = true) :
    (fastMap (tableOf (slotsOf p))).form ⟨c, inside⟩ = (keyLayout p logical).map.form ⟨c, inside⟩ := by
  obtain ⟨f, computed⟩ := Option.isSome_iff_exists.mp confirmedAt
  show (fastForm _ c).getD _ = _
  rw [computed, Option.getD_some]
  exact fastForm_eq logical increase inside computed

theorem fastFormAt_eq {c : ℕ} (confirmedAt : c < p.width → (fastForm (tableOf (slotsOf p)) c).isSome = true) :
    fastFormAt (tableOf (slotsOf p)) c = (keyLayout p logical).form c := by
  unfold fastFormAt LogicalPlan.Layout.form
  split_ifs with inside
  · exact fastMap_form logical increase inside (confirmedAt inside)
  · rfl

theorem fastTerms_eq : ∀ terms : List (ℕ × F),
    (∀ term ∈ terms, term.1 < p.width → (fastForm (tableOf (slotsOf p)) term.1).isSome = true) →
    terms.foldr (fun term rest => .add (.scale term.2 (fastFormAt (tableOf (slotsOf p)) term.1)) rest) .empty =
      (keyLayout p logical).terms terms
  | [], _ => rfl
  | (column, coefficient) :: rest, confirmedAt => by
      simp only [List.foldr_cons, LogicalPlan.Layout.terms]
      rw [fastFormAt_eq logical increase (confirmedAt (column, coefficient) (by simp)),
        fastTerms_eq rest fun term member => confirmedAt term (by simp [member])]

theorem fastCombination_eq (combination : R1CS.LinearCombination)
    (confirmedAt : ∀ c ∈ termColumns combination, c < p.width → (fastForm (tableOf (slotsOf p)) c).isSome = true) :
    fastCombination (tableOf (slotsOf p)) combination = (keyLayout p logical).combination combination := by
  unfold fastCombination LogicalPlan.Layout.combination
  rw [fastTerms_eq logical increase combination.terms fun term member =>
    confirmedAt term.1 (List.mem_map_of_mem member)]
  rfl

theorem compileCombination_fast (combination : R1CS.LinearCombination)
    (fastBounded slowBounded : SourceCompiler.CombinationBounded p.width combination)
    (each : ∀ c ∈ termColumns combination, (fastForm (tableOf (slotsOf p)) c).isSome = true) :
    SourceCompiler.compileCombination (fastMap (tableOf (slotsOf p))) oneColumn combination fastBounded =
      SourceCompiler.compileCombination (keyLayout p logical).map (keyLayout p logical).one combination slowBounded := by
  unfold SourceCompiler.compileCombination
  rw [compileTerms_congr combination.terms fastBounded slowBounded fun term member inside =>
    fastMap_form logical increase inside (each term.1 (List.mem_map_of_mem member))]
  rfl

/-- An ordinary row that the check accepts has runs that match its forms under the key layout. -/
theorem ordinary_rows {row : R1CS.Row} {runs : List Push}
    (pass : ordinaryFailure (tableOf (slotsOf p)) row runs = none) :
    ∃ bounded : SourceCompiler.RowBounded p.width row,
      rowFailure (SourceCompiler.compileRow (keyLayout p logical).map (keyLayout p logical).one row
        bounded).meaningfulForm (sortedRuns runs) = none := by
  unfold ordinaryFailure at pass
  split_ifs at pass with bounded confirmedRow
  refine ⟨bounded, ?_⟩
  have each : ∀ c ∈ termColumns row.a ++ termColumns row.b ++ termColumns row.c,
      (fastForm (tableOf (slotsOf p)) c).isSome = true := List.all_eq_true.mp confirmedRow
  have same : SourceCompiler.compileRow (fastMap (tableOf (slotsOf p))) oneColumn row bounded =
      SourceCompiler.compileRow (keyLayout p logical).map (keyLayout p logical).one row bounded := by
    unfold SourceCompiler.compileRow
    rw [compileCombination_fast logical increase row.a _ _ fun c member => each c (by simp [member]),
      compileCombination_fast logical increase row.b _ _ fun c member => each c (by simp [member]),
      compileCombination_fast logical increase row.c _ _ fun c member => each c (by simp [member])]
  rw [← same]
  simpa using pass

/-- The interface of block `k` under the key layout (`CircuitRelation.sboxInterface`). -/
abbrev blockInterface (blocks : List Frame.Block) (sboxStart : ℕ) {k : ℕ} (inside : k < blocks.length) :
    PoseidonSboxPlan.Interface StepRelation.keyWidth :=
  PoseidonSboxFamilyPlan.invocationInterface (CircuitRelation.sboxInterface (keyLayout p logical) blocks sboxStart)
    ⟨k, inside⟩

theorem blockAtoms_form {blocks : List Frame.Block} {sboxStart k : ℕ} (inside : k < blocks.length)
    (each : ∀ c ∈ (blockColumns blocks[k] sboxStart k).filter (· < p.width),
      (fastForm (tableOf (slotsOf p)) c).isSome = true)
    (a : ℕ) (small : a < 167) (c : Fin StepRelation.keyWidth) :
    ((blockAtoms (tableOf (slotsOf p)) blocks[k] sboxStart k).form a).coefficient c =
      (atomOf (blockInterface logical blocks sboxStart inside) a).coefficient c := by
  have confirmedAt : ∀ x ∈ blockColumns blocks[k] sboxStart k, x < p.width →
      (fastForm (tableOf (slotsOf p)) x).isSome = true :=
    fun x member below => each x (List.mem_filter.mpr ⟨member, by simpa using below⟩)
  unfold Atoms.form atomOf
  by_cases lane : a < 16
  · rw [ite_eq_left lane, dite_eq_left lane]
    simp only [blockAtoms, Vector.getElem?_ofFn, dite_eq_left lane, Option.getD_some]
    rw [normalize_coefficient, fastCombination_eq logical increase _ fun x member below =>
      confirmedAt x (List.mem_append_left _ (List.mem_flatMap.mpr ⟨a, List.mem_range.mpr lane, member⟩)) below]
    rfl
  · rw [ite_eq_right lane, dite_eq_right lane]
    by_cases sbox : a < 166
    · have row : a - 16 < 150 := by omega
      rw [ite_eq_left sbox, ite_eq_left sbox]
      simp only [blockAtoms, Vector.getElem?_ofFn, dite_eq_left row, Option.getD_some]
      rw [show (⟨(fastFormAt (tableOf (slotsOf p)) (CircuitRelation.sboxColumn sboxStart k (a - 16))).entries⟩ :
          SparseForm StepRelation.keyWidth) = fastFormAt (tableOf (slotsOf p)) (CircuitRelation.sboxColumn sboxStart k (a - 16))
          from rfl,
        fastFormAt_eq logical increase fun below => confirmedAt _ (List.mem_append_right _
          (List.mem_map.mpr ⟨a - 16, List.mem_range.mpr row, rfl⟩)) below,
        PoseidonSboxPlan.sboxOutputAt, dite_eq_left (by rw [PoseidonRetainedSlots.rows_length]; exact row)]
      rfl
    · rw [ite_eq_right sbox, ite_eq_right sbox]
      rfl

omit increase in
theorem mergeRuns_sum (j x : ℕ) : ∀ (current : Push) (rest : List Push),
    (((mergeRuns current rest).filter (·.matrix = j)).map (·.value x)).sum =
      (((current :: rest).filter (·.matrix = j)).map (·.value x)).sum
  | _, [] => rfl
  | current, next :: rest => by
      unfold mergeRuns
      split_ifs with same
      · obtain ⟨matrix, start, len, ratio⟩ := same
        rw [mergeRuns_sum j x _ rest]
        have value : ({ current with initial := current.initial + next.initial } : Push).value x =
            current.value x + next.value x := by
          unfold Push.value
          dsimp only
          rw [← start, ← len, ← ratio]
          split_ifs
          · ring
          · ring
        by_cases hit : current.matrix = j
        · have hit' : next.matrix = j := matrix ▸ hit
          rw [List.filter_cons_of_pos (by simpa using hit), List.filter_cons_of_pos (by simpa using hit),
            List.filter_cons_of_pos (by simpa using hit')]
          simp only [List.map_cons, List.sum_cons, value, add_assoc]
        · have hit' : ¬next.matrix = j := matrix ▸ hit
          rw [List.filter_cons_of_neg (by simpa using hit), List.filter_cons_of_neg (by simpa using hit),
            List.filter_cons_of_neg (by simpa using hit')]
      · rw [List.filter_cons, List.filter_cons]
        by_cases hit : current.matrix = j
        · simp only [hit, decide_true, ite_true, List.map_cons, List.sum_cons]
          rw [mergeRuns_sum j x next rest, List.filter_cons]
        · simp only [hit, decide_false, Bool.false_eq_true, ite_false]
          rw [mergeRuns_sum j x next rest, List.filter_cons]

omit increase in
theorem sortedRuns_sum (runs : List Push) (j x : ℕ) :
    (((sortedRuns runs).filter (·.matrix = j)).map (·.value x)).sum = ((runs.filter (·.matrix = j)).map (·.value x)).sum := by
  have nonzero : ∀ l : List Push, (((l.filter (·.initial ≠ 0)).filter (·.matrix = j)).map (·.value x)).sum =
      ((l.filter (·.matrix = j)).map (·.value x)).sum := by
    intro l
    induction l with
    | nil => rfl
    | cons p l ih =>
        by_cases zero : p.initial = 0
        · have value : p.value x = 0 := by unfold Push.value; split_ifs <;> simp [zero]
          rw [List.filter_cons_of_neg (by simpa using zero), ih]
          by_cases hit : p.matrix = j
          · rw [List.filter_cons_of_pos (by simpa using hit)]
            simp [value]
          · rw [List.filter_cons_of_neg (by simpa using hit)]
        · rw [List.filter_cons_of_pos (by simpa using zero)]
          by_cases hit : p.matrix = j
          · rw [List.filter_cons_of_pos (by simpa using hit), List.filter_cons_of_pos (by simpa using hit),
              List.map_cons, List.map_cons, List.sum_cons, List.sum_cons, ih]
          · rw [List.filter_cons_of_neg (by simpa using hit), List.filter_cons_of_neg (by simpa using hit), ih]
  unfold sortedRuns
  rw [nonzero]
  rw [← (((List.mergeSort_perm runs runBefore).filter _).map _).sum_eq]
  split
  · rename_i empty
    rw [empty]
  · rename_i first rest sorted
    rw [mergeRuns_sum, ← sorted]

omit increase in
/-- The runs of a row whose forms the check accepts sum to the coefficients of forms with the
same coefficients. -/
theorem rowSum_congr {n : ℕ} {forms other : RowForms n} {runs : List Push} (pass : rowFailure forms runs = none)
    (same : ∀ j c, (forms j).coefficient c = (other j).coefficient c)
    (j : Fin Spec.ProductionRelation.meaningfulPortCount) (x : ℕ) :
    ((runs.filter (·.matrix = j.val)).map (·.value x)).sum =
      if h : x < n then gl ((other j).coefficient ⟨x, h⟩) else 0 := by
  rw [rowFailure_sum pass j x]
  split_ifs
  · rw [same]
  · rfl

/-- An S-box row of a block that the check accepts has runs that sum to its forms' coefficients
under the key layout. -/
theorem block_rows {blocks : List Frame.Block} {sboxStart k : ℕ} (inside : k < blocks.length)
    {runs : ℕ → List Push} (pass : blockFailure (tableOf (slotsOf p)) blocks[k] sboxStart k runs = none)
    {r : ℕ} (small : r < 150) (j : Fin Spec.ProductionRelation.meaningfulPortCount) (x : ℕ) :
    (((runs r).filter (·.matrix = j.val)).map (·.value x)).sum =
      if h : x < StepRelation.keyWidth then
        gl ((PoseidonSboxFamilyPlan.rowForms (CircuitRelation.sboxInterface (keyLayout p logical) blocks sboxStart)
          ⟨k, inside⟩ ⟨r, small⟩ j).coefficient ⟨x, h⟩)
      else 0 := by
  unfold blockFailure at pass
  split_ifs at pass with each
  have each := List.all_eq_true.mp each
  have rep := rep_trace (I := blockInterface logical blocks sboxStart inside)
  have length : atomRows.size = 150 := by
    have equal := rep.length_eq
    rw [PoseidonSboxPlan.trace_rows_length] at equal
    simpa using equal.symm
  have checked := List.findSome?_eq_none_iff.mp pass r (List.mem_range.mpr small)
  rw [Option.map_eq_none_iff] at checked
  have rs : r < (atomRows.map (blockAtoms (tableOf (slotsOf p)) blocks[k] sboxStart k).row).size := by
    rw [Array.size_map, length]; exact small
  have atR : r < atomRows.size := by rw [length]; exact small
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem rs, Option.getD_some] at checked
  rw [← sortedRuns_sum]
  refine rowSum_congr checked (fun port c => ?_) j x
  have pair := rep.get (by rw [PoseidonSboxPlan.trace_rows_length]; exact small) (by simpa [length] using small)
  simp only [List.get_eq_getElem, Array.getElem_toList] at pair
  obtain ⟨repInput, repOutput⟩ := pair
  simp only [Array.getElem_map, PoseidonSboxFamilyPlan.rowForms, PoseidonSboxFamilyPlan.rowAt,
    PoseidonRetainedRows.rows, List.get_eq_getElem, List.getElem_map, PoseidonSboxPlan.Row.meaningfulForm]
  refine meaningfulForm_coefficient (fun c => ?_) (fun c => ?_) port c
  · rw [Atoms.row, expand_coefficient, repInput c]
    exact eval_congr _ c (blockAtoms_form logical increase inside each · · c) _
  · rw [Atoms.row, repOutput]
    dsimp only
    by_cases sbox : atomRows[r].output < 150
    · have form := blockAtoms_form logical increase inside each (16 + atomRows[r].output) (by omega) c
      rw [Atoms.form, ite_eq_right (show ¬16 + atomRows[r].output < 16 by omega),
        ite_eq_left (show 16 + atomRows[r].output < 166 by omega),
        show 16 + atomRows[r].output - 16 = atomRows[r].output by omega,
        atomOf, dite_eq_right (show ¬16 + atomRows[r].output < 16 by omega),
        ite_eq_left (show 16 + atomRows[r].output < 166 by omega),
        show 16 + atomRows[r].output - 16 = atomRows[r].output by omega] at form
      exact form
    · have none : (blockAtoms (tableOf (slotsOf p)) blocks[k] sboxStart k).sbox[atomRows[r].output]? = none := by
        simp [sbox]
      rw [none, PoseidonSboxPlan.sboxOutputAt, dite_eq_right (by rw [PoseidonRetainedSlots.rows_length]; exact sbox)]
      rfl

end package

theorem logical_of_check {p : StepPackage.Package} {pushes : List Push} (pass : keyRunsFailure p pushes = none) :
    ProductionAssignment.logicalWidth (slotsOf p) ≤ StepRelation.keyWidth := by
  unfold keyRunsFailure at pass
  split_ifs at pass with logical
  exact logical

/-- The check's fit conditions are the relation's (`CircuitRelation.Fits`). -/
theorem fits_of_check {p : StepPackage.Package} {pushes : List Push} (pass : keyRunsFailure p pushes = none)
    {pieces : List Pieces.Piece} {next : ℕ}
    (rows : p.relationRows = CircuitRelation.relationRows CircuitRelation.publicColumns pieces next p.width)
    (blocks : p.blocks = CircuitRelation.blocksOf pieces) (start : p.sboxStart = CircuitRelation.sboxStart pieces next) :
    CircuitRelation.Fits CircuitRelation.publicColumns pieces next p.width := by
  unfold keyRunsFailure at pass
  rw [ite_eq_left (logical_of_check (by unfold keyRunsFailure; exact pass))] at pass
  dsimp only at pass
  split_ifs at pass with notFits
  all_goals
    have fitsPackage : packageFits p = true := by simpa using notFits
    simp only [packageFits, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at fitsPackage
    obtain ⟨⟨⟨cube, columns⟩, sboxes⟩, inputs⟩ := fitsPackage
    simp only [rows, blocks, start] at columns sboxes inputs cube
    exact ⟨⟨by omega, columns⟩, fun _ => ⟨fun block member lane term inTerms =>
      by simpa using inputs block member lane.val (List.mem_range.mpr lane.isLt) term inTerms, sboxes, cube⟩⟩

/-- `keyRunsFailure` gives the matrices of the circuit key's relation, for the package that the
emitter writes from the circuit's pieces (`StepPackage.relationRows_packageOf`). -/
theorem keyRunsFailure_sound {p : StepPackage.Package} {pushes : List Push} (pass : keyRunsFailure p pushes = none)
    {pieces : List Pieces.Piece} {next : ℕ}
    (fits : CircuitRelation.Fits CircuitRelation.publicColumns pieces next p.width)
    (packed : CircuitRelation.packable CircuitRelation.publicColumns pieces = true)
    (rows : p.relationRows = CircuitRelation.relationRows CircuitRelation.publicColumns pieces next p.width)
    (blocks : p.blocks = CircuitRelation.blocksOf pieces) (start : p.sboxStart = CircuitRelation.sboxStart pieces next)
    (logical : ProductionAssignment.logicalWidth (slotsOf p) ≤ StepRelation.keyWidth) :
    RunsGive pushes (keyRelation pieces next p.width fits (keyLayout p logical)) := by
  unfold keyRunsFailure at pass
  rw [ite_eq_left logical] at pass
  dsimp only at pass
  split_ifs at pass with notFits increase
  have fitsPackage : packageFits p = true := by simpa using notFits
  simp only [packageFits, Bool.and_eq_true, decide_eq_true_eq] at fitsPackage
  obtain ⟨⟨⟨cube, -⟩, -⟩, -⟩ := fitsPackage
  rw [rows, blocks] at cube
  simp only [rows, blocks, start] at pass
  split at pass
  · simp at pass
  · rename_i none_past
    rw [List.findSome?_eq_none_iff] at pass
    have below : ∀ push ∈ pushes, push.row < (CircuitRelation.relationRows CircuitRelation.publicColumns pieces next
        p.width).length + 150 * (CircuitRelation.blocksOf pieces).length := by
      intro push member
      simpa using List.find?_eq_none.mp none_past push member
    have planEq : CircuitRelation.plan CircuitRelation.publicColumns pieces next p.width fits (keyLayout p logical) = _ :=
      dite_eq_left packed
    unfold keyRelation
    rw [planEq]
    refine ⟨fun push member => by have := below push member; omega, fun j i c cubeRow => ?_⟩
    rw [pushedEntry_eq]
    by_cases ordinary : i < (CircuitRelation.relationRows CircuitRelation.publicColumns pieces next p.width).length
    · have member : ((CircuitRelation.relationRows CircuitRelation.publicColumns pieces next p.width)[i], i) ∈
          (CircuitRelation.relationRows CircuitRelation.publicColumns pieces next p.width).zipIdx := by
        simp [List.mem_zipIdx_iff_getElem?, List.getElem?_eq_getElem ordinary]
      have checked := pass _ (List.mem_append_left _ (List.mem_map_of_mem member))
      obtain ⟨bounded, accepted⟩ := ordinary_rows logical increase (Option.map_eq_none_iff.mp checked)
      rw [← runsByRow_getD pushes (show i < (CircuitRelation.relationRows CircuitRelation.publicColumns pieces next
          p.width).length + 150 * (CircuitRelation.blocksOf pieces).length by omega),
        ← sortedRuns_sum, rowFailure_sum accepted ⟨j.val, j.isLt⟩ c, appendSource_left]
      · split_ifs <;> rfl
      · rw [CircuitRelation.ordinary_rowCount]
        exact ordinary
    · by_cases sbox : i < (CircuitRelation.relationRows CircuitRelation.publicColumns pieces next p.width).length +
          150 * (CircuitRelation.blocksOf pieces).length
      · obtain ⟨k, r, small, rfl⟩ : ∃ k r, r < 150 ∧
            i = (CircuitRelation.relationRows CircuitRelation.publicColumns pieces next p.width).length + 150 * k + r :=
          ⟨(i - (CircuitRelation.relationRows CircuitRelation.publicColumns pieces next p.width).length) / 150,
            (i - (CircuitRelation.relationRows CircuitRelation.publicColumns pieces next p.width).length) % 150,
            Nat.mod_lt _ (by decide), by omega⟩
        have inside : k < (CircuitRelation.blocksOf pieces).length := by omega
        have member : ((CircuitRelation.blocksOf pieces)[k], k) ∈ (CircuitRelation.blocksOf pieces).zipIdx := by
          simp [List.mem_zipIdx_iff_getElem?, List.getElem?_eq_getElem inside]
        have checked := pass _ (List.mem_append_right _ (List.mem_map_of_mem member))
        have sum := block_rows logical increase inside (Option.map_eq_none_iff.mp checked) small ⟨j.val, j.isLt⟩ c
        rw [← runsByRow_getD pushes sbox, sum,
          show (CircuitRelation.relationRows CircuitRelation.publicColumns pieces next p.width).length + 150 * k + r =
              (LogicalPlan.plan (CircuitRelation.relationRows CircuitRelation.publicColumns pieces next p.width) p.width
                fits.rows (keyLayout p logical)).rowCount + (150 * k + r) by
            rw [CircuitRelation.ordinary_rowCount]; omega,
          appendSource_right]
        · split_ifs with h
          · exact congrArg (fun form => gl (SparseForm.coefficient form ⟨c.val, h⟩))
              (indexed_forms_at _ _ inside small _ ⟨j.val, j.isLt⟩).symm
          · rfl
        · rw [PoseidonSboxFamilyPlan.plan_rowCount]
          have := Nat.mul_le_mul_right 150 (show k + 1 ≤ (CircuitRelation.blocksOf pieces).length by omega)
          omega
      · have empty : pushes.filter (·.row = i) = [] := List.filter_eq_nil_iff.mpr fun push member => by
          have := below push member
          rw [decide_eq_true_eq]
          omega
        rw [empty, planSource_padding (cube := cubeRow)]
        · rfl
        · rw [Plan.append_rowCount, CircuitRelation.ordinary_rowCount, PoseidonSboxFamilyPlan.plan_rowCount]
          omega

end CompressionLean.Checks.OptionB
