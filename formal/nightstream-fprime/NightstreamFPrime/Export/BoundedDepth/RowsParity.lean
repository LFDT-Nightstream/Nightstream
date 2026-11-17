import NightstreamFPrime.Export.Generic.PackageRows
import NightstreamFPrime.Export.BoundedDepth.StepPackage

/-! A checksum of the dense rows of a step package (`denseSum`), and a fast computation of it
(`fastSum`, `fastSum_eq`).

`denseSum` gives cell `x` the weight `cellWeight x` and port `4 · row + matrix` the weight
`portWeight`. The dense rows `PackageRows.rows` hold their entries over the source columns, so
their checksum reads each source column with `columnWeight`: the cell weights of its span, cell
`j` times `3^j` (`denseSum_rows`). `fastSum` computes the checksum of the model of the final
verifier's rows (`Export.BoundedDepth.FinalRows.packageRows`) without the S-box forms. It reads
the column weights from one table (`weightTable`, `weightAt_eq`) and evaluates each R1CS row on
them (`planSum_ordinary`). It evaluates each block's input lanes and S-box columns, and it runs a
copy of `PoseidonSboxPlan.compile` on values (`valueCompile`, `valueCompile_eval`). The S-box
forms do not merge terms and grow about sixteenfold in each partial round. A Rust test computes
the same checksum from Rust's factored rows. -/

namespace NightstreamFPrime.Export.BoundedDepth.RowsParity

open Finset
open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Spec.RingRelation.PiDECAlgebra.Radix (fieldOfNat)
open NightstreamFPrime.Export
open NightstreamFPrime.Export.Package (invocationInputCombination)
open NightstreamFPrime.Export.Generic
open NightstreamFPrime.Export.Generic.LogicalPlan (publicCount)
open scoped NightstreamFPrime.Spec.Compression.Production

/-- A field element that depends on a unit and a column (Rust `mix`). -/
def mix (unit column : Nat) : F :=
  let point : F := Spec.Poseidon2.ofNat (unit * 16777259 + column + 1)
  point * point * point + Spec.Poseidon2.ofNat unit

/-- The weight of logical column `x` (Rust `mix(2, x)`). -/
def cellWeight (x : Nat) : F := mix 2 x

/-- The weight of port `unit = 4 · row + matrix` (Rust `mix(unit, 1)`). -/
def portWeight (unit : Nat) : F := mix unit 1

/-- The checksum of dense matrices: `Σ_row Σ_matrix portWeight (4 · row + matrix) ·
Σ_x cellWeight x · M_matrix[row, x]`. -/
def denseSum (M : Compression.Verifier.Matrices F) : F :=
  ∑ row ∈ range M.rows, ∑ matrix ∈ range 4, portWeight (4 * row + matrix) *
    ∑ x ∈ range M.columns, cellWeight x * M.entry matrix row x

variable {width : Nat}

/-- The weight of source column `column` (Rust `column_weight`): cell `j` of its span has the
weight `3^j` (`PackageRows.form_coefficient`). -/
def columnWeight (slots : List (LowNormAssignment.Slot width)) (column : Nat) : F :=
  ∑ j ∈ range (PackageRows.spanWidth slots column),
    cellWeight (PackageRows.spanStart slots column + j) * fieldOfNat 3 ^ j

/-! ### The checksum of a plan -/

/-- Port `matrix` of row `row` of a plan at an assignment, and 0 past the ports. -/
def portValue (plan : Plan width) (assignment : Assignment F width) (row : Fin plan.rowCount)
    (matrix : Nat) : F :=
  if inside : matrix < Spec.ProductionRelation.matrixCount then
    (plan.portForm row ⟨matrix, inside⟩).eval assignment
  else 0

/-- The weighted ports of the plan row with index `unit`. -/
def rowSum (unit : Nat) (value : Nat → F) : F :=
  ∑ matrix ∈ range 4, portWeight (4 * unit + matrix) * value matrix

/-- The weighted ports of a plan at an assignment, with its rows from index `base`. -/
def planSum (plan : Plan width) (assignment : Assignment F width) (base : Nat) : F :=
  ∑ row : Fin plan.rowCount, rowSum (base + row.val) (portValue plan assignment row)

theorem eval_entries {columns : Nat} (form : SparseForm columns)
    (assignment : Assignment F columns) :
    form.eval assignment =
      (form.entries.map fun entry => entry.coefficient * assignment entry.column).sum := by
  obtain ⟨entries⟩ := form
  induction entries with
  | nil => exact SparseForm.empty_eval assignment
  | cons entry rest hypothesis =>
      rw [show (⟨entry :: rest⟩ : SparseForm columns).eval assignment =
          (SparseForm.add (SparseForm.singleton entry.column entry.coefficient) ⟨rest⟩).eval
            assignment from rfl, SparseForm.add_eval, SparseForm.singleton_eval, hypothesis]
      simp only [List.map_cons, List.sum_cons]

/-- The cells of port `matrix` of a dense row, against the cell weights, are the port's form at
the column weights. -/
private theorem cells_eq (slots : List (LowNormAssignment.Slot width)) (plan : Plan width)
    {columns : Nat} (wide : ProductionAssignment.logicalWidth slots ≤ columns)
    (row : Fin plan.rowCount) (matrix : Nat) (small : matrix < 4) :
    ∑ x ∈ range columns, cellWeight x *
        (PackageRows.rows slots plan columns).matrices.entry matrix row x =
      portValue plan (fun column => columnWeight slots column.val) row matrix := by
  let R := PackageRows.rows slots plan columns
  let Φ : ℕ → F := fun x => if x < columns then cellWeight x else 0
  have inside : matrix < Spec.ProductionRelation.matrixCount ∧ row.val < plan.rowCount :=
    ⟨by rw [Spec.ProductionRelation.matrixCount_eq]; exact small, row.isLt⟩
  have entries : R.entries matrix row =
      (plan.portForm row ⟨matrix, inside.1⟩).entries.map fun entry =>
        (entry.column.val, entry.coefficient) :=
    dite_eq_left inside
  have cells : ∑ x ∈ range columns, cellWeight x * R.matrices.entry matrix row x =
      ∑ x ∈ range columns, algebraMap F F (R.matrices.entry matrix row x) * Φ x :=
    sum_congr rfl fun x member => by
      simp only [Φ, mem_range.mp member, ite_true, Algebra.algebraMap_self_apply, mul_comm]
  change ∑ x ∈ range columns, cellWeight x * R.matrices.entry matrix row x = _
  rw [cells, Compression.Verifier.sum_entry, entries, List.map_map, portValue,
    dite_eq_left inside.1, eval_entries]
  refine congrArg List.sum (List.map_congr_left fun entry _ => ?_)
  change ∑ x ∈ range columns, algebraMap F F (R.run (entry.column.val, entry.coefficient) x) *
      Φ x = entry.coefficient * columnWeight slots entry.column.val
  rw [Compression.Verifier.sum_run R _ columns Φ (fun x large => by
    simp only [Φ, ite_eq_right (show ¬ x < columns by omega)])]
  change algebraMap F F entry.coefficient *
      ∑ k ∈ range (PackageRows.spanWidth slots entry.column.val),
      algebraMap F F (fieldOfNat 3 ^ k) * Φ (PackageRows.spanStart slots entry.column.val + k) =
    entry.coefficient * columnWeight slots entry.column.val
  rw [Algebra.algebraMap_self_apply]
  congr 1
  refine sum_congr rfl fun k member => ?_
  have := PackageRows.span_le slots entry.column.val
  have below : PackageRows.spanStart slots entry.column.val + k < columns := by
    have := mem_range.mp member
    omega
  simp only [Φ, below, ite_true, Algebra.algebraMap_self_apply, mul_comm]

/-- The checksum of the dense rows of a plan is the plan's checksum at the column weights. -/
theorem denseSum_rows (slots : List (LowNormAssignment.Slot width)) (plan : Plan width)
    {columns : Nat} (wide : ProductionAssignment.logicalWidth slots ≤ columns) :
    denseSum (PackageRows.rows slots plan columns).matrices =
      planSum plan (fun column => columnWeight slots column.val) 0 := by
  unfold denseSum planSum
  change ∑ row ∈ range plan.rowCount, _ = _
  rw [Finset.sum_range]
  refine sum_congr rfl fun row _ => ?_
  rw [zero_add]
  exact sum_congr rfl fun matrix member =>
    congrArg _ (cells_eq slots plan wide row matrix (mem_range.mp member))

private theorem portValue_forms (plan : Plan width) (assignment : Assignment F width)
    (row : Fin plan.rowCount) (matrix : Nat) (small : matrix < 4) :
    portValue plan assignment row matrix = (plan.forms row
      ⟨matrix, by rw [Spec.ProductionRelation.meaningfulPortCount_eq]; exact small⟩).eval
        assignment := by
  rw [portValue, dite_eq_left (by rw [Spec.ProductionRelation.matrixCount_eq]; exact small)]
  unfold Plan.portForm meaningfulPort?
  rw [dite_eq_left (by rw [Spec.ProductionRelation.meaningfulPortCount_eq]; exact small)]

private theorem rowSum_expand (unit : Nat) (value : Nat → F) :
    rowSum unit value = portWeight (4 * unit) * value 0 + portWeight (4 * unit + 1) * value 1 +
      portWeight (4 * unit + 2) * value 2 + portWeight (4 * unit + 3) * value 3 := by
  simp only [rowSum, sum_range_succ, sum_range_zero, zero_add, add_zero]

theorem planSum_append (left right : Plan width)
    (bound : left.rowCount + right.rowCount ≤ 2 ^ Lifecycle.cubeVariables)
    (assignment : Assignment F width) (base : Nat) :
    planSum (Plan.append left right bound) assignment base =
      planSum left assignment base + planSum right assignment (base + left.rowCount) := by
  unfold planSum
  change ∑ row : Fin (left.rowCount + right.rowCount), rowSum (base + row.val)
    (portValue (Plan.append left right bound) assignment row) = _
  rw [Fin.sum_univ_add]
  refine congrArg₂ (· + ·) ?_ ?_
  · refine sum_congr rfl fun row _ => ?_
    have form : portValue (Plan.append left right bound) assignment
        (Fin.castAdd right.rowCount row) = portValue left assignment row := by
      funext matrix
      unfold portValue
      split
      · rw [show Fin.castAdd right.rowCount row = Plan.leftIndex _ _ row from rfl,
          Plan.append_portForm_left]
      · rfl
    rw [form, Fin.val_castAdd]
  · refine sum_congr rfl fun row _ => ?_
    have form : portValue (Plan.append left right bound) assignment
        (Fin.natAdd left.rowCount row) = portValue right assignment row := by
      funext matrix
      unfold portValue
      split
      · rw [show Fin.natAdd left.rowCount row = Plan.rightIndex _ _ row from rfl,
          Plan.append_portForm_right]
      · rfl
    rw [form, Fin.val_natAdd, Nat.add_assoc]

/-- `Σ` over a list with its indices from `k`, as a sum over the positions of the list. -/
theorem sum_zipIdx {α : Type} (f : Nat → α → F) :
    ∀ (l : List α) (k : Nat),
      ((l.zipIdx k).map fun entry => f entry.2 entry.1).sum =
        ∑ i : Fin l.length, f (k + i.val) l[i.val]
  | [], _ => by simp
  | a :: l, k => by
      rw [List.zipIdx_cons, List.map_cons, List.sum_cons, sum_zipIdx f l (k + 1)]
      change _ = ∑ i : Fin (l.length + 1), _
      rw [Fin.sum_univ_succ]
      simp only [Fin.val_zero, add_zero, Fin.val_succ, List.getElem_cons_zero,
        List.getElem_cons_succ]
      congr 1
      exact sum_congr rfl fun i _ => by rw [show k + 1 + i.val = k + (i.val + 1) by omega]

/-! ### The R1CS rows -/

/-- A linear combination at the column weights; its constant reads column 0. -/
def combinationValue (weight : Nat → F) (combination : R1CS.LinearCombination) : F :=
  combination.constant * weight 0 + (combination.terms.map fun term => term.2 * weight term.1).sum

/-- The weighted ports `a`, `b` and `c` of an R1CS row at plan row `unit`. -/
def ordinaryValue (weight : Nat → F) (unit : Nat) (row : R1CS.Row) : F :=
  portWeight (4 * unit) * combinationValue weight row.a +
    portWeight (4 * unit + 1) * combinationValue weight row.b +
    portWeight (4 * unit + 2) * combinationValue weight row.c

private theorem identity_form_eval (inside : publicCount ≤ width) {weight : Nat → F}
    (zero : ∀ column, width ≤ column → weight column = 0) (column : Nat) :
    ((PlanPush.identity width inside).form column).eval (fun c => weight c.val) =
      weight column := by
  unfold LogicalPlan.Layout.form
  by_cases bounded : column < width
  · rw [dite_eq_left bounded]
    change (SparseForm.singleton ⟨column, bounded⟩ 1).eval _ = _
    rw [SparseForm.singleton_eval, one_mul]
  · rw [dite_eq_right bounded, SparseForm.empty_eval, zero column (by omega)]

private theorem identity_combination_eval (inside : publicCount ≤ width) {weight : Nat → F}
    (zero : ∀ column, width ≤ column → weight column = 0)
    (combination : R1CS.LinearCombination) :
    ((PlanPush.identity width inside).combination combination).eval (fun c => weight c.val) =
      combinationValue weight combination := by
  unfold LogicalPlan.Layout.combination combinationValue
  rw [SparseForm.add_eval, SparseForm.singleton_eval, LogicalPlan.Layout.terms_eval]
  congr 1
  refine congrArg List.sum (List.map_congr_left fun term _ => ?_)
  rw [← LogicalPlan.Layout.form_eval, identity_form_eval inside zero]

theorem planSum_ordinary (rows : List R1CS.Row) (fits : LogicalPlan.Fits rows width)
    (inside : publicCount ≤ width) {weight : Nat → F}
    (zero : ∀ column, width ≤ column → weight column = 0) (base : Nat) :
    planSum (LogicalPlan.plan rows width fits (PlanPush.identity width inside))
        (fun c => weight c.val) base =
      ∑ row : Fin rows.length, ordinaryValue weight (base + row.val) rows[row.val] := by
  unfold planSum
  refine sum_congr rfl fun row _ => ?_
  rw [rowSum_expand]
  simp only [portValue_forms _ _ _ _ (show 0 < 4 by decide),
    portValue_forms _ _ _ _ (show 1 < 4 by decide), portValue_forms _ _ _ _ (show 2 < 4 by decide),
    portValue_forms _ _ _ _ (show 3 < 4 by decide)]
  rw [PlanPush.ordinary_forms _ rows fits row.val row.isLt,
    PlanPush.ordinary_forms _ rows fits row.val row.isLt,
    PlanPush.ordinary_forms _ rows fits row.val row.isLt,
    PlanPush.ordinary_forms _ rows fits row.val row.isLt]
  simp only [OrdinaryRow.Forms.meaningfulForm, identity_combination_eval inside zero,
    SparseForm.empty_eval, mul_zero, add_zero, ordinaryValue]
  rfl

/-! ### The S-box rows on values -/

/-- One full round on values: the S-box rows read the state plus the round constants, and the
next state is the external layer of the S-box outputs (`PoseidonSboxPlan.compileStep`). -/
def fullStep (constants : List (List Nat)) (round : Nat) (one : F) (sbox : Nat → F)
    (next : Nat) (state : Vector F 16) : Nat × Vector F 16 × List (F × F) :=
  let outputs := Vector.ofFn fun lane : Fin 16 => sbox (next + lane.val)
  (next + 16, Vector.ofFn (Layer.externalF fun lane => outputs[lane.val]),
    List.ofFn fun lane : Fin 16 =>
      (state[lane.val] + Spec.Poseidon2.constantAt constants round lane.val * one,
        outputs[lane.val]))

/-- `PoseidonSboxPlan.compileStep` on values: the next S-box, the state, and the input and the
output of each S-box row. `one` is the value of the constant column and `sbox index` the value
of S-box output `index`. -/
def valueStep (one : F) (sbox : Nat → F) (next : Nat) (state : Vector F 16) :
    Permutation.Step → Nat × Vector F 16 × List (F × F)
  | .initialLayer => (next, Vector.ofFn (Layer.externalF fun lane => state[lane.val]), [])
  | .initialFullRound round =>
      fullStep Spec.Poseidon2.initialConstants round one sbox next state
  | .partialRound round =>
      (next + 1, Vector.ofFn (Layer.internalF fun lane =>
          if lane.val = 0 then sbox next else state[lane.val]),
        [(state[0] + Spec.Poseidon2.ofNat (Spec.Poseidon2.internalConstants.getD round 0) * one,
          sbox next)])
  | .terminalFullRound round =>
      fullStep Spec.Poseidon2.terminalConstants round one sbox next state

/-- `PoseidonSboxPlan.compile` on values. -/
def valueCompile (one : F) (sbox : Nat → F) :
    Nat → Vector F 16 → List Permutation.Step → Nat × Vector F 16 × List (F × F)
  | next, state, [] => (next, state, [])
  | next, state, step :: rest =>
      let head := valueStep one sbox next state step
      let tail := valueCompile one sbox head.1 head.2.1 rest
      (tail.1, tail.2.1, head.2.2 ++ tail.2.2)

/-- The input and the output of an S-box row at an assignment. -/
def evalPair (assignment : Assignment F width) (row : SboxRow.Forms width) : F × F :=
  (row.input.eval assignment, row.output.eval assignment)

private theorem addConstant_eval (assignment : Assignment F width) (column : Fin width)
    (form : SparseForm width) (value : F) :
    (SparseLayer.addConstant column form value).eval assignment =
      form.eval assignment + value * assignment column := by
  simp only [SparseLayer.addConstant, SparseLayer.add, SparseLayer.constant, SparseForm.add_eval,
    SparseForm.singleton_eval]

section Values

variable {interface : PoseidonSboxPlan.Interface width} {assignment : Assignment F width}
  {sbox : Nat → F}

/-- The value step and the step on forms agree: the same next S-box, a state with the values of
the forms, and rows with the values of the forms. -/
private def StepAgrees (assignment : Assignment F width)
    (value : Nat × Vector F 16 × List (F × F)) (result : PoseidonSboxPlan.StepResult width) :
    Prop :=
  value.1 = result.nextSbox ∧
    (∀ lane : Fin 16, value.2.1[lane.val] = (result.state lane).eval assignment) ∧
    value.2.2 = result.rows.map (evalPair assignment)

private theorem fullStep_agrees
    (sboxEq : ∀ index, (PoseidonSboxPlan.sboxOutputAt interface index).eval assignment =
      sbox index)
    (constants : List (List Nat)) (round next : Nat) {state : SparseLayer.State width}
    {values : Vector F 16}
    (same : ∀ lane : Fin 16, values[lane.val] = (state lane).eval assignment) :
    StepAgrees assignment
      (fullStep constants round (assignment interface.oneColumn) sbox next values)
      ⟨next + 16, SparseLayer.external (PoseidonSboxPlan.fullOutput interface next),
        PoseidonSboxPlan.fullRows interface constants round next state⟩ := by
  refine ⟨rfl, fun lane => ?_, ?_⟩
  · simp only [fullStep, Vector.getElem_ofFn, SparseLayer.eval_external]
    congr 1
    funext index
    simp only [SparseLayer.evalState, PoseidonSboxPlan.fullOutput, sboxEq]
  · simp only [fullStep, PoseidonSboxPlan.fullRows, List.map_ofFn]
    refine congrArg List.ofFn (funext fun lane => ?_)
    simp only [Function.comp_apply, evalPair, PoseidonSboxPlan.fullInput, addConstant_eval,
      PoseidonSboxPlan.fullOutput, sboxEq, Vector.getElem_ofFn, same]

private theorem valueStep_agrees
    (sboxEq : ∀ index, (PoseidonSboxPlan.sboxOutputAt interface index).eval assignment =
      sbox index)
    (next : Nat) {state : SparseLayer.State width} {values : Vector F 16}
    (same : ∀ lane : Fin 16, values[lane.val] = (state lane).eval assignment)
    (step : Permutation.Step) :
    StepAgrees assignment (valueStep (assignment interface.oneColumn) sbox next values step)
      (PoseidonSboxPlan.compileStep interface next state step) := by
  cases step with
  | initialLayer =>
      refine ⟨rfl, fun lane => ?_, rfl⟩
      simp only [valueStep, PoseidonSboxPlan.compileStep, Vector.getElem_ofFn,
        SparseLayer.eval_external]
      congr 1
      funext index
      exact same index
  | initialFullRound round => exact fullStep_agrees sboxEq _ round next same
  | partialRound round =>
      refine ⟨rfl, fun lane => ?_, ?_⟩
      · simp only [valueStep, PoseidonSboxPlan.compileStep, Vector.getElem_ofFn,
          SparseLayer.eval_internal]
        congr 1
        funext index
        by_cases zero : index.val = 0
        · simp only [SparseLayer.evalState, PoseidonSboxPlan.partialState, zero, ite_true,
            PoseidonSboxPlan.partialOutput, sboxEq]
        · simp only [SparseLayer.evalState, PoseidonSboxPlan.partialState, zero, ite_false, same]
      · simp only [valueStep, PoseidonSboxPlan.compileStep, PoseidonSboxPlan.partialRows,
          List.map_cons, List.map_nil, evalPair, PoseidonSboxPlan.partialInput, addConstant_eval,
          PoseidonSboxPlan.partialOutput, sboxEq]
        have first := same 0
        simp only [Fin.val_zero] at first
        rw [first]
  | terminalFullRound round => exact fullStep_agrees sboxEq _ round next same

/-- `valueCompile` gives the values of the rows of `PoseidonSboxPlan.compile` at an
assignment. -/
theorem valueCompile_eval {one : F} (oneEq : assignment interface.oneColumn = one)
    (sboxEq : ∀ index, (PoseidonSboxPlan.sboxOutputAt interface index).eval assignment =
      sbox index) :
    ∀ (steps : List Permutation.Step) (next : Nat) {state : SparseLayer.State width}
      {values : Vector F 16},
      (∀ lane : Fin 16, values[lane.val] = (state lane).eval assignment) →
        (valueCompile one sbox next values steps).2.2 =
          (PoseidonSboxPlan.compile interface next state steps).rows.map (evalPair assignment) := by
  subst oneEq
  exact fun steps next _ _ same => (agrees steps next next rfl same).2.2
where
  agrees : ∀ (steps : List Permutation.Step) (next next' : Nat)
      {state : SparseLayer.State width} {values : Vector F 16}, next = next' →
      (∀ lane : Fin 16, values[lane.val] = (state lane).eval assignment) →
        StepAgrees assignment (valueCompile (assignment interface.oneColumn) sbox next values steps)
          (PoseidonSboxPlan.compile interface next' state steps)
    | [], _, _, _, _, rfl, same => ⟨rfl, same, rfl⟩
    | step :: rest, next, _, _, _, rfl, same => by
        obtain ⟨nextEq, states, rows⟩ := valueStep_agrees sboxEq next same step
        obtain ⟨tailNext, tailStates, tailRows⟩ := agrees rest _ _ nextEq states
        refine ⟨tailNext, tailStates, ?_⟩
        change _ ++ _ = List.map _ (_ ++ _)
        rw [List.map_append, rows, tailRows]
end Values

/-! ### The S-box family -/

/-- The weighted ports of an S-box row at plan row `unit`: port `c` reads the output and port
`sboxInput` reads the input. -/
def pairValue (unit : Nat) (pair : F × F) : F :=
  portWeight (4 * unit + 2) * pair.2 + portWeight (4 * unit + 3) * pair.1

/-- The weighted ports of a list of S-box rows, with their indices from `base`. -/
def pairsSum (base : Nat) (pairs : List (F × F)) : F :=
  ((pairs.zipIdx base).map fun entry => pairValue entry.2 entry.1).sum

theorem planSum_family {count : Nat} (interface : PoseidonSboxFamilyPlan.Interface width count)
    (bound : count * 150 ≤ 2 ^ Lifecycle.cubeVariables) (assignment : Assignment F width)
    (base : Nat) :
    planSum (PoseidonSboxFamilyPlan.plan interface bound) assignment base =
      ∑ invocation : Fin count, pairsSum (base + 150 * invocation.val)
        ((PoseidonSboxPlan.trace (PoseidonSboxFamilyPlan.invocationInterface interface
          invocation)).rows.map (evalPair assignment)) := by
  let e : Fin count × Fin 150 ≃ Fin (count * 150) :=
    ⟨Fin.encodeProd, Fin.decodeProd, Fin.decodeProd_encodeProd, Fin.encodeProd_decodeProd⟩
  unfold planSum
  change ∑ row : Fin (count * 150), _ = _
  refine (Fintype.sum_equiv e (fun pair => rowSum (base + (e pair).val)
      (portValue (PoseidonSboxFamilyPlan.plan interface bound) assignment (e pair)))
    (fun row => rowSum (base + row.val)
      (portValue (PoseidonSboxFamilyPlan.plan interface bound) assignment row))
    (fun _ => rfl)).symm.trans ?_
  rw [Fintype.sum_prod_type]
  refine sum_congr rfl fun invocation _ => ?_
  let rows := (PoseidonSboxPlan.trace
    (PoseidonSboxFamilyPlan.invocationInterface interface invocation)).rows
  have length : (rows.map (evalPair assignment)).length = 150 := by
    simp only [List.length_map, rows, PoseidonSboxPlan.trace_rows_length]
  unfold pairsSum
  rw [sum_zipIdx, ← Fin.sum_congr' _ length]
  refine sum_congr rfl fun row _ => ?_
  have value : ∀ matrix (small : matrix < 4),
      portValue (PoseidonSboxFamilyPlan.plan interface bound) assignment
          (e (invocation, Fin.cast length row)) matrix =
        ((rows[row.val]'(by simpa using row.isLt)).meaningfulForm
          ⟨matrix, by rw [Spec.ProductionRelation.meaningfulPortCount_eq]; exact small⟩).eval
          assignment := by
    intro matrix small
    refine (portValue_forms _ _ _ _ small).trans ?_
    change ((Plan.indexed (PoseidonSboxFamilyPlan.rowForms interface) bound).forms
      (Plan.indexedRow count 150 invocation (Fin.cast length row)) _).eval _ = _
    rw [Plan.indexed_forms]
    simp only [PoseidonSboxFamilyPlan.rowForms, PoseidonSboxFamilyPlan.rowAt,
      PoseidonRetainedRows.rows, List.get_eq_getElem, List.getElem_map, Fin.val_cast,
      PoseidonSboxPlan.Row.meaningfulForm, rows]
  have unit : (e (invocation, Fin.cast length row)).val = 150 * invocation.val + row.val := by
    simp only [e, Equiv.coe_fn_mk, Fin.encodeProd, Fin.coe_mkDivMod, Fin.val_cast]
  show rowSum (base + (e (invocation, Fin.cast length row)).val)
    (portValue (PoseidonSboxFamilyPlan.plan interface bound) assignment
      (e (invocation, Fin.cast length row))) = _
  rw [unit, rowSum_expand, value 0 (by decide), value 1 (by decide), value 2 (by decide),
    value 3 (by decide)]
  simp only [SboxRow.Forms.meaningfulForm, SparseForm.empty_eval, mul_zero, zero_add,
    pairValue, List.getElem_map, evalPair, Nat.add_assoc]
  rfl

/-! ### The column weights -/

/-- `Σ_{j < count} cellWeight (start + j) · 3^j`, by Horner's rule (Rust `column_weight`). -/
def recompose (start : Nat) : Nat → F
  | 0 => 0
  | count + 1 => cellWeight start + fieldOfNat 3 * recompose (start + 1) count

theorem recompose_eq (start count : Nat) :
    recompose start count = ∑ j ∈ range count, cellWeight (start + j) * fieldOfNat 3 ^ j := by
  induction count generalizing start with
  | zero => simp [recompose]
  | succ count hypothesis =>
      rw [recompose, hypothesis, sum_range_succ', mul_sum, add_zero, pow_zero, mul_one, add_comm]
      congr 1
      refine sum_congr rfl fun j _ => ?_
      rw [show start + (j + 1) = start + 1 + j by omega, pow_succ]
      ring

/-- Each slot as its source column, its first logical column and its width, with the first
slot at logical column `offset`. -/
def placed : List (LowNormAssignment.Slot width) → Nat → List (Nat × Nat × Nat)
  | [], _ => []
  | slot :: rest, offset =>
      (slot.source.val, offset, slot.width) :: placed rest (offset + slot.width)

private def placeStep (state : List (Nat × Nat × Nat) × Nat)
    (slot : LowNormAssignment.Slot width) : List (Nat × Nat × Nat) × Nat :=
  ((slot.source.val, state.2, slot.width) :: state.1, state.2 + slot.width)

/-- One more than the largest source column of a slot. -/
def tableSize (slots : List (LowNormAssignment.Slot width)) : Nat :=
  slots.foldl (fun size slot => max size (slot.source.val + 1)) 0

private def write (table : Array F) (entry : Nat × Nat × Nat) : Array F :=
  table.setIfInBounds entry.1 (recompose entry.2.1 entry.2.2)

/-- The weights of the slotted source columns. The slots are written in reverse order, so the
first slot of a column gives its weight, as `SlotLayout.slotIndex` does. -/
def weightTable (slots : List (LowNormAssignment.Slot width)) : Array F :=
  (slots.foldl placeStep ([], ProductionAssignment.publicWidth)).1.foldl write
    (Array.replicate (tableSize slots) 0)

/-- `columnWeight`, with the weights of the slotted columns read from a table. -/
def weightAt (table : Array F) (column : Nat) : F :=
  if column < publicCount then cellWeight column else table.getD column 0

private theorem foldl_placeStep : ∀ (slots : List (LowNormAssignment.Slot width))
    (done : List (Nat × Nat × Nat)) (offset : Nat),
    (slots.foldl placeStep (done, offset)).1 = (placed slots offset).reverse ++ done
  | [], _, _ => rfl
  | slot :: rest, done, offset => by
      rw [List.foldl_cons, show placeStep (done, offset) slot =
        ((slot.source.val, offset, slot.width) :: done, offset + slot.width) from rfl,
        foldl_placeStep rest, placed, List.reverse_cons, List.append_assoc]
      rfl

private theorem foldl_max_le : ∀ (slots : List (LowNormAssignment.Slot width)) (size : Nat),
    size ≤ slots.foldl (fun size slot => max size (slot.source.val + 1)) size ∧
      ∀ slot ∈ slots,
        slot.source.val < slots.foldl (fun size slot => max size (slot.source.val + 1)) size
  | [], _ => ⟨le_refl _, fun _ member => by simp at member⟩
  | head :: rest, size => by
      obtain ⟨grows, covers⟩ := foldl_max_le rest (max size (head.source.val + 1))
      refine ⟨by simp only [List.foldl_cons]; omega, fun slot member => ?_⟩
      simp only [List.foldl_cons]
      rcases List.mem_cons.mp member with rfl | inRest
      · omega
      · exact covers slot inRest

/-- The first logical column and the width of the first slot of `column`, with the first slot
at logical column `offset`. -/
private def firstSpan : List (LowNormAssignment.Slot width) → Nat → Nat → Option (Nat × Nat)
  | [], _, _ => none
  | slot :: rest, offset, column =>
      if slot.source.val = column then some (offset, slot.width)
      else firstSpan rest (offset + slot.width) column

private theorem foldr_size (start : Array F) :
    ∀ entries : List (Nat × Nat × Nat),
      (entries.foldr (fun entry table => write table entry) start).size = start.size
  | [] => rfl
  | entry :: rest => by
      rw [List.foldr_cons, write, Array.size_setIfInBounds, foldr_size start rest]

private theorem foldr_read (start : Array F) (column : Nat) :
    ∀ (slots : List (LowNormAssignment.Slot width)) (offset : Nat),
      (∀ slot ∈ slots, slot.source.val < start.size) →
        ((placed slots offset).foldr (fun entry table => write table entry) start).getD column 0 =
          match firstSpan slots offset column with
          | some (first, count) => recompose first count
          | none => start.getD column 0
  | [], _, _ => rfl
  | slot :: rest, offset, bounded => by
      have restBounded := fun other member => bounded other (List.mem_cons_of_mem _ member)
      have inside : slot.source.val <
          ((placed rest (offset + slot.width)).foldr (fun entry table => write table entry)
            start).size := by
        rw [foldr_size]
        exact bounded slot List.mem_cons_self
      rw [placed, List.foldr_cons, write, Array.getD_eq_getD_getElem?,
        Array.getElem?_setIfInBounds, firstSpan]
      by_cases same : slot.source.val = column
      · simp only [same, ite_true] at inside ⊢
        simp only [inside, ite_true, Option.getD_some]
      · simp only [same, ite_false]
        rw [← Array.getD_eq_getD_getElem?]
        exact foldr_read start column rest _ restBounded

private theorem firstSpan_outside {column : Nat} (outside : width ≤ column) :
    ∀ (slots : List (LowNormAssignment.Slot width)) (offset : Nat),
      firstSpan slots offset column = none
  | [], _ => rfl
  | slot :: rest, offset => by
      have := slot.source.isLt
      rw [firstSpan, ite_eq_right (by omega), firstSpan_outside outside rest]

private theorem slotIndex_cons (slot : LowNormAssignment.Slot width)
    (rest : List (LowNormAssignment.Slot width)) (column : Fin width) :
    SlotLayout.slotIndex (slot :: rest) column =
      if slot.source = column then some ⟨0, Nat.succ_pos _⟩
      else (SlotLayout.slotIndex rest column).map Fin.succ := by
  unfold SlotLayout.slotIndex
  change (List.finRange (rest.length + 1)).find? _ = _
  rw [List.finRange_succ, List.find?_cons, List.find?_map]
  by_cases same : slot.source = column
  · simp [same]
  · simp only [List.get_eq_getElem, Fin.val_zero, List.getElem_cons_zero, same, decide_false,
      ite_false, Function.comp_def, Fin.val_succ, List.getElem_cons_succ]

private theorem prefixWidth_succ (slot : LowNormAssignment.Slot width)
    (rest : List (LowNormAssignment.Slot width)) (index : Fin rest.length) :
    LowNormAssignment.prefixWidth (slot :: rest) index.succ =
      slot.width + LowNormAssignment.prefixWidth rest index := by
  simp only [LowNormAssignment.prefixWidth, LowNormAssignment.logicalWidth, Fin.val_succ,
    List.take_succ_cons, List.map_cons, List.sum_cons]

private theorem firstSpan_eq (column : Fin width) :
    ∀ (slots : List (LowNormAssignment.Slot width)) (offset : Nat),
      firstSpan slots offset column.val = (SlotLayout.slotIndex slots column).map fun slot =>
        (offset + LowNormAssignment.prefixWidth slots slot, (slots.get slot).width)
  | [], _ => rfl
  | slot :: rest, offset => by
      rw [firstSpan, slotIndex_cons]
      by_cases same : slot.source = column
      · rw [ite_eq_left (congrArg Fin.val same), ite_eq_left same]
        simp only [Option.map_some, LowNormAssignment.prefixWidth, List.take_zero,
          LowNormAssignment.logicalWidth, List.map_nil, List.sum_nil, add_zero, List.get_eq_getElem,
          List.getElem_cons_zero]
      · rw [ite_eq_right (fun equal => same (Fin.ext equal)), ite_eq_right same,
          firstSpan_eq column rest,
          Option.map_map]
        congr 1
        funext index
        simp only [Function.comp_apply, prefixWidth_succ, List.get_eq_getElem, Fin.val_succ,
          List.getElem_cons_succ, Nat.add_assoc]

/-- The table gives each column its weight. -/
theorem weightAt_eq (slots : List (LowNormAssignment.Slot width)) (inside : publicCount ≤ width)
    (column : Nat) : weightAt (weightTable slots) column = columnWeight slots column := by
  have publicEq : publicCount = 320 := rfl
  unfold weightAt columnWeight PackageRows.spanWidth PackageRows.spanStart PackageRows.span
  by_cases isPublic : column < publicCount
  · rw [ite_eq_left isPublic, dite_eq_left (by omega), ite_eq_left isPublic]
    simp only [sum_range_one, add_zero, pow_zero, mul_one]
  · rw [ite_eq_right isPublic]
    have bounded : ∀ slot ∈ slots,
        slot.source.val < (Array.replicate (tableSize slots) (0 : F)).size := by
      rw [Array.size_replicate]
      exact (foldl_max_le slots 0).2
    have read := foldr_read (Array.replicate (tableSize slots) (0 : F)) column slots
      ProductionAssignment.publicWidth bounded
    have zero : (Array.replicate (tableSize slots) (0 : F)).getD column 0 = 0 := by
      simp only [Array.getD_eq_getD_getElem?, Array.getElem?_replicate]
      split <;> rfl
    rw [zero] at read
    unfold weightTable
    rw [foldl_placeStep, List.append_nil, List.foldl_reverse]
    rw [read]
    by_cases bounded : column < width
    · rw [dite_eq_left bounded, ite_eq_right isPublic,
        firstSpan_eq ⟨column, bounded⟩ slots ProductionAssignment.publicWidth]
      cases SlotLayout.slotIndex slots ⟨column, bounded⟩ with
      | none => simp
      | some slot => exact recompose_eq _ _
    · rw [dite_eq_right bounded, firstSpan_outside (by omega)]
      simp

private theorem columnWeight_outside (slots : List (LowNormAssignment.Slot width)) {column : Nat}
    (outside : width ≤ column) : columnWeight slots column = 0 := by
  unfold columnWeight PackageRows.spanWidth PackageRows.span
  rw [dite_eq_right (by omega)]
  simp

/-! ### The checksum of a step package -/

/-- The S-box rows of one block on values: the input lanes and the S-box columns at the column
weights (`CircuitRelation.sboxInterface`), through `valueCompile`. -/
def blockRows (weight : Nat → F) (forms : Nat → Option R1CS.LinearCombination)
    (start index : Nat) (block : Frame.Block) : List (F × F) :=
  (valueCompile (weight 0)
    (fun row => if row < 150 then weight (CircuitRelation.sboxColumn start index row) else 0) 0
    (Vector.ofFn fun lane : Fin 16 => combinationValue weight
      (Substitute.substitute forms (invocationInputCombination block.invocation lane.val).toR1CS))
    Permutation.schedule).2.2

/-- The weighted ports of the S-box rows of block `index`, with the S-box rows from plan row
`base`. -/
def blockValue (weight : Nat → F) (forms : Nat → Option R1CS.LinearCombination)
    (start base index : Nat) (block : Frame.Block) : F :=
  pairsSum (base + 150 * index) (blockRows weight forms start index block)

/-- The checksum of a step package's relation, from its R1CS rows and its slots: the R1CS rows
at the column weights, then, when the blocks pack, each block's S-box rows on values. -/
def denseFast (width : Nat) (pieces : List Pieces.Piece) (next : Nat) (rows : List R1CS.Row)
    (slots : List (LowNormAssignment.Slot width)) : F :=
  let table := weightTable slots
  let weight := weightAt table
  let ordinary := ((rows.zipIdx).map fun entry => ordinaryValue weight entry.2 entry.1).sum
  if CircuitRelation.packable CircuitRelation.publicColumns pieces then
    let blocks := CircuitRelation.blocksOf pieces
    let start := CircuitRelation.sboxStart pieces next
    let outputs := CircuitRelation.outputTable width start blocks
    let base := rows.length
    ordinary + ((blocks.zipIdx).map fun entry =>
      blockValue weight (Substitute.lookup outputs) start base entry.2 entry.1).sum
  else ordinary

private theorem sboxInput_eq {logicalWidth : Nat} (layout : LogicalPlan.Layout width logicalWidth)
    (blocks : List Frame.Block) (start : Nat) (invocation : Fin blocks.length) (lane : Fin 16) :
    (PoseidonSboxFamilyPlan.invocationInterface (CircuitRelation.sboxInterface layout blocks start)
        invocation).input lane =
      layout.combination (Substitute.substitute
        (Substitute.lookup (CircuitRelation.outputTable width start blocks))
        (invocationInputCombination blocks[invocation.val].invocation lane.val).toR1CS) :=
  rfl

private theorem sboxOutput_eq {logicalWidth : Nat} (layout : LogicalPlan.Layout width logicalWidth)
    (blocks : List Frame.Block) (start : Nat) (invocation : Fin blocks.length)
    (row : Fin PoseidonRetainedSlots.rows.length) :
    (PoseidonSboxFamilyPlan.invocationInterface (CircuitRelation.sboxInterface layout blocks start)
        invocation).sboxOutput row =
      layout.form (CircuitRelation.sboxColumn start invocation.val row.val) :=
  rfl

/-- The S-box rows of a block on values are the values of the S-box rows of the block's
interface. The blocks and the first S-box column are variables here, so no proof step evaluates
`CircuitRelation.sboxStart`. -/
private theorem blockRows_eq (inside : publicCount ≤ width) {weight : Nat → F}
    (zero : ∀ column, width ≤ column → weight column = 0) (blocks : List Frame.Block)
    (start : Nat) (invocation : Fin blocks.length) :
    blockRows weight (Substitute.lookup (CircuitRelation.outputTable width start blocks)) start
        invocation.val blocks[invocation.val] =
      (PoseidonSboxPlan.trace (PoseidonSboxFamilyPlan.invocationInterface
        (CircuitRelation.sboxInterface (PlanPush.identity width inside) blocks start)
        invocation)).rows.map (evalPair fun column => weight column.val) := by
  unfold blockRows PoseidonSboxPlan.trace
  refine valueCompile_eval (interface := PoseidonSboxFamilyPlan.invocationInterface
      (CircuitRelation.sboxInterface (PlanPush.identity width inside) blocks start) invocation)
    (assignment := fun column => weight column.val) (one := weight 0) ?_ ?_
    Permutation.schedule 0 ?_
  · rfl
  · intro row
    unfold PoseidonSboxPlan.sboxOutputAt
    by_cases small : row < 150
    · rw [ite_eq_left small, dite_eq_left (by rw [PoseidonRetainedSlots.rows_length]; exact small)]
      rw [sboxOutput_eq]
      exact identity_form_eval inside zero _
    · rw [ite_eq_right small,
        dite_eq_right (by rw [PoseidonRetainedSlots.rows_length]; exact small)]
      exact SparseForm.empty_eval _
  · intro lane
    rw [Vector.getElem_ofFn, sboxInput_eq]
    exact (identity_combination_eval inside zero _).symm

/-- The checksum of a step package (Rust: the factored rows of `Relation::new`): `denseFast` of
the circuit's relation rows and its package's slots. -/
def fastSum (width : Nat) (pieces : List Pieces.Piece) (next : Nat)
    (batches : List WitnessBatch) : F :=
  denseFast width pieces next
    (CircuitRelation.relationRows CircuitRelation.publicColumns pieces next width)
    (SlotLayout.runSlots width (StepPackage.slotRuns CircuitRelation.publicColumns pieces next
      batches))

theorem denseFast_eq (pieces : List Pieces.Piece) (next width : Nat)
    (fits : CircuitRelation.Fits CircuitRelation.publicColumns pieces next width)
    (inside : publicCount ≤ width) (slots : List (LowNormAssignment.Slot width)) :
    denseFast width pieces next
        (CircuitRelation.relationRows CircuitRelation.publicColumns pieces next width) slots =
      denseSum (PackageRows.rows slots
        (CircuitRelation.plan CircuitRelation.publicColumns pieces next width fits
          (PlanPush.identity width inside))
        (ringDegree * ConcreteRingColumnLayout.blockCount
          (ProductionAssignment.logicalWidth slots))).matrices := by
  have wide : ProductionAssignment.logicalWidth slots ≤ ringDegree *
      ConcreteRingColumnLayout.blockCount (ProductionAssignment.logicalWidth slots) := by
    simp only [ConcreteRingColumnLayout.blockCount, ringDegree]
    omega
  have weight : weightAt (weightTable slots) = columnWeight slots :=
    funext (weightAt_eq slots inside)
  have zero := fun column => columnWeight_outside slots (column := column)
  rw [denseSum_rows slots _ wide]
  unfold denseFast CircuitRelation.plan
  simp only [weight]
  by_cases packed : CircuitRelation.packable CircuitRelation.publicColumns pieces = true
  · rw [dite_eq_left packed, ite_eq_left packed, planSum_append, planSum_ordinary _ _ inside zero,
      planSum_family, sum_zipIdx, sum_zipIdx]
    refine congrArg₂ (· + ·) ?_ (sum_congr rfl fun invocation _ => ?_)
    · exact sum_congr rfl fun _ _ => rfl
    simp only [blockValue, zero_add, CircuitRelation.ordinary_rowCount]
    exact congrArg (pairsSum _) (blockRows_eq inside zero _ _ invocation)
  · rw [dite_eq_right packed, ite_eq_right packed, planSum_ordinary _ _ inside zero, sum_zipIdx]

/-- `fastSum` is the checksum of Lean's model of the factored rows of a step package
(`Export.BoundedDepth.FinalRows.packageRows`). -/
theorem fastSum_eq (width : Nat) (pieces : List Pieces.Piece) (next : Nat)
    (batches : List WitnessBatch)
    (fits : CircuitRelation.Fits CircuitRelation.publicColumns pieces next width)
    (inside : publicCount ≤ width) :
    fastSum width pieces next batches =
      denseSum (PackageRows.rows
        (SlotLayout.runSlots width (StepPackage.slotRuns CircuitRelation.publicColumns pieces next
          batches))
        (CircuitRelation.plan CircuitRelation.publicColumns pieces next width fits
          (PlanPush.identity width inside))
        (ringDegree * ConcreteRingColumnLayout.blockCount
          (ProductionAssignment.logicalWidth (SlotLayout.runSlots width
            (StepPackage.slotRuns CircuitRelation.publicColumns pieces next batches))))).matrices :=
  denseFast_eq pieces next width fits inside _

end NightstreamFPrime.Export.BoundedDepth.RowsParity
