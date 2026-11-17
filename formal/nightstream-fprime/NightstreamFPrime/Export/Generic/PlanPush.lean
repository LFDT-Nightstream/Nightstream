import NightstreamFPrime.Export.Generic.CircuitRelation

/-! A step circuit's relation through any layout is its relation over the source columns, pushed
through the layout (`plan_coefficient`).

`identity` reads each source column `c` as the logical column `c`. `push layout form` replaces
each entry `(c, k)` of a form over the source columns by `k` times the layout's form of `c`
(`push_coefficient`). The rows of `CircuitRelation.plan` through `layout` have the coefficients
of the rows through `identity`, pushed through `layout`. The proof follows the plan's builders:
the R1CS rows (`SourceCompiler.compileRow`), each `SparseLayer` operation, the S-box trace
(`PoseidonSboxPlan.compile`, by induction over the schedule), `Plan.append` and
`Plan.indexed`. Each builder makes its forms only with `SparseForm.add`, `SparseForm.scale`,
the constant column and the layout's forms. The row count does not depend on the layout
(`plan_rowCount`). -/

namespace NightstreamFPrime.Export.Generic.PlanPush

open NightstreamFPrime.Spec
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Generic.LogicalPlan (Layout publicCount)

variable {width logicalWidth : Nat}

/-- The layout that reads source column `c` as logical column `c`. -/
def identity (width : Nat) (inside : publicCount ≤ width) : Layout width width where
  map := ⟨fun column => SparseForm.singleton column 1⟩
  publicFits := inside
  publicForm := fun _ _ => rfl

/-- A form over the source columns, read through `layout`: entry `(c, k)` adds `k` times the
layout's form of `c`. -/
def push (layout : Layout width logicalWidth) (form : SparseForm width) :
    SparseForm logicalWidth :=
  ⟨form.entries.flatMap fun entry =>
    (SparseForm.scale entry.coefficient (layout.form entry.column.val)).entries⟩

private theorem push_cons (layout : Layout width logicalWidth) (entry : SparseEntry width)
    (rest : List (SparseEntry width)) :
    push layout ⟨entry :: rest⟩ =
      SparseForm.add (SparseForm.scale entry.coefficient (layout.form entry.column.val))
        (push layout ⟨rest⟩) := by
  simp only [push, List.flatMap_cons, SparseForm.add]

theorem push_coefficient (layout : Layout width logicalWidth) (form : SparseForm width)
    (x : Fin logicalWidth) :
    (push layout form).coefficient x =
      (form.entries.map fun entry =>
        entry.coefficient * (layout.form entry.column.val).coefficient x).sum := by
  obtain ⟨entries⟩ := form
  induction entries with
  | nil => rfl
  | cons entry rest inductionHypothesis =>
      rw [push_cons, SparseForm.add_coefficient, SparseForm.scale_coefficient,
        inductionHypothesis, List.map_cons, List.sum_cons]

/-! ### Forms that agree -/

/-- `form` has the coefficients of `source` pushed through `layout`. -/
private def Agrees (layout : Layout width logicalWidth) (form : SparseForm logicalWidth)
    (source : SparseForm width) : Prop :=
  ∀ x, form.coefficient x = (push layout source).coefficient x

variable {layout : Layout width logicalWidth}

private theorem agrees_empty : Agrees layout .empty .empty := fun _ => rfl

private theorem Agrees.add {form form' : SparseForm logicalWidth}
    {source source' : SparseForm width}
    (left : Agrees layout form source) (right : Agrees layout form' source') :
    Agrees layout (SparseForm.add form form') (SparseForm.add source source') := by
  intro x
  rw [SparseForm.add_coefficient, left x, right x, push_coefficient, push_coefficient,
    push_coefficient]
  simp only [SparseForm.add, List.map_append, List.sum_append]

private theorem Agrees.scale {form : SparseForm logicalWidth} {source : SparseForm width}
    (agrees : Agrees layout form source) (scalar : F) :
    Agrees layout (SparseForm.scale scalar form) (SparseForm.scale scalar source) := by
  intro x
  rw [SparseForm.scale_coefficient, agrees x, push_coefficient, push_coefficient]
  simp only [SparseForm.scale, List.map_map, Function.comp_def, mul_assoc, List.sum_map_mul_left]

private theorem push_singleton (column : Fin width) (value : F) (x : Fin logicalWidth) :
    (push layout (SparseForm.singleton column value)).coefficient x =
      value * (layout.form column.val).coefficient x := by
  rw [push_coefficient]
  simp only [SparseForm.singleton, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
    add_zero]

private theorem agrees_form (inside : publicCount ≤ width) (column : Nat) :
    Agrees layout (layout.form column) ((identity width inside).form column) := by
  intro x
  by_cases bounded : column < width
  · simp only [LogicalPlan.Layout.form, identity, bounded, dite_true]
    rw [push_singleton, one_mul]
    simp only [LogicalPlan.Layout.form, bounded, dite_true]
  · simp only [LogicalPlan.Layout.form, bounded, dite_false]
    rfl

private theorem agrees_one (inside : publicCount ≤ width) (value : F) :
    Agrees layout (SparseForm.singleton layout.one value)
      (SparseForm.singleton (identity width inside).one value) := by
  intro x
  have positive : 0 < width := Nat.lt_of_lt_of_le (by decide) inside
  have zero : layout.form 0 = SparseForm.singleton layout.one 1 := by
    simp only [LogicalPlan.Layout.form, positive, dite_true]
    exact layout.publicForm ⟨0, positive⟩ (show 0 < publicCount by decide)
  rw [push_singleton]
  change _ = value * (layout.form 0).coefficient x
  rw [zero, SparseForm.singleton_coefficient, SparseForm.singleton_coefficient]
  split <;> simp

/-! ### The R1CS rows -/

private theorem compileTerms_map (layout : Layout width logicalWidth) :
    ∀ (terms : List (Nat × F)) (bounded : ∀ term ∈ terms, term.1 < width),
      SourceCompiler.compileTerms layout.map terms bounded = layout.terms terms
  | [], _ => rfl
  | (column, coefficient) :: rest, bounded => by
      have inside : column < width := bounded (column, coefficient) List.mem_cons_self
      simp only [SourceCompiler.compileTerms, LogicalPlan.Layout.terms,
        compileTerms_map layout rest, LogicalPlan.Layout.form, inside, dite_true]

private theorem compileCombination_map (layout : Layout width logicalWidth)
    (combination : R1CS.LinearCombination)
    (bounded : SourceCompiler.CombinationBounded width combination) :
    SourceCompiler.compileCombination layout.map layout.one combination bounded =
      layout.combination combination :=
  congrArg (SparseForm.add _) (compileTerms_map layout combination.terms bounded)

private theorem agrees_terms (inside : publicCount ≤ width) :
    ∀ terms : List (Nat × F),
      Agrees layout (layout.terms terms) ((identity width inside).terms terms)
  | [] => agrees_empty
  | (column, coefficient) :: rest =>
      ((agrees_form inside column).scale coefficient).add (agrees_terms inside rest)

private theorem agrees_combination (inside : publicCount ≤ width)
    (combination : R1CS.LinearCombination) :
    Agrees layout (layout.combination combination)
      ((identity width inside).combination combination) :=
  (agrees_one inside combination.constant).add (agrees_terms inside combination.terms)

private theorem ordinary_agrees {forms : OrdinaryRow.Forms logicalWidth}
    {source : OrdinaryRow.Forms width} (a : Agrees layout forms.a source.a)
    (b : Agrees layout forms.b source.b) (c : Agrees layout forms.c source.c)
    (port : Fin Spec.ProductionRelation.meaningfulPortCount) :
    Agrees layout (forms.meaningfulForm port) (source.meaningfulForm port) := by
  rcases port with ⟨_ | _ | _ | _, _⟩
  exacts [a, b, c, agrees_empty]

/-! ### Plans that agree -/

/-- Two plans have the same row count, and rows with the same index agree on each port. -/
private def PlanAgrees (layout : Layout width logicalWidth) (plan : Plan logicalWidth)
    (source : Plan width) : Prop :=
  plan.rowCount = source.rowCount ∧
    ∀ (row : Fin plan.rowCount) (row' : Fin source.rowCount), row.val = row'.val →
      ∀ port, Agrees layout (plan.forms row port) (source.forms row' port)

private theorem ordinary_forms (layout : Layout width logicalWidth) (rows : List R1CS.Row)
    (fits : LogicalPlan.Fits rows width) (row : Nat) (below : row < rows.length)
    (port : Fin Spec.ProductionRelation.meaningfulPortCount) :
    (LogicalPlan.plan rows width fits layout).forms ⟨row, below⟩ port =
      OrdinaryRow.Forms.meaningfulForm ⟨layout.combination rows[row].a,
        layout.combination rows[row].b, layout.combination rows[row].c⟩ port := by
  change (SourceCompiler.compileRow layout.map layout.one rows[row]
    ((LogicalPlan.program rows width fits).bounded ⟨row, below⟩)).meaningfulForm port = _
  simp only [SourceCompiler.compileRow, compileCombination_map]

private theorem ordinaryPlan_agrees (inside : publicCount ≤ width) (rows : List R1CS.Row)
    (fits : LogicalPlan.Fits rows width) :
    PlanAgrees layout (LogicalPlan.plan rows width fits layout)
      (LogicalPlan.plan rows width fits (identity width inside)) := by
  refine ⟨rfl, fun ⟨row, below⟩ ⟨row', below'⟩ same port => ?_⟩
  simp only at same
  subst same
  rw [ordinary_forms layout rows fits row below, ordinary_forms _ rows fits row below']
  exact ordinary_agrees (agrees_combination inside _) (agrees_combination inside _)
    (agrees_combination inside _) port

private theorem append_forms (left right : Plan logicalWidth)
    (rowCount_le : left.rowCount + right.rowCount ≤ 2 ^ Lifecycle.cubeVariables)
    (row : Fin (Plan.append left right rowCount_le).rowCount)
    (port : Fin Spec.ProductionRelation.meaningfulPortCount) :
    (Plan.append left right rowCount_le).forms row port =
      if inLeft : row.val < left.rowCount then left.forms ⟨row.val, inLeft⟩ port
      else right.forms ⟨row.val - left.rowCount, by
        have := row.isLt
        simp only [Plan.append_rowCount] at this
        omega⟩ port := by
  by_cases inLeft : row.val < left.rowCount
  · simp only [Plan.append, Plan.splitIndex, inLeft, dite_true]
  · simp only [Plan.append, Plan.splitIndex, inLeft, dite_false]

private theorem append_agrees {left right : Plan logicalWidth} {left' right' : Plan width}
    (agreesLeft : PlanAgrees layout left left') (agreesRight : PlanAgrees layout right right')
    (bound : left.rowCount + right.rowCount ≤ 2 ^ Lifecycle.cubeVariables)
    (bound' : left'.rowCount + right'.rowCount ≤ 2 ^ Lifecycle.cubeVariables) :
    PlanAgrees layout (Plan.append left right bound) (Plan.append left' right' bound') := by
  obtain ⟨leftCount, leftForms⟩ := agreesLeft
  obtain ⟨rightCount, rightForms⟩ := agreesRight
  refine ⟨by simp only [Plan.append_rowCount, leftCount, rightCount],
    fun row row' same port => ?_⟩
  rw [append_forms, append_forms]
  by_cases inLeft : row.val < left.rowCount
  · have inLeft' : row'.val < left'.rowCount := by omega
    simp only [inLeft, inLeft', dite_true]
    exact leftForms _ _ same port
  · have inLeft' : ¬ row'.val < left'.rowCount := by omega
    simp only [inLeft, inLeft', dite_false]
    exact rightForms _ _ (by simp only; omega) port

private theorem indexed_agrees {blockCount rowCount : Nat}
    {forms : Fin blockCount → Fin rowCount →
      Fin Spec.ProductionRelation.meaningfulPortCount → SparseForm logicalWidth}
    {source : Fin blockCount → Fin rowCount →
      Fin Spec.ProductionRelation.meaningfulPortCount → SparseForm width}
    (agrees : ∀ block row port, Agrees layout (forms block row port) (source block row port))
    (bound : blockCount * rowCount ≤ 2 ^ Lifecycle.cubeVariables)
    (bound' : blockCount * rowCount ≤ 2 ^ Lifecycle.cubeVariables) :
    PlanAgrees layout (Plan.indexed forms bound) (Plan.indexed source bound') := by
  refine ⟨rfl, fun ⟨row, below⟩ ⟨row', below'⟩ same port => ?_⟩
  simp only at same
  subst same
  exact agrees _ _ port

/-! ### The S-box rows -/

/-- Two states agree lane by lane. -/
private def StateAgrees (layout : Layout width logicalWidth)
    (state : SparseLayer.State logicalWidth) (source : SparseLayer.State width) : Prop :=
  ∀ lane, Agrees layout (state lane) (source lane)

/-- Two S-box rows agree on their input and output. -/
private def RowAgrees (layout : Layout width logicalWidth) (row : SboxRow.Forms logicalWidth)
    (source : SboxRow.Forms width) : Prop :=
  Agrees layout row.input source.input ∧ Agrees layout row.output source.output

variable {state : SparseLayer.State logicalWidth} {source : SparseLayer.State width}

private theorem get_agrees (agrees : StateAgrees layout state source) (index : Nat) :
    Agrees layout (SparseLayer.get state index) (SparseLayer.get source index) := by
  by_cases bounded : index < 16
  · simp only [SparseLayer.get, bounded, dite_true]
    exact agrees _
  · simp only [SparseLayer.get, bounded, dite_false]
    exact agrees_empty

private theorem mat4_agrees (agrees : StateAgrees layout state source) (base lane : Nat) :
    Agrees layout (SparseLayer.mat4 state base lane) (SparseLayer.mat4 source base lane) := by
  have get := get_agrees agrees
  rcases lane with _ | _ | _ | _
  · exact ((((get _).scale _).add ((get _).scale _)).add (get _)).add (get _)
  · exact (((get _).add ((get _).scale _)).add ((get _).scale _)).add (get _)
  · exact (((get _).add (get _)).add ((get _).scale _)).add ((get _).scale _)
  · exact ((((get _).scale _).add (get _)).add (get _)).add ((get _).scale _)

private theorem block_agrees (agrees : StateAgrees layout state source) (index : Nat) :
    Agrees layout (SparseLayer.block state index) (SparseLayer.block source index) :=
  mat4_agrees agrees _ _

private theorem foldl_agrees {α : Type} {forms : α → SparseForm logicalWidth}
    {sources : α → SparseForm width} (agrees : ∀ a, Agrees layout (forms a) (sources a)) :
    ∀ (values : List α) {initial : SparseForm logicalWidth} {initial' : SparseForm width},
      Agrees layout initial initial' →
        Agrees layout ((values.map forms).foldl SparseLayer.add initial)
          ((values.map sources).foldl SparseLayer.add initial')
  | [], _, _, start => start
  | value :: rest, _, _, start => foldl_agrees agrees rest (start.add (agrees value))

private theorem external_agrees (agrees : StateAgrees layout state source) :
    StateAgrees layout (SparseLayer.external state) (SparseLayer.external source) := fun lane =>
  (block_agrees agrees lane.val).add
    (foldl_agrees (fun _ => block_agrees agrees _) _ agrees_empty)

private theorem internal_agrees (agrees : StateAgrees layout state source) :
    StateAgrees layout (SparseLayer.internal state) (SparseLayer.internal source) := by
  intro lane
  unfold SparseLayer.internal SparseLayer.sum SparseLayer.sumForms
  exact ((agrees lane).scale _).add (foldl_agrees (get_agrees agrees) _ agrees_empty)

/-- The interfaces agree on the constant column, the input and each S-box output. -/
private structure InterfaceAgrees (layout : Layout width logicalWidth)
    (interface : PoseidonSboxPlan.Interface logicalWidth)
    (source : PoseidonSboxPlan.Interface width) : Prop where
  one : ∀ value, Agrees layout (SparseForm.singleton interface.oneColumn value)
    (SparseForm.singleton source.oneColumn value)
  input : StateAgrees layout interface.input source.input
  sboxOutput : ∀ row, Agrees layout (interface.sboxOutput row) (source.sboxOutput row)

variable {interface : PoseidonSboxPlan.Interface logicalWidth}
  {sourceInterface : PoseidonSboxPlan.Interface width}

private theorem sboxOutputAt_agrees (agrees : InterfaceAgrees layout interface sourceInterface)
    (index : Nat) :
    Agrees layout (PoseidonSboxPlan.sboxOutputAt interface index)
      (PoseidonSboxPlan.sboxOutputAt sourceInterface index) := by
  by_cases bounded : index < PoseidonRetainedSlots.rows.length
  · rw [PoseidonSboxPlan.sboxOutputAt, PoseidonSboxPlan.sboxOutputAt, dite_eq_left bounded,
      dite_eq_left bounded]
    exact agrees.sboxOutput _
  · rw [PoseidonSboxPlan.sboxOutputAt, PoseidonSboxPlan.sboxOutputAt, dite_eq_right bounded,
      dite_eq_right bounded]
    exact agrees_empty

private theorem addConstant_agrees (agrees : InterfaceAgrees layout interface sourceInterface)
    {form : SparseForm logicalWidth} {source : SparseForm width} (same : Agrees layout form source)
    (value : F) :
    Agrees layout (SparseLayer.addConstant interface.oneColumn form value)
      (SparseLayer.addConstant sourceInterface.oneColumn source value) :=
  same.add (agrees.one value)

private theorem fullOutput_agrees (agrees : InterfaceAgrees layout interface sourceInterface)
    (next : Nat) :
    StateAgrees layout (PoseidonSboxPlan.fullOutput interface next)
      (PoseidonSboxPlan.fullOutput sourceInterface next) := by
  intro lane
  simp only [PoseidonSboxPlan.fullOutput]
  exact sboxOutputAt_agrees agrees _

private theorem fullRows_agrees (agrees : InterfaceAgrees layout interface sourceInterface)
    (constants : List (List Nat)) (round next : Nat) (states : StateAgrees layout state source) :
    List.Forall₂ (RowAgrees layout)
      (PoseidonSboxPlan.fullRows interface constants round next state)
      (PoseidonSboxPlan.fullRows sourceInterface constants round next source) := by
  unfold PoseidonSboxPlan.fullRows
  refine List.forall₂_iff_get.mpr ⟨by rw [List.length_ofFn, List.length_ofFn],
    fun index _ _ => ?_⟩
  rw [List.get_ofFn, List.get_ofFn]
  refine ⟨?_, ?_⟩ <;> dsimp only
  · exact addConstant_agrees agrees (states _) _
  · exact fullOutput_agrees agrees next _

private theorem compileStep_agrees (agrees : InterfaceAgrees layout interface sourceInterface)
    (next : Nat) (states : StateAgrees layout state source) (step : Permutation.Step) :
    (PoseidonSboxPlan.compileStep interface next state step).nextSbox =
        (PoseidonSboxPlan.compileStep sourceInterface next source step).nextSbox ∧
      StateAgrees layout (PoseidonSboxPlan.compileStep interface next state step).state
        (PoseidonSboxPlan.compileStep sourceInterface next source step).state ∧
      List.Forall₂ (RowAgrees layout)
        (PoseidonSboxPlan.compileStep interface next state step).rows
        (PoseidonSboxPlan.compileStep sourceInterface next source step).rows := by
  cases step with
  | initialLayer => exact ⟨rfl, external_agrees states, .nil⟩
  | initialFullRound round =>
      exact ⟨rfl, external_agrees (fullOutput_agrees agrees next),
        fullRows_agrees agrees _ round next states⟩
  | partialRound round =>
      refine ⟨rfl, internal_agrees fun lane => ?_, List.Forall₂.cons ?_ .nil⟩
      · by_cases zero : lane.val = 0
        · simp only [PoseidonSboxPlan.partialState, zero, ite_true, PoseidonSboxPlan.partialOutput]
          exact sboxOutputAt_agrees agrees next
        · simp only [PoseidonSboxPlan.partialState, zero, ite_false]
          exact states lane
      · refine ⟨?_, ?_⟩ <;> dsimp only
        · exact addConstant_agrees agrees (states 0) _
        · exact sboxOutputAt_agrees agrees next
  | terminalFullRound round =>
      exact ⟨rfl, external_agrees (fullOutput_agrees agrees next),
        fullRows_agrees agrees _ round next states⟩

private theorem compile_agrees (agrees : InterfaceAgrees layout interface sourceInterface) :
    ∀ (steps : List Permutation.Step) (next next' : Nat) {state : SparseLayer.State logicalWidth}
      {source : SparseLayer.State width}, next = next' → StateAgrees layout state source →
      StateAgrees layout (PoseidonSboxPlan.compile interface next state steps).state
          (PoseidonSboxPlan.compile sourceInterface next' source steps).state ∧
        List.Forall₂ (RowAgrees layout) (PoseidonSboxPlan.compile interface next state steps).rows
          (PoseidonSboxPlan.compile sourceInterface next' source steps).rows
  | [], _, _, _, _, _, states => ⟨states, .nil⟩
  | step :: rest, next, _, _, _, rfl, states => by
      obtain ⟨sameNext, headStates, headRows⟩ := compileStep_agrees agrees next states step
      obtain ⟨tailStates, tailRows⟩ := compile_agrees agrees rest _ _ sameNext headStates
      exact ⟨tailStates, List.rel_append headRows tailRows⟩

private theorem sboxForms_agrees {row : SboxRow.Forms logicalWidth} {source : SboxRow.Forms width}
    (agrees : RowAgrees layout row source)
    (port : Fin Spec.ProductionRelation.meaningfulPortCount) :
    Agrees layout (row.meaningfulForm port) (source.meaningfulForm port) := by
  rcases port with ⟨_ | _ | _ | _ | _, _⟩
  exacts [agrees_empty, agrees_empty, agrees.2, agrees.1, agrees_empty]

private theorem family_agrees {count : Nat}
    {interface : PoseidonSboxFamilyPlan.Interface logicalWidth count}
    {sourceInterface : PoseidonSboxFamilyPlan.Interface width count}
    (agrees : ∀ invocation, InterfaceAgrees layout
      (PoseidonSboxFamilyPlan.invocationInterface interface invocation)
      (PoseidonSboxFamilyPlan.invocationInterface sourceInterface invocation))
    (bound : count * 150 ≤ 2 ^ Lifecycle.cubeVariables)
    (bound' : count * 150 ≤ 2 ^ Lifecycle.cubeVariables) :
    PlanAgrees layout (PoseidonSboxFamilyPlan.plan interface bound)
      (PoseidonSboxFamilyPlan.plan sourceInterface bound') := by
  refine indexed_agrees (fun invocation row port => ?_) bound bound'
  have rows : List.Forall₂ (RowAgrees layout)
      (PoseidonSboxPlan.trace
        (PoseidonSboxFamilyPlan.invocationInterface interface invocation)).rows
      (PoseidonSboxPlan.trace
        (PoseidonSboxFamilyPlan.invocationInterface sourceInterface invocation)).rows :=
    (compile_agrees (agrees invocation) Permutation.schedule 0 0 rfl (agrees invocation).input).2
  simp only [PoseidonSboxFamilyPlan.rowForms, PoseidonSboxFamilyPlan.rowAt,
    PoseidonRetainedRows.rows, List.get_eq_getElem, List.getElem_map]
  exact sboxForms_agrees (by simpa only [List.get_eq_getElem] using rows.get _ _) port

/-! ### The step circuit's relation -/

private theorem sboxInterface_agrees (inside : publicCount ≤ width) (blocks : List Frame.Block)
    (start : Nat) (invocation : Fin blocks.length) :
    InterfaceAgrees layout
      (PoseidonSboxFamilyPlan.invocationInterface
        (CircuitRelation.sboxInterface layout blocks start) invocation)
      (PoseidonSboxFamilyPlan.invocationInterface
        (CircuitRelation.sboxInterface (identity width inside) blocks start) invocation) where
  one := agrees_one inside
  input := fun _ => agrees_combination inside _
  sboxOutput := fun _ => agrees_form inside _

private theorem plan_agrees (publicColumns : Nat) (pieces : List Pieces.Piece) (next : Nat)
    (fits : CircuitRelation.Fits publicColumns pieces next width) (inside : publicCount ≤ width)
    (layout : Layout width logicalWidth) :
    PlanAgrees layout (CircuitRelation.plan publicColumns pieces next width fits layout)
      (CircuitRelation.plan publicColumns pieces next width fits (identity width inside)) := by
  unfold CircuitRelation.plan
  by_cases packed : CircuitRelation.packable publicColumns pieces = true
  · rw [dite_eq_left packed, dite_eq_left packed]
    exact append_agrees (ordinaryPlan_agrees inside _ _)
      (family_agrees (sboxInterface_agrees inside _ _) _ _) _ _
  · rw [dite_eq_right packed, dite_eq_right packed]
    exact ordinaryPlan_agrees inside _ _

/-- The row count of a step circuit's relation does not depend on the layout. -/
theorem plan_rowCount (publicColumns : Nat) (pieces : List Pieces.Piece) (next width : Nat)
    (fits : CircuitRelation.Fits publicColumns pieces next width) {logicalWidth logicalWidth' : Nat}
    (layout : Layout width logicalWidth) (layout' : Layout width logicalWidth') :
    (CircuitRelation.plan publicColumns pieces next width fits layout).rowCount =
      (CircuitRelation.plan publicColumns pieces next width fits layout').rowCount := by
  unfold CircuitRelation.plan
  by_cases packed : CircuitRelation.packable publicColumns pieces = true
  · rw [dite_eq_left packed, dite_eq_left packed]
    rfl
  · rw [dite_eq_right packed, dite_eq_right packed]
    rfl

/-- Each port of each row of a step circuit's relation through `layout` has the coefficients of
the same row through `identity`, pushed through `layout`. -/
theorem plan_coefficient (publicColumns : Nat) (pieces : List Pieces.Piece) (next width : Nat)
    (fits : CircuitRelation.Fits publicColumns pieces next width) (inside : publicCount ≤ width)
    {logicalWidth : Nat} (layout : Layout width logicalWidth)
    (row : Fin (CircuitRelation.plan publicColumns pieces next width fits layout).rowCount)
    (port : Fin Spec.ProductionRelation.matrixCount) (x : Fin logicalWidth) :
    ((CircuitRelation.plan publicColumns pieces next width fits layout).portForm row
        port).coefficient x =
      (push layout ((CircuitRelation.plan publicColumns pieces next width fits
        (identity width inside)).portForm
          (Fin.cast (plan_rowCount publicColumns pieces next width fits layout
            (identity width inside)) row) port)).coefficient x := by
  unfold Plan.portForm
  cases meaningfulPort? port with
  | none => rfl
  | some meaningful =>
      exact (plan_agrees publicColumns pieces next fits inside layout).2 row _ rfl meaningful x

end NightstreamFPrime.Export.Generic.PlanPush
