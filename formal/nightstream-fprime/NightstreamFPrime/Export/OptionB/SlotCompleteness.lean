import NightstreamFPrime.Export.OptionB.SlotRuns
import NightstreamFPrime.Export.Generic.RelationCompleteness
import NightstreamFPrime.Export.Generic.WideLayout

/-! Completeness of a step package's relation at its slot layout (CW-10.2). The honest assignment
reads the public input and every slot from the completed environment
(`RelationCompleteness.completeEnv`, `honestAssignment`). The slot layout reads that environment
below `slotEnd`: a column with a slot reads its slot, and a column without one is a dead local,
which the completed environment holds at 0 (`dead_of_code`). So the relation's plan holds
(`rowsZero_of_holds`).

Each coordinate is small when each slot's value is valid for its kind. A field slot is always
valid. The package gives a slot the kind bit only at a `bit` hint. `bitsForced` is a computed
check that a constraint `x * (x - 1)` of the pieces forces each such column to be a bit, as
`CircuitRelation.packable` is a computed check of the blocks. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.OptionB.SlotCompleteness

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export
open NightstreamFPrime.Export.Generic
open NightstreamFPrime.Export.Generic.Pieces
open NightstreamFPrime.Export.Generic.Frame
open NightstreamFPrime.Export.Generic.CircuitRelation
open NightstreamFPrime.Export.Generic.PieceRowsCompleteness
open NightstreamFPrime.Export.Generic.RelationCompleteness
open NightstreamFPrime.Export.OptionB.SlotRuns
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-! ### Arrays -/

theorem getD_set! {α : Type} (codes : Array α) (position : Nat) (value : α) (column : Nat)
    (fallback : α) :
    (codes.set! position value).getD column fallback =
      if position = column ∧ column < codes.size then value else codes.getD column fallback := by
  rw [Array.getD_eq_getD_getElem?, Array.getD_eq_getD_getElem?]
  by_cases same : position = column
  · subst same
    by_cases inside : position < codes.size <;> simp [inside, Array.set!]
  · simp [same, Array.set!, Array.getElem?_setIfInBounds_ne same]

theorem size_set! {α : Type} (codes : Array α) (position : Nat) (value : α) :
    (codes.set! position value).size = codes.size := by
  simp [Array.set!]

theorem foldl_invariant {α β : Type} (holds : β → Prop) (step : β → α → β) :
    ∀ (items : List α) (initial : β), holds initial →
      (∀ value item, item ∈ items → holds value → holds (step value item)) →
      holds (items.foldl step initial)
  | [], _, initial, _ => initial
  | item :: rest, initial, start, preserves =>
      foldl_invariant holds step rest _ (preserves initial item List.mem_cons_self start)
        fun value other member => preserves value other (List.mem_cons_of_mem _ member)

/-! ### Codes -/

/-- A scanned column whose code is 3 is a dead local of a block. -/
theorem dead_of_code (pieces : List Piece) (next : Nat) (batches : List WitnessBatch)
    {column : Nat} (inside : column < slotSize pieces next)
    (code : (slotCodes pieces next batches).getD column 3 = 3) :
    deadIn (blocksOf pieces) column = true := by
  let size := slotSize pieces next
  let Good : Array Nat → Prop := fun codes => codes.size = size ∧
    ∀ column, column < size → codes.getD column 3 = 3 → deadIn (blocksOf pieces) column = true
  have good : Good (slotCodes pieces next batches) := by
    unfold slotCodes
    dsimp only
    apply foldl_invariant Good
    · apply foldl_invariant Good
      · refine ⟨by simp [size, slotSize], fun current below found => ?_⟩
        rw [Array.getD_eq_getD_getElem?] at found
        simp [size, slotSize] at below
        simp [below] at found
      · intro codes block member before
        apply foldl_invariant Good
        · exact before
        · intro codes offset offsetMember previous
          refine ⟨by rw [size_set!]; exact previous.1, fun current below found => ?_⟩
          rw [getD_set!] at found
          split at found
          · rename_i hit
            unfold deadIn
            rw [List.any_eq_true]
            refine ⟨block, member, ?_⟩
            have offsetBound := List.mem_range.mp offsetMember
            simp only [Block.dead, Bool.and_eq_true, decide_eq_true_eq]
            omega
          · exact previous.2 current below found
    · intro codes batch _ before
      apply foldl_invariant Good
      · exact before
      · intro codes entry _ previous
        obtain ⟨hint, index⟩ := entry
        cases hint with
        | bit source bit =>
            refine ⟨by rw [size_set!]; exact previous.1, fun current below found => ?_⟩
            rw [getD_set!] at found
            split at found
            · exact absurd found (by decide)
            · exact previous.2 current below found
        | inverseOrZero source => exact previous
        | quotientFive source => exact previous
        | remainderFive source => exact previous
  exact good.2 column inside code

/-! ### Forced bits -/

/-- The column that a constraint `x * (x - 1)` forces to be a bit. -/
def booleanColumn? : Expr → Option Nat
  | .mul (.var column) other => if other = Expr.var column - 1 then some column else none
  | _ => none

theorem bit_of_booleanColumn {expression : Expr} {column : Nat}
    (found : booleanColumn? expression = some column) {env : Env} (holds : expression.eval env = 0) :
    env column = 0 ∨ env column = 1 := by
  match expression, found with
  | .mul (.var current) other, found =>
      simp only [booleanColumn?] at found
      split at found
      · rename_i same
        have columnEq : current = column := Option.some.inj found
        subst columnEq
        subst same
        have product : env current * (env current - 1) = 0 := by
          simpa using! holds
        rcases Spec.GoldilocksPrime.baseFieldNoZeroDivisors _ _ product with zero | one
        · exact Or.inl zero
        · exact Or.inr (sub_eq_zero.mp one)
      · cases found

theorem column_lt_of_booleanColumn {expression : Expr} {column bound : Nat}
    (found : booleanColumn? expression = some column) (below : expression.VarsBelow bound) :
    column < bound := by
  match expression, found with
  | .mul (.var current) other, found =>
      simp only [booleanColumn?] at found
      split at found
      · cases found
        exact below.1
      · cases found

/-- The columns that a constraint piece `x * (x - 1)` forces to be bits. -/
def forcedColumns (pieces : List Piece) : List Nat :=
  pieces.filterMap fun piece => match piece with
    | .constraint expression => booleanColumn? expression
    | _ => none

def forcedMarks (pieces : List Piece) (size : Nat) : Array Bool :=
  (forcedColumns pieces).foldl (fun marks column => marks.set! column true)
    (Array.replicate size false)

theorem mem_of_marked {pieces : List Piece} {size column : Nat}
    (marked : (forcedMarks pieces size).getD column false = true) :
    column ∈ forcedColumns pieces := by
  let Good : Array Bool → Prop := fun marks =>
    ∀ column, marks.getD column false = true → column ∈ forcedColumns pieces
  have good : Good (forcedMarks pieces size) := by
    apply foldl_invariant Good
    · intro current found
      rw [Array.getD_eq_getD_getElem?] at found
      by_cases inside : current < size <;> simp [inside] at found
    · intro marks forced member previous current found
      rw [getD_set!] at found
      split at found
      · rename_i hit
        rw [← hit.1]
        exact member
      · exact previous current found
  exact good column marked

/-! ### Locality -/

/-- The variables of an expression. -/
def exprVars : Expr → List Nat
  | .var column => [column]
  | .const _ => []
  | .add left right => exprVars left ++ exprVars right
  | .mul left right => exprVars left ++ exprVars right

theorem eval_of_vars {left right : Env} :
    ∀ expression : Expr, (∀ column ∈ exprVars expression, left column = right column) →
      expression.eval left = expression.eval right
  | .var column, agree => agree column (by simp [exprVars])
  | .const _, _ => rfl
  | .add first second, agree => by
      simp only [Expr.eval_add,
        eval_of_vars first fun column member => agree column (by simp [exprVars, member]),
        eval_of_vars second fun column member => agree column (by simp [exprVars, member])]
  | .mul first second, agree => by
      simp only [Expr.eval_mul,
        eval_of_vars first fun column member => agree column (by simp [exprVars, member]),
        eval_of_vars second fun column member => agree column (by simp [exprVars, member])]

/-- The columns that a piece's constraints read: a constraint's variables, and a block's or a ring
product's lane variables and locals. -/
def pieceReads : Piece → List Nat
  | .constraint expression => exprVars expression
  | .permutation start lanes => lanes.flatMap exprVars ++ (List.range 1096).map (start + ·)
  | .ringProduct cells start lanes =>
      lanes.flatMap exprVars ++ (List.range (ringDegree * cells)).map (start + ·)

theorem lanes_agree {left right : Env} {lanes : List Expr}
    (agree : ∀ column ∈ lanes.flatMap exprVars, left column = right column) (index : Nat) :
    (lanes.getD index 0).eval left = (lanes.getD index 0).eval right := by
  by_cases inside : index < lanes.length
  · rw [List.getD_eq_getElem _ _ inside]
    exact eval_of_vars _ fun column member =>
      agree column (List.mem_flatMap.mpr ⟨_, List.getElem_mem inside, member⟩)
  · rw [List.getD_eq_default _ _ (by omega)]
    rfl

/-- A piece's constraints hold in any environment that agrees on the columns the piece reads. -/
theorem holds_of_reads (piece : Piece) (valid : piece.Valid) {env other : Env}
    (agree : ∀ column ∈ pieceReads piece, other column = env column)
    (holds : ConstraintsHold env (piece.constraints (PilotData.canonicalConstraints ()))) :
    ConstraintsHold other (piece.constraints (PilotData.canonicalConstraints ())) := by
  cases piece with
  | constraint expression =>
      intro current member
      simp only [Piece.constraints, List.mem_singleton] at member
      subst member
      rw [eval_of_vars current agree]
      exact holds current List.mem_cons_self
  | ringProduct cells start lanes =>
      have lanesSame : ∀ column ∈ lanes.flatMap exprVars, other column = env column :=
        fun column member => agree column (List.mem_append_left _ member)
      intro current member
      have original := holds current member
      simp only [Piece.constraints, ringTemplate, List.mem_map, List.mem_ofFn] at member
      obtain ⟨_, ⟨index, rfl⟩, rfl⟩ := member
      rw [Lifecycle.Nebula.Expr.eval_subst] at original ⊢
      rw [Expr.eval_eq_of_agree_below _ _ _ _ (ringConstraint_varsBelow cells index)]
      · exact original
      · intro column below
        by_cases input : column < ringInputs cells
        · simp only [laneSubst_input lanes input]
          exact lanes_agree lanesSame column
        · obtain ⟨offset, rfl⟩ : ∃ offset, column = ringInputs cells + offset :=
            ⟨column - ringInputs cells, by omega⟩
          simp only [laneSubst_local, Expr.eval_var]
          exact agree _ (List.mem_append_right _
            (List.mem_map.mpr ⟨offset, List.mem_range.mpr (by omega), rfl⟩))
  | permutation start lanes =>
      have linear : LinearLanes lanes := valid
      have lanesSame : ∀ column ∈ lanes.flatMap exprVars, other column = env column :=
        fun column member => agree column (List.mem_append_left _ member)
      have before := Pilot.canonicalPermutationInvocation_complete _ env (block_canonical linear holds)
      have after := Pilot.permutationInvocationHolds_of_agreement (invocation 0 0 start lanes) env
        other
        (fun lane => (block_input (block := ⟨start, lanes⟩) linear other lane).trans
          ((lanes_agree lanesSame lane.val).trans
            (block_input (block := ⟨start, lanes⟩) linear env lane).symm))
        (fun index below => agree _ (List.mem_append_right _
          (List.mem_map.mpr ⟨index, List.mem_range.mpr below, rfl⟩)))
        before
      exact permutation_sound 0 0 start lanes valid other after

/-! ### Quiet columns -/

/-- The scanned columns that the honest environment sets to 0: the bit and centered slots that
no constraint `x * (x - 1)` forces. -/
def quietMarks (publicColumns : Nat) (pieces : List Piece) (next : Nat)
    (batches : List WitnessBatch) : Array Bool :=
  let codes := slotCodes pieces next batches
  let marks := forcedMarks pieces (slotSize pieces next)
  Array.ofFn (n := slotSize pieces next) fun column =>
    decide (publicColumns ≤ column.val) && decide (codes.getD column.val 3 < 2) &&
      !marks.getD column.val false

theorem quietMarks_getD (publicColumns : Nat) (pieces : List Piece) (next : Nat)
    (batches : List WitnessBatch) (column : Nat) :
    (quietMarks publicColumns pieces next batches).getD column false =
      (decide (column < slotSize pieces next) && decide (publicColumns ≤ column) &&
        decide ((slotCodes pieces next batches).getD column 3 < 2) &&
          !(forcedMarks pieces (slotSize pieces next)).getD column false) := by
  rw [Array.getD_eq_getD_getElem?]
  unfold quietMarks
  by_cases inside : column < slotSize pieces next <;> simp [inside]

/-- `env` with its quiet columns at 0. -/
def quiet (publicColumns : Nat) (pieces : List Piece) (next : Nat) (batches : List WitnessBatch)
    (env : Env) : Env :=
  fun column => if (quietMarks publicColumns pieces next batches).getD column false then 0
    else env column

/-- Each bit or centered slot lies below `next`, and no piece reads a quiet column. -/
def bitsSafe (publicColumns : Nat) (pieces : List Piece) (next : Nat)
    (batches : List WitnessBatch) : Bool :=
  let codes := slotCodes pieces next batches
  let quietColumns := quietMarks publicColumns pieces next batches
  (List.range (slotSize pieces next)).all (fun column =>
      decide (2 ≤ codes.getD column 3) || decide (column < next)) &&
    pieces.all fun piece => (pieceReads piece).all fun column => !quietColumns.getD column false

theorem forced_of_marked {pieces : List Piece} {size column : Nat}
    (marked : (forcedMarks pieces size).getD column false = true) :
    ∃ expression, Piece.constraint expression ∈ pieces ∧
      booleanColumn? expression = some column := by
  obtain ⟨piece, member, found⟩ := List.mem_filterMap.mp (mem_of_marked marked)
  cases piece with
  | constraint expression => exact ⟨expression, member, found⟩
  | permutation _ _ => cases found
  | ringProduct _ _ _ => cases found

/-! ### The honest assignment -/

/-- The honest assignment of a slot layout: the public input and every slot read from `env`. -/
def honestAssignment {width : Nat} (slots : List (LowNormAssignment.Slot width)) (env : Env) :
    Assignment F (ProductionAssignment.logicalWidth slots) :=
  ProductionAssignment.logicalAssignment (fun column => env column.val) slots
    (fun column => env column.val)

theorem kindOf_of_large {code : Nat} (large : 2 ≤ code) : SlotLayout.kindOf code = .field := by
  match code, large with
  | _ + 2, _ => rfl

variable {pieces : List Piece} {next width : Nat}

/-- The completed environment satisfies the package's relation at its slot layout, and every
coordinate of the honest assignment is small, when each bit or centered slot holds 0 or 1. -/
theorem rowsZero_of_valid (fits : Fits publicColumns pieces next width)
    (packed : packable publicColumns pieces = true)
    (valid : ∀ piece ∈ pieces, piece.Valid) (below : ∀ piece ∈ pieces, pieceBelow next piece = true)
    (publicBelow : publicColumns ≤ next) (batches : List WitnessBatch) {wide : Nat}
    (wideFits : ProductionAssignment.logicalWidth
      (SlotLayout.runSlots width (StepPackage.slotRuns publicColumns pieces next batches)) ≤ wide)
    (env : Env)
    (holds : ∀ piece ∈ pieces,
      ConstraintsHold env (piece.constraints (PilotData.canonicalConstraints ())))
    (oneValue : env 0 = 1)
    (publicNorm : ∀ column, column < publicColumns → centeredMagnitude (env column) < 2)
    (bits : ∀ column, publicColumns ≤ column → column < slotSize pieces next →
      (slotCodes pieces next batches).getD column 3 < 2 →
        completeEnv pieces next env column = 0 ∨ completeEnv pieces next env column = 1) :
    let slots := SlotLayout.runSlots width (StepPackage.slotRuns publicColumns pieces next batches)
    let assignment := WideLayout.extend (wide := wide)
      (honestAssignment slots (completeEnv pieces next env))
    (plan publicColumns pieces next width fits
        (WideLayout.widen (SlotLayout.layout slots) wideFits)).RowsZero assignment ∧
      ∀ column, centeredMagnitude (assignment column) < 2 := by
  intro slots
  let assignment := honestAssignment slots (completeEnv pieces next env)
  have publicEq : publicColumns = 270 := rfl
  have countEq : LogicalPlan.publicCount = 270 := rfl
  have slotEndLe : slotEnd pieces next ≤ width := by
    have := (fits.blocks packed).sboxColumns
    unfold slotEnd
    omega
  have nextLe := next_le_sboxStart pieces next
  have startLeEnd : sboxStart pieces next ≤ slotEnd pieces next := Nat.le_add_right _ _
  have completeLive : ∀ column, column < next → (completeEnv pieces next env) column = liveEnv pieces env column :=
    fun column columnBelow => (completeEnv_low (by omega)).trans
      (executePieces_below _ _ _ _ _ columnBelow)
  have deadBelow : ∀ column, deadIn (blocksOf pieces) column = true → column < next := by
    intro column dead
    unfold deadIn at dead
    obtain ⟨block, member, inside⟩ := List.any_eq_true.mp dead
    have pieceBelowNext := below _ (permutation_mem_of_block member)
    simp only [pieceBelow, Bool.and_eq_true, decide_eq_true_eq] at pieceBelowNext
    simp only [Block.dead, Bool.and_eq_true, decide_eq_true_eq] at inside
    omega
  have publicValue : ∀ column, column < publicColumns → (completeEnv pieces next env) column = env column := by
    intro column columnBelow
    rw [completeLive column (by omega)]
    simp [liveEnv, packable_public packed columnBelow]
  have represents : ∀ column, column < slotEnd pieces next →
      LogicalPlan.sourceEnv (SlotLayout.layout slots) assignment column = (completeEnv pieces next env) column := by
    intro column columnBelow
    have inside : column < width := by omega
    unfold LogicalPlan.sourceEnv
    rw [dite_eq_left inside]
    show (if bound : column < LogicalPlan.publicCount then
        SparseForm.singleton ⟨column, Nat.lt_of_lt_of_le bound (SlotLayout.publicFits slots)⟩ 1
      else match SlotLayout.slotIndex slots ⟨column, inside⟩ with
        | some index => RetainedSlot.form slots index
        | none => .empty).eval assignment = _
    by_cases publicRegion : column < LogicalPlan.publicCount
    · rw [dite_eq_left publicRegion, SparseForm.singleton_eval, one_mul]
      exact ProductionAssignment.logicalAssignment_publicColumn _ slots _ ⟨column, publicRegion⟩
    · rw [dite_eq_right publicRegion]
      cases found : SlotLayout.slotIndex slots ⟨column, inside⟩ with
      | some index =>
          simp only
          change (RetainedSlot.form slots index).eval (ProductionAssignment.logicalAssignment
            (fun column => completeEnv pieces next env column.val) slots
            (fun column => completeEnv pieces next env column.val)) = _
          rw [RetainedSlot.form_eval]
          have source : (slots.get index).source = ⟨column, inside⟩ := by
            have := List.find?_some found
            simpa using this
          rw [source]
      | none =>
          simp only [SparseForm.empty_eval]
          have code : (slotCodes pieces next batches).getD column 3 = 3 := by
            by_contra different
            obtain ⟨index, someIndex, _⟩ := slotIndex_of_mem (slots := slots)
              (column := ⟨column, inside⟩)
              ⟨⟨⟨column, inside⟩, SlotLayout.kindOf ((slotCodes pieces next batches).getD column 3)⟩,
                (mem_runSlots_runsOf _ _ _ _ _).mpr
                  ⟨by show publicColumns ≤ column; omega,
                    by show column < slotSize pieces next; simpa [slotSize, slotEnd] using columnBelow,
                    different, rfl⟩, rfl⟩
            rw [found] at someIndex
            cases someIndex
          have dead := dead_of_code pieces next batches
            (by simpa [slotSize, slotEnd] using columnBelow) code
          rw [completeLive column (deadBelow column dead)]
          simp [liveEnv, dead]
  have one : assignment (SlotLayout.layout slots).one = 1 :=
    (ProductionAssignment.logicalAssignment_publicColumn
      (fun column => completeEnv pieces next env column.val) slots
      (fun column => completeEnv pieces next env column.val) ⟨0, by decide⟩).trans
      ((publicValue 0 (by decide)).trans oneValue)
  refine ⟨RelationCompleteness.rowsZero_of_holds fits packed valid below env holds
    (WideLayout.widen (SlotLayout.layout slots) wideFits) (WideLayout.extend assignment)
    ((WideLayout.one_widen _ wideFits assignment).trans one)
    (fun column columnBelow => by
      rw [WideLayout.sourceEnv_widen]
      exact represents column columnBelow),
    WideLayout.extend_norm assignment fun column => ?_⟩
  apply ProductionAssignment.logicalAssignment_norm
  · intro current
    rw [publicValue current.val current.isLt]
    exact publicNorm current.val current.isLt
  · intro slot member
    change slot ∈ SlotLayout.runSlots width (StepPackage.slotRuns publicColumns pieces next batches)
      at member
    rw [slotRuns_eq] at member
    obtain ⟨publicLe, sourceBelow, _, kindEq⟩ := (mem_runSlots_runsOf _ _ _ _ _).mp member
    change LowNormSlot.Valid slot.kind ((completeEnv pieces next env) slot.source.val)
    by_cases large : 2 ≤ (slotCodes pieces next batches).getD slot.source.val 3
    · rw [kindEq, kindOf_of_large large]
      trivial
    · have bit := bits slot.source.val publicLe sourceBelow (by omega)
      have code : (slotCodes pieces next batches).getD slot.source.val 3 = 0 ∨
          (slotCodes pieces next batches).getD slot.source.val 3 = 1 := by omega
      rw [kindEq]
      rcases code with codeValue | codeValue <;> rw [codeValue] <;>
        rcases bit with value | value <;> rw [value]
      · exact Or.inl rfl
      · exact Or.inr rfl
      · show centeredMagnitude (0 : F) < 2
        decide
      · show centeredMagnitude (1 : F) < 2
        decide

/-- An honest step satisfies the package's relation at its slot layout. Set its quiet columns to
0: every piece still holds, because no piece reads them, and every bit or centered slot of the
completed environment holds 0 or 1, by its constraint `x * (x - 1)` or because it is quiet. -/
theorem rowsZero_of_holds (fits : Fits publicColumns pieces next width)
    (packed : packable publicColumns pieces = true)
    (valid : ∀ piece ∈ pieces, piece.Valid) (below : ∀ piece ∈ pieces, pieceBelow next piece = true)
    (publicBelow : publicColumns ≤ next) (batches : List WitnessBatch)
    (safe : bitsSafe publicColumns pieces next batches = true) {wide : Nat}
    (wideFits : ProductionAssignment.logicalWidth
      (SlotLayout.runSlots width (StepPackage.slotRuns publicColumns pieces next batches)) ≤ wide)
    (env : Env)
    (holds : ∀ piece ∈ pieces,
      ConstraintsHold env (piece.constraints (PilotData.canonicalConstraints ())))
    (oneValue : env 0 = 1)
    (publicNorm : ∀ column, column < publicColumns → centeredMagnitude (env column) < 2) :
    let slots := SlotLayout.runSlots width (StepPackage.slotRuns publicColumns pieces next batches)
    let assignment := WideLayout.extend (wide := wide) (honestAssignment slots
      (completeEnv pieces next (quiet publicColumns pieces next batches env)))
    (plan publicColumns pieces next width fits
        (WideLayout.widen (SlotLayout.layout slots) wideFits)).RowsZero assignment ∧
      ∀ column, centeredMagnitude (assignment column) < 2 := by
  simp only [bitsSafe, Bool.and_eq_true, List.all_eq_true] at safe
  obtain ⟨belowNext, unread⟩ := safe
  have keep : ∀ column,
      (quietMarks publicColumns pieces next batches).getD column false = false →
        quiet publicColumns pieces next batches env column = env column := by
    intro column notQuiet
    simp [quiet, notQuiet]
  have publicKeep : ∀ column, column < publicColumns →
      quiet publicColumns pieces next batches env column = env column := by
    intro column columnBelow
    apply keep
    rw [quietMarks_getD]
    simp only [Bool.and_eq_false_iff, decide_eq_false_iff_not]
    left; left; right
    omega
  have quietHolds : ∀ piece ∈ pieces, ConstraintsHold (quiet publicColumns pieces next batches env)
      (piece.constraints (PilotData.canonicalConstraints ())) := by
    intro piece member
    apply holds_of_reads piece (valid piece member) _ (holds piece member)
    intro column read
    apply keep
    simpa using unread piece member column read
  refine rowsZero_of_valid fits packed valid below publicBelow batches wideFits _ quietHolds
    ((publicKeep 0 (by decide)).trans oneValue)
    (fun column columnBelow => (publicKeep column columnBelow) ▸ publicNorm column columnBelow) ?_
  intro column publicLe inside small
  have columnBelow : column < next := by
    have atColumn := belowNext column (List.mem_range.mpr inside)
    simp only [Bool.or_eq_true, decide_eq_true_eq] at atColumn
    omega
  rw [completeEnv_low (by have := next_le_sboxStart pieces next; omega),
    executePieces_below _ _ _ _ _ columnBelow]
  unfold liveEnv
  split
  · exact Or.inl rfl
  · by_cases marked : (forcedMarks pieces (slotSize pieces next)).getD column false = true
    · obtain ⟨expression, member, found⟩ := forced_of_marked marked
      exact bit_of_booleanColumn found (quietHolds _ member expression List.mem_cons_self)
    · left
      have isQuiet : (quietMarks publicColumns pieces next batches).getD column false = true := by
        have notMarked : (forcedMarks pieces (slotSize pieces next)).getD column false = false := by
          simpa using marked
        rw [quietMarks_getD, decide_eq_true inside, decide_eq_true publicLe, decide_eq_true small,
          notMarked]
        rfl
      simp [quiet, isQuiet]

end NightstreamFPrime.Export.OptionB.SlotCompleteness
