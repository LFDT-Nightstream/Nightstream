import Nightstream.Implementation.Lowering.Goldilocks.CallRecipe

/-!
Contract: artifact-independent physical recipes for the non-call primitives of
the direct Goldilocks lowering.

Owns:
- exact multi-coordinate literal pin rows;
- the one-row active Boolean assertion over `boolCodec`;
- the two branch-activation rows;
- one mux row per joined coordinate;
- local soundness, honest completeness, and inactive/selected behavior.

Does not own: program traversal, block compilation, call implementations,
generated artifacts, Rust layouts, or whole-verifier acceptance.

Every returned row has a caller-visible `PhysicalOwner` and ordinal.  No
additional rows are implicit.

Emits constraints:
- literal: exactly the selected codec width;
- Boolean assertion: exactly one row;
- branch activation: exactly two rows;
- mux: exactly one row per coordinate of the common layout.
-/

namespace Nightstream.Implementation.Lowering.Goldilocks

open Nightstream.Implementation.Lowering.Typed

universe u

/-! ## One-coordinate Boolean decoding -/

theorem boolCodec_decode_true_iff (coordinate : Field) :
    boolCodec.decode [coordinate] = some true ↔ coordinate = 1 := by
  constructor
  · intro decoded
    have encoded :=
      (boolCodec.encode_decode [coordinate] true decoded).2
    have trueEncoding : boolCodec.encode true = [(1 : Field)] := rfl
    rw [trueEncoding] at encoded
    have coordinates : coordinate = 1 ∧ True := by
      simpa only [List.cons.injEq] using encoded.symm
    exact coordinates.1
  · intro coordinateOne
    subst coordinate
    exact boolCodec.decode_encode true True.intro

theorem boolCodec_decode_false_iff (coordinate : Field) :
    boolCodec.decode [coordinate] = some false ↔ coordinate = 0 := by
  constructor
  · intro decoded
    have encoded :=
      (boolCodec.encode_decode [coordinate] false decoded).2
    have falseEncoding : boolCodec.encode false = [(0 : Field)] := rfl
    rw [falseEncoding] at encoded
    have coordinates : coordinate = 0 ∧ True := by
      simpa only [List.cons.injEq] using encoded.symm
    exact coordinates.1
  · intro coordinateZero
    subst coordinate
    exact boolCodec.decode_encode false True.intro

/-! ## Multi-coordinate literal pins -/

private def pinRowsFrom
    (owner : PhysicalOwner)
    (one : ColumnId) :
    Nat -> List OwnedColumn -> List Field -> List OwnedRow
  | _, [], _ => []
  | _, _ :: _, [] => []
  | ordinal, column :: columns, value :: values =>
      { id := { owner := owner, ordinal := ordinal }
        row := (CanonicalRow.pin one column.id value).row } ::
        pinRowsFrom owner one (ordinal + 1) columns values

private theorem pinRowsFrom_length_of_equal
    (owner : PhysicalOwner)
    (one : ColumnId)
    (ordinal : Nat)
    (columns : List OwnedColumn)
    (values : List Field)
    (lengthEqual : columns.length = values.length) :
    (pinRowsFrom owner one ordinal columns values).length =
      columns.length := by
  induction columns generalizing ordinal values with
  | nil =>
      cases values <;> simp [pinRowsFrom]
  | cons column columns inductionHypothesis =>
      cases values with
      | nil =>
          simp at lengthEqual
      | cons value values =>
          simp only [List.length_cons, Nat.succ.injEq] at lengthEqual
          simp only [pinRowsFrom, List.length_cons]
          rw [inductionHypothesis (ordinal := ordinal + 1)
            (values := values) lengthEqual]

private theorem pinRowsFrom_owned
    (owner : PhysicalOwner)
    (one : ColumnId)
    (ordinal : Nat)
    (columns : List OwnedColumn)
    (values : List Field) :
    ∀ row, row ∈ pinRowsFrom owner one ordinal columns values ->
      row.id.owner = owner := by
  induction columns generalizing ordinal values with
  | nil =>
      intro row member
      simp [pinRowsFrom] at member
  | cons column columns inductionHypothesis =>
      cases values with
      | nil =>
          intro row member
          simp [pinRowsFrom] at member
      | cons value values =>
          intro row member
          simp only [pinRowsFrom, List.mem_cons] at member
          rcases member with equal | tailMember
          · subst row
            rfl
          · exact inductionHypothesis
              (ordinal := ordinal + 1) values row tailMember

private theorem pinRowsFrom_supported
    (owner : PhysicalOwner)
    (one : ColumnId)
    (ordinal : Nat)
    (columns : List OwnedColumn)
    (values : List Field) :
    ∀ row, row ∈ pinRowsFrom owner one ordinal columns values ->
      ∀ column, column ∈ row.columnIds ->
        column ∈ [one] ++ columns.map (fun item => item.id) := by
  induction columns generalizing ordinal values with
  | nil =>
      intro row member
      simp [pinRowsFrom] at member
  | cons output outputs inductionHypothesis =>
      cases values with
      | nil =>
          intro row member
          simp [pinRowsFrom] at member
      | cons value values =>
          intro row member column columnMember
          simp only [pinRowsFrom, List.mem_cons] at member
          rcases member with equal | tailMember
          · subst row
            simp [OwnedRow.columnIds, CanonicalRow.row,
              Row.columnIds, singleton] at columnMember
            rcases columnMember with equal | equal
            · subst column
              apply List.mem_append.mpr
              exact Or.inr (by simp)
            · subst column
              apply List.mem_append.mpr
              exact Or.inl (by simp)
          · have supported :=
              inductionHypothesis
                (ordinal := ordinal + 1) values
                row tailMember column columnMember
            rcases List.mem_append.mp supported with
              oneMember | tailOutput
            · apply List.mem_append.mpr
              exact Or.inl oneMember
            · apply List.mem_append.mpr
              exact Or.inr (by
                simp only [List.map_cons]
                exact List.mem_cons_of_mem output.id tailOutput)

private theorem pinRowsFrom_row_ids
    (owner : PhysicalOwner)
    (one : ColumnId)
    (ordinal : Nat)
    (columns : List OwnedColumn)
    (values : List Field)
    (lengthEqual : columns.length = values.length) :
    (pinRowsFrom owner one ordinal columns values).map
        (fun row => row.id) =
      (List.range' ordinal columns.length).map
        (fun index => { owner := owner, ordinal := index }) := by
  induction columns generalizing ordinal values with
  | nil =>
      cases values <;> simp [pinRowsFrom]
  | cons column columns inductionHypothesis =>
      cases values with
      | nil =>
          simp at lengthEqual
      | cons value values =>
          simp only [List.length_cons, Nat.succ.injEq] at lengthEqual
          simp only [pinRowsFrom, List.map_cons, List.length_cons,
            List.range'_succ,
            inductionHypothesis
              (ordinal := ordinal + 1)
              (values := values)
              lengthEqual]

private theorem pinRowsFrom_satisfies_iff
    (owner : PhysicalOwner)
    (one : ColumnId)
    (ordinal : Nat)
    (columns : List OwnedColumn)
    (values : List Field)
    (assignment : ColumnId -> Field)
    (constantOne : assignment one = 1)
    (lengthEqual : columns.length = values.length) :
    Satisfies (pinRowsFrom owner one ordinal columns values) assignment ↔
      columns.map (fun column => assignment column.id) = values := by
  induction columns generalizing ordinal values with
  | nil =>
      cases values with
      | nil => simp [pinRowsFrom]
      | cons value values =>
          simp at lengthEqual
  | cons column columns inductionHypothesis =>
      cases values with
      | nil =>
          simp at lengthEqual
      | cons value values =>
          simp only [List.length_cons, Nat.succ.injEq] at lengthEqual
          constructor
          · intro holds
            have headHolds :
                (CanonicalRow.pin one column.id value).row.Holds assignment :=
              holds.1
            have tailHolds :
                Satisfies
                  (pinRowsFrom owner one (ordinal + 1) columns values)
                  assignment :=
              holds.2
            have headEqual : assignment column.id = value :=
              (CanonicalRow.pin_iff
                assignment one column.id value constantOne).mp headHolds
            have tailEqual :
                columns.map (fun item => assignment item.id) = values :=
              (inductionHypothesis
                (ordinal := ordinal + 1)
                (values := values) lengthEqual).mp tailHolds
            simpa only [List.map_cons, List.cons.injEq] using
              And.intro headEqual tailEqual
          · intro coordinates
            have split :
                assignment column.id = value ∧
                  columns.map (fun item => assignment item.id) = values := by
              simpa only [List.map_cons, List.cons.injEq] using coordinates
            exact ⟨
              (CanonicalRow.pin_iff
                assignment one column.id value constantOne).mpr split.1,
              (inductionHypothesis
                (ordinal := ordinal + 1)
                (values := values) lengthEqual).mpr split.2⟩

/-- Exact physical data for one typed literal occurrence. -/
structure LiteralPinRecipe
    {α : Type u}
    (codec : Codec α)
    (layout : Layout) where
  owner : PhysicalOwner
  firstOrdinal : Nat
  one : ColumnId
  output : ColumnBundle layout
  value : α
  widthAgrees : codec.width = layout.owners.length

namespace LiteralPinRecipe

private theorem coordinate_lengths_equal
    {α : Type u}
    {codec : Codec α}
    {layout : Layout}
    (recipe : LiteralPinRecipe codec layout) :
    recipe.output.columns.length =
      (codec.encode recipe.value).length := by
  calc
    recipe.output.columns.length = layout.owners.length :=
      recipe.output.length_eq
    _ = codec.width := recipe.widthAgrees.symm
    _ = (codec.encode recipe.value).length :=
      (codec.encode_length recipe.value).symm

/-- Complete ordered literal row list. -/
def rows
    {α : Type u}
    {codec : Codec α}
    {layout : Layout}
    (recipe : LiteralPinRecipe codec layout) : List OwnedRow :=
  pinRowsFrom recipe.owner recipe.one recipe.firstOrdinal
    recipe.output.columns (codec.encode recipe.value)

theorem row_count
    {α : Type u}
    {codec : Codec α}
    {layout : Layout}
    (recipe : LiteralPinRecipe codec layout) :
    recipe.rows.length = codec.width := by
  calc
    recipe.rows.length = recipe.output.columns.length :=
      pinRowsFrom_length_of_equal
        recipe.owner recipe.one recipe.firstOrdinal
        recipe.output.columns (codec.encode recipe.value)
        recipe.coordinate_lengths_equal
    _ = layout.owners.length := recipe.output.length_eq
    _ = codec.width := recipe.widthAgrees.symm

/-- Every literal row is owned by the literal occurrence. -/
theorem rows_owned
    {α : Type u}
    {codec : Codec α}
    {layout : Layout}
    (recipe : LiteralPinRecipe codec layout) :
    ∀ row, row ∈ recipe.rows -> row.id.owner = recipe.owner :=
  pinRowsFrom_owned recipe.owner recipe.one recipe.firstOrdinal
    recipe.output.columns (codec.encode recipe.value)

/-- Literal rows mention only verifier one and the exact output bundle. -/
theorem rows_supported
    {α : Type u}
    {codec : Codec α}
    {layout : Layout}
    (recipe : LiteralPinRecipe codec layout) :
    ∀ row, row ∈ recipe.rows ->
      ∀ column, column ∈ row.columnIds ->
        column ∈ [recipe.one] ++ recipe.output.ids := by
  simpa only [ColumnBundle.ids] using
    pinRowsFrom_supported recipe.owner recipe.one recipe.firstOrdinal
      recipe.output.columns (codec.encode recipe.value)

/-- Literal row identities are the consecutive occurrence-local ordinals. -/
theorem row_ids_nodup
    {α : Type u}
    {codec : Codec α}
    {layout : Layout}
    (recipe : LiteralPinRecipe codec layout) :
    (recipe.rows.map fun row => row.id).Nodup := by
  unfold rows
  rw [pinRowsFrom_row_ids
    recipe.owner recipe.one recipe.firstOrdinal
    recipe.output.columns (codec.encode recipe.value)
    recipe.coordinate_lengths_equal]
  exact
    (List.nodup_range' :
      (List.range' recipe.firstOrdinal
        recipe.output.columns.length).Nodup).map
    (fun index : Nat =>
      ({ owner := recipe.owner, ordinal := index } : RowId)) (by
      intro first second different equal
      apply different
      exact congrArg RowId.ordinal equal)

/-- Satisfying all pins recovers the exact canonical coordinate string. -/
theorem coordinates_of_satisfies
    {α : Type u}
    {codec : Codec α}
    {layout : Layout}
    (recipe : LiteralPinRecipe codec layout)
    (assignment : ColumnId -> Field)
    (constantOne : assignment recipe.one = 1)
    (holds : Satisfies recipe.rows assignment) :
    recipe.output.values assignment = codec.encode recipe.value := by
  have coordinates :=
    (pinRowsFrom_satisfies_iff
      recipe.owner recipe.one recipe.firstOrdinal
      recipe.output.columns (codec.encode recipe.value)
      assignment constantOne recipe.coordinate_lengths_equal).mp holds
  simpa only [ColumnBundle.values] using coordinates

/-- An admissible literal decoded from satisfying rows is the static value. -/
theorem decode_of_satisfies
    {α : Type u}
    {codec : Codec α}
    {layout : Layout}
    (recipe : LiteralPinRecipe codec layout)
    (assignment : ColumnId -> Field)
    (constantOne : assignment recipe.one = 1)
    (admissible : codec.Admissible recipe.value)
    (holds : Satisfies recipe.rows assignment) :
    codec.decode (recipe.output.values assignment) = some recipe.value := by
  rw [recipe.coordinates_of_satisfies assignment constantOne holds]
  exact codec.decode_encode recipe.value admissible

/-- Canonically encoded coordinates satisfy every literal pin row. -/
theorem satisfies_of_coordinates
    {α : Type u}
    {codec : Codec α}
    {layout : Layout}
    (recipe : LiteralPinRecipe codec layout)
    (assignment : ColumnId -> Field)
    (constantOne : assignment recipe.one = 1)
    (coordinates :
      recipe.output.values assignment = codec.encode recipe.value) :
    Satisfies recipe.rows assignment := by
  apply (pinRowsFrom_satisfies_iff
    recipe.owner recipe.one recipe.firstOrdinal
    recipe.output.columns (codec.encode recipe.value)
    assignment constantOne recipe.coordinate_lengths_equal).mpr
  simpa only [ColumnBundle.values] using coordinates

/-- Successful decoding at the literal's declared value is sufficient for all
pin rows; codec exactness reconstructs the coordinates first. -/
theorem satisfies_of_decode
    {α : Type u}
    {codec : Codec α}
    {layout : Layout}
    (recipe : LiteralPinRecipe codec layout)
    (assignment : ColumnId -> Field)
    (constantOne : assignment recipe.one = 1)
    (decoded :
      codec.decode (recipe.output.values assignment) =
        some recipe.value) :
    Satisfies recipe.rows assignment := by
  have encoded :=
    (codec.encode_decode
      (recipe.output.values assignment) recipe.value decoded).2
  exact recipe.satisfies_of_coordinates
    assignment constantOne encoded.symm

end LiteralPinRecipe

/-! ## Active Boolean assertion -/

/-- One active assertion of one canonical Boolean coordinate. -/
structure BoolAssertRecipe where
  owner : PhysicalOwner
  ordinal : Nat
  one : ColumnId
  active : ColumnId
  condition : ColumnId

namespace BoolAssertRecipe

/-- Complete ordered assertion row list. -/
def rows (recipe : BoolAssertRecipe) : List OwnedRow :=
  [{ id := { owner := recipe.owner, ordinal := recipe.ordinal }
     row :=
       (CanonicalRow.gatedAssert
          recipe.one recipe.active recipe.condition).row }]

@[simp] theorem row_count (recipe : BoolAssertRecipe) :
    recipe.rows.length = 1 :=
  rfl

/-- Under active execution, row satisfaction is exactly canonical decoding as
Boolean true. -/
theorem active_iff_decode_true
    (recipe : BoolAssertRecipe)
    (laws : FieldLaws)
    (assignment : ColumnId -> Field)
    (constantOne : assignment recipe.one = 1)
    (activeOne : assignment recipe.active = 1) :
    Satisfies recipe.rows assignment ↔
      boolCodec.decode [assignment recipe.condition] = some true := by
  constructor
  · intro holds
    have conditionOne :
        assignment recipe.condition = 1 :=
      (CanonicalRow.gatedAssert_iff_of_active
        laws assignment recipe.one recipe.active recipe.condition
        constantOne activeOne).mp holds.1
    exact (boolCodec_decode_true_iff
      (assignment recipe.condition)).2 conditionOne
  · intro decoded
    have conditionOne :
        assignment recipe.condition = 1 :=
      (boolCodec_decode_true_iff
        (assignment recipe.condition)).1 decoded
    exact ⟨
      (CanonicalRow.gatedAssert_iff_of_active
        laws assignment recipe.one recipe.active recipe.condition
        constantOne activeOne).mpr conditionOne,
      True.intro⟩

/-- An inactive assertion imposes no condition value. -/
theorem inactive_complete
    (recipe : BoolAssertRecipe)
    (assignment : ColumnId -> Field)
    (activeZero : assignment recipe.active = 0) :
    Satisfies recipe.rows assignment := by
  exact ⟨
    CanonicalRow.gatedAssert_complete_of_inactive
      assignment recipe.one recipe.active recipe.condition activeZero,
    True.intro⟩

end BoolAssertRecipe

/-! ## Branch activations -/

/-- The two rows that derive mutually selected child activations from one
parent activation and one canonical Boolean selector. -/
structure BranchActivationRecipe where
  path : OwnerPath
  one : ColumnId
  active : ColumnId
  selector : ColumnId
  onTrue : ColumnId
  onFalse : ColumnId

namespace BranchActivationRecipe

/-- Complete ordered activation row list.  Each branch owns its own ordinal
zero; the owner includes the branch-selection bit. -/
def rows (recipe : BranchActivationRecipe) : List OwnedRow :=
  [{ id :=
       { owner := .branchActivation recipe.path true
         ordinal := 0 }
     row :=
       (CanonicalRow.activateTrue
          recipe.onTrue recipe.active recipe.selector).row },
   { id :=
       { owner := .branchActivation recipe.path false
         ordinal := 0 }
     row :=
       (CanonicalRow.activateFalse
          recipe.one recipe.onFalse recipe.active recipe.selector).row }]

@[simp] theorem row_count (recipe : BranchActivationRecipe) :
    recipe.rows.length = 2 :=
  rfl

/-- Exact algebraic meaning of both activation rows. -/
theorem satisfies_iff
    (recipe : BranchActivationRecipe)
    (assignment : ColumnId -> Field)
    (constantOne : assignment recipe.one = 1) :
    Satisfies recipe.rows assignment ↔
      assignment recipe.active * assignment recipe.selector =
          assignment recipe.onTrue ∧
        assignment recipe.active *
            (1 - assignment recipe.selector) =
          assignment recipe.onFalse := by
  constructor
  · intro holds
    exact ⟨
      (CanonicalRow.activateTrue_iff
        assignment recipe.onTrue recipe.active recipe.selector).mp holds.1,
      (CanonicalRow.activateFalse_iff
        assignment recipe.one recipe.onFalse recipe.active recipe.selector
        constantOne).mp holds.2.1⟩
  · intro equations
    exact ⟨
      (CanonicalRow.activateTrue_iff
        assignment recipe.onTrue recipe.active recipe.selector).mpr
          equations.1,
      (CanonicalRow.activateFalse_iff
        assignment recipe.one recipe.onFalse recipe.active recipe.selector
        constantOne).mpr equations.2,
      True.intro⟩

theorem selected_true_sound
    (recipe : BranchActivationRecipe)
    (assignment : ColumnId -> Field)
    (constantOne : assignment recipe.one = 1)
    (selectorDecoded :
      boolCodec.decode [assignment recipe.selector] = some true)
    (holds : Satisfies recipe.rows assignment) :
    assignment recipe.onTrue = assignment recipe.active ∧
      assignment recipe.onFalse = 0 := by
  have selectorOne :
      assignment recipe.selector = 1 :=
    (boolCodec_decode_true_iff
      (assignment recipe.selector)).1 selectorDecoded
  have equations := (recipe.satisfies_iff assignment constantOne).mp holds
  constructor
  · simpa only [selectorOne, Fin.mul_one] using equations.1.symm
  · have zeroDifference : (1 : Field) - 1 = 0 :=
      Lean.Grind.AddCommGroup.sub_eq_zero_iff.mpr rfl
    rw [selectorOne, zeroDifference, Fin.mul_zero] at equations
    exact equations.2.symm

theorem selected_true_complete
    (recipe : BranchActivationRecipe)
    (assignment : ColumnId -> Field)
    (constantOne : assignment recipe.one = 1)
    (selectorDecoded :
      boolCodec.decode [assignment recipe.selector] = some true)
    (selected :
      assignment recipe.onTrue = assignment recipe.active ∧
        assignment recipe.onFalse = 0) :
    Satisfies recipe.rows assignment := by
  have selectorOne :
      assignment recipe.selector = 1 :=
    (boolCodec_decode_true_iff
      (assignment recipe.selector)).1 selectorDecoded
  apply (recipe.satisfies_iff assignment constantOne).mpr
  constructor
  · rw [selectorOne, Fin.mul_one, selected.1]
  · have zeroDifference : (1 : Field) - 1 = 0 :=
      Lean.Grind.AddCommGroup.sub_eq_zero_iff.mpr rfl
    rw [selectorOne, zeroDifference, Fin.mul_zero, selected.2]

theorem selected_false_sound
    (recipe : BranchActivationRecipe)
    (assignment : ColumnId -> Field)
    (constantOne : assignment recipe.one = 1)
    (selectorDecoded :
      boolCodec.decode [assignment recipe.selector] = some false)
    (holds : Satisfies recipe.rows assignment) :
    assignment recipe.onTrue = 0 ∧
      assignment recipe.onFalse = assignment recipe.active := by
  have selectorZero :
      assignment recipe.selector = 0 :=
    (boolCodec_decode_false_iff
      (assignment recipe.selector)).1 selectorDecoded
  have equations := (recipe.satisfies_iff assignment constantOne).mp holds
  constructor
  · rw [selectorZero, Fin.mul_zero] at equations
    exact equations.1.symm
  · have oneSubZero : (1 : Field) - 0 = 1 := by
      simpa only [Fin.sub_eq_add_neg, Lean.Grind.AddCommGroup.neg_zero,
        Fin.add_zero]
    rw [selectorZero, oneSubZero, Fin.mul_one] at equations
    exact equations.2.symm

theorem selected_false_complete
    (recipe : BranchActivationRecipe)
    (assignment : ColumnId -> Field)
    (constantOne : assignment recipe.one = 1)
    (selectorDecoded :
      boolCodec.decode [assignment recipe.selector] = some false)
    (selected :
      assignment recipe.onTrue = 0 ∧
        assignment recipe.onFalse = assignment recipe.active) :
    Satisfies recipe.rows assignment := by
  have selectorZero :
      assignment recipe.selector = 0 :=
    (boolCodec_decode_false_iff
      (assignment recipe.selector)).1 selectorDecoded
  apply (recipe.satisfies_iff assignment constantOne).mpr
  constructor
  · rw [selectorZero, Fin.mul_zero, selected.1]
  · have oneSubZero : (1 : Field) - 0 = 1 := by
      simpa only [Fin.sub_eq_add_neg, Lean.Grind.AddCommGroup.neg_zero,
        Fin.add_zero]
    rw [selectorZero, oneSubZero, Fin.mul_one, selected.2]

/-- An inactive parent forces both child activations to zero, independently of
the selector coordinate. -/
theorem inactive_sound
    (recipe : BranchActivationRecipe)
    (assignment : ColumnId -> Field)
    (constantOne : assignment recipe.one = 1)
    (activeZero : assignment recipe.active = 0)
    (holds : Satisfies recipe.rows assignment) :
    assignment recipe.onTrue = 0 ∧
      assignment recipe.onFalse = 0 := by
  have equations := (recipe.satisfies_iff assignment constantOne).mp holds
  have trueZero := equations.1
  have falseZero := equations.2
  rw [activeZero, Fin.zero_mul] at trueZero falseZero
  exact ⟨trueZero.symm, falseZero.symm⟩

theorem inactive_complete
    (recipe : BranchActivationRecipe)
    (assignment : ColumnId -> Field)
    (constantOne : assignment recipe.one = 1)
    (activeZero : assignment recipe.active = 0)
    (inactive :
      assignment recipe.onTrue = 0 ∧
        assignment recipe.onFalse = 0) :
    Satisfies recipe.rows assignment := by
  apply (recipe.satisfies_iff assignment constantOne).mpr
  constructor
  · rw [activeZero, Fin.zero_mul, inactive.1]
  · rw [activeZero, Fin.zero_mul, inactive.2]

end BranchActivationRecipe

/-! ## Coordinate-wise branch joins -/

/-- Structural row emitter for aligned joined, true-arm, and false-arm
coordinate lists.  It is public so finite normal-form certificates can prove
their selected rows are exactly this emitter's output. -/
def muxRowsFrom
    (owner : PhysicalOwner)
    (selector : ColumnId) :
    Nat -> List OwnedColumn -> List OwnedColumn -> List OwnedColumn ->
      List OwnedRow
  | ordinal,
      joined :: joinedTail,
      onTrue :: trueTail,
      onFalse :: falseTail =>
      { id := { owner := owner, ordinal := ordinal }
        row :=
          (CanonicalRow.mux
            joined.id selector onTrue.id onFalse.id).row } ::
        muxRowsFrom owner selector (ordinal + 1)
          joinedTail trueTail falseTail
  | _, _, _, _ => []

private theorem muxRowsFrom_length
    (owner : PhysicalOwner)
    (selector : ColumnId)
    (ordinal : Nat)
    (joined onTrue onFalse : List OwnedColumn)
    (trueLength : joined.length = onTrue.length)
    (falseLength : joined.length = onFalse.length) :
    (muxRowsFrom owner selector ordinal joined onTrue onFalse).length =
      joined.length := by
  induction joined generalizing ordinal onTrue onFalse with
  | nil =>
      cases onTrue <;> cases onFalse <;> simp_all [muxRowsFrom]
  | cons joined joinedTail inductionHypothesis =>
      cases onTrue with
      | nil =>
          simp at trueLength
      | cons onTrue trueTail =>
          cases onFalse with
          | nil =>
              simp at falseLength
          | cons onFalse falseTail =>
              simp only [List.length_cons, Nat.succ.injEq] at trueLength falseLength
              simp only [muxRowsFrom, List.length_cons]
              rw [inductionHypothesis
                (ordinal := ordinal + 1)
                (onTrue := trueTail)
                (onFalse := falseTail)
                trueLength falseLength]

private theorem muxRowsFrom_owned
    (owner : PhysicalOwner)
    (selector : ColumnId)
    (ordinal : Nat)
    (joined onTrue onFalse : List OwnedColumn) :
    ∀ row,
      row ∈ muxRowsFrom owner selector ordinal joined onTrue onFalse ->
        row.id.owner = owner := by
  induction joined generalizing ordinal onTrue onFalse with
  | nil =>
      intro row member
      simp [muxRowsFrom] at member
  | cons joined joinedTail inductionHypothesis =>
      cases onTrue with
      | nil =>
          intro row member
          simp [muxRowsFrom] at member
      | cons onTrue trueTail =>
          cases onFalse with
          | nil =>
              intro row member
              simp [muxRowsFrom] at member
          | cons onFalse falseTail =>
              intro row member
              simp only [muxRowsFrom, List.mem_cons] at member
              rcases member with equal | tailMember
              · subst row
                rfl
              · exact inductionHypothesis
                  (ordinal := ordinal + 1)
                  (onTrue := trueTail)
                  (onFalse := falseTail)
                  row tailMember

private theorem muxRowsFrom_supported
    (owner : PhysicalOwner)
    (selector : ColumnId)
    (ordinal : Nat)
    (joined onTrue onFalse : List OwnedColumn) :
    ∀ row,
      row ∈ muxRowsFrom owner selector ordinal joined onTrue onFalse ->
        ∀ column, column ∈ row.columnIds ->
          column ∈
            [selector] ++
              joined.map (fun item => item.id) ++
              onTrue.map (fun item => item.id) ++
              onFalse.map (fun item => item.id) := by
  induction joined generalizing ordinal onTrue onFalse with
  | nil =>
      intro row member
      simp [muxRowsFrom] at member
  | cons joined joinedTail inductionHypothesis =>
      cases onTrue with
      | nil =>
          intro row member
          simp [muxRowsFrom] at member
      | cons onTrue trueTail =>
          cases onFalse with
          | nil =>
              intro row member
              simp [muxRowsFrom] at member
          | cons onFalse falseTail =>
              intro row member column columnMember
              simp only [muxRowsFrom, List.mem_cons] at member
              rcases member with equal | tailMember
              · subst row
                simp [OwnedRow.columnIds, CanonicalRow.row,
                  Row.columnIds, singleton, difference] at columnMember ⊢
                rcases columnMember with equal | equal | equal | equal | equal
                all_goals simp_all
              · have supported :=
                  inductionHypothesis
                    (ordinal := ordinal + 1)
                    (onTrue := trueTail)
                    (onFalse := falseTail)
                    row tailMember column columnMember
                simp only [List.map_cons, List.mem_append, List.mem_cons,
                  List.mem_singleton, List.not_mem_nil, or_false] at supported ⊢
                grind

private theorem muxRowsFrom_row_ids
    (owner : PhysicalOwner)
    (selector : ColumnId)
    (ordinal : Nat)
    (joined onTrue onFalse : List OwnedColumn)
    (trueLength : joined.length = onTrue.length)
    (falseLength : joined.length = onFalse.length) :
    (muxRowsFrom owner selector ordinal joined onTrue onFalse).map
        (fun row => row.id) =
      (List.range' ordinal joined.length).map
        (fun index => { owner := owner, ordinal := index }) := by
  induction joined generalizing ordinal onTrue onFalse with
  | nil =>
      cases onTrue <;> cases onFalse <;> simp_all [muxRowsFrom]
  | cons joined joinedTail inductionHypothesis =>
      cases onTrue with
      | nil =>
          simp at trueLength
      | cons onTrue trueTail =>
          cases onFalse with
          | nil =>
              simp at falseLength
          | cons onFalse falseTail =>
              simp only [List.length_cons, Nat.succ.injEq] at trueLength falseLength
              simp only [muxRowsFrom, List.map_cons, List.length_cons,
                List.range'_succ,
                inductionHypothesis
                  (ordinal := ordinal + 1)
                  (onTrue := trueTail)
                  (onFalse := falseTail)
                  trueLength falseLength]

private theorem muxRowsFrom_selects_true
    (owner : PhysicalOwner)
    (selector : ColumnId)
    (ordinal : Nat)
    (joined onTrue onFalse : List OwnedColumn)
    (assignment : ColumnId -> Field)
    (selectorOne : assignment selector = 1)
    (trueLength : joined.length = onTrue.length)
    (falseLength : joined.length = onFalse.length)
    (holds :
      Satisfies
        (muxRowsFrom owner selector ordinal joined onTrue onFalse)
        assignment) :
    joined.map (fun column => assignment column.id) =
      onTrue.map (fun column => assignment column.id) := by
  induction joined generalizing ordinal onTrue onFalse with
  | nil =>
      cases onTrue <;> cases onFalse <;> simp_all [muxRowsFrom]
  | cons joined joinedTail inductionHypothesis =>
      cases onTrue with
      | nil =>
          simp at trueLength
      | cons onTrue trueTail =>
          cases onFalse with
          | nil =>
              simp at falseLength
          | cons onFalse falseTail =>
              simp only [List.length_cons, Nat.succ.injEq] at trueLength falseLength
              have headEqual :
                  assignment joined.id = assignment onTrue.id :=
                CanonicalRow.mux_selects_true
                  assignment joined.id selector onTrue.id onFalse.id
                  selectorOne holds.1
              have tailEqual :
                  joinedTail.map (fun column => assignment column.id) =
                    trueTail.map (fun column => assignment column.id) :=
                inductionHypothesis
                  (ordinal := ordinal + 1)
                  (onTrue := trueTail)
                  (onFalse := falseTail)
                  trueLength falseLength holds.2
              simpa only [List.map_cons, List.cons.injEq] using
                And.intro headEqual tailEqual

private theorem muxRowsFrom_selects_false
    (owner : PhysicalOwner)
    (selector : ColumnId)
    (ordinal : Nat)
    (joined onTrue onFalse : List OwnedColumn)
    (assignment : ColumnId -> Field)
    (selectorZero : assignment selector = 0)
    (trueLength : joined.length = onTrue.length)
    (falseLength : joined.length = onFalse.length)
    (holds :
      Satisfies
        (muxRowsFrom owner selector ordinal joined onTrue onFalse)
        assignment) :
    joined.map (fun column => assignment column.id) =
      onFalse.map (fun column => assignment column.id) := by
  induction joined generalizing ordinal onTrue onFalse with
  | nil =>
      cases onTrue <;> cases onFalse <;> simp_all [muxRowsFrom]
  | cons joined joinedTail inductionHypothesis =>
      cases onTrue with
      | nil =>
          simp at trueLength
      | cons onTrue trueTail =>
          cases onFalse with
          | nil =>
              simp at falseLength
          | cons onFalse falseTail =>
              simp only [List.length_cons, Nat.succ.injEq] at trueLength falseLength
              have headEqual :
                  assignment joined.id = assignment onFalse.id :=
                CanonicalRow.mux_selects_false
                  assignment joined.id selector onTrue.id onFalse.id
                  selectorZero holds.1
              have tailEqual :
                  joinedTail.map (fun column => assignment column.id) =
                    falseTail.map (fun column => assignment column.id) :=
                inductionHypothesis
                  (ordinal := ordinal + 1)
                  (onTrue := trueTail)
                  (onFalse := falseTail)
                  trueLength falseLength holds.2
              simpa only [List.map_cons, List.cons.injEq] using
                And.intro headEqual tailEqual

private theorem muxRow_complete_true
    (assignment : ColumnId -> Field)
    (joined selector onTrue onFalse : ColumnId)
    (selectorOne : assignment selector = 1)
    (joinedEqual : assignment joined = assignment onTrue) :
    (CanonicalRow.mux joined selector onTrue onFalse).row.Holds
      assignment := by
  apply (CanonicalRow.mux_iff
    assignment joined selector onTrue onFalse).mpr
  rw [selectorOne, Fin.one_mul, joinedEqual]

private theorem muxRow_complete_false
    (assignment : ColumnId -> Field)
    (joined selector onTrue onFalse : ColumnId)
    (selectorZero : assignment selector = 0)
    (joinedEqual : assignment joined = assignment onFalse) :
    (CanonicalRow.mux joined selector onTrue onFalse).row.Holds
      assignment := by
  apply (CanonicalRow.mux_iff
    assignment joined selector onTrue onFalse).mpr
  rw [selectorZero, Fin.zero_mul, joinedEqual]
  exact
    (Lean.Grind.AddCommGroup.sub_eq_zero_iff.mpr rfl).symm

private theorem muxRowsFrom_complete_true
    (owner : PhysicalOwner)
    (selector : ColumnId)
    (ordinal : Nat)
    (joined onTrue onFalse : List OwnedColumn)
    (assignment : ColumnId -> Field)
    (selectorOne : assignment selector = 1)
    (trueLength : joined.length = onTrue.length)
    (falseLength : joined.length = onFalse.length)
    (joinedEqual :
      joined.map (fun column => assignment column.id) =
        onTrue.map (fun column => assignment column.id)) :
    Satisfies
      (muxRowsFrom owner selector ordinal joined onTrue onFalse)
      assignment := by
  induction joined generalizing ordinal onTrue onFalse with
  | nil =>
      cases onTrue <;> cases onFalse <;> simp_all [muxRowsFrom]
  | cons joined joinedTail inductionHypothesis =>
      cases onTrue with
      | nil =>
          simp at trueLength
      | cons onTrue trueTail =>
          cases onFalse with
          | nil =>
              simp at falseLength
          | cons onFalse falseTail =>
              simp only [List.length_cons, Nat.succ.injEq] at trueLength falseLength
              have split :
                  assignment joined.id = assignment onTrue.id ∧
                    joinedTail.map (fun column => assignment column.id) =
                      trueTail.map (fun column => assignment column.id) := by
                simpa only [List.map_cons, List.cons.injEq] using joinedEqual
              exact ⟨
                muxRow_complete_true assignment
                  joined.id selector onTrue.id onFalse.id
                  selectorOne split.1,
                inductionHypothesis
                  (ordinal := ordinal + 1)
                  (onTrue := trueTail)
                  (onFalse := falseTail)
                  trueLength falseLength split.2⟩

private theorem muxRowsFrom_complete_false
    (owner : PhysicalOwner)
    (selector : ColumnId)
    (ordinal : Nat)
    (joined onTrue onFalse : List OwnedColumn)
    (assignment : ColumnId -> Field)
    (selectorZero : assignment selector = 0)
    (trueLength : joined.length = onTrue.length)
    (falseLength : joined.length = onFalse.length)
    (joinedEqual :
      joined.map (fun column => assignment column.id) =
        onFalse.map (fun column => assignment column.id)) :
    Satisfies
      (muxRowsFrom owner selector ordinal joined onTrue onFalse)
      assignment := by
  induction joined generalizing ordinal onTrue onFalse with
  | nil =>
      cases onTrue <;> cases onFalse <;> simp_all [muxRowsFrom]
  | cons joined joinedTail inductionHypothesis =>
      cases onTrue with
      | nil =>
          simp at trueLength
      | cons onTrue trueTail =>
          cases onFalse with
          | nil =>
              simp at falseLength
          | cons onFalse falseTail =>
              simp only [List.length_cons, Nat.succ.injEq] at trueLength falseLength
              have split :
                  assignment joined.id = assignment onFalse.id ∧
                    joinedTail.map (fun column => assignment column.id) =
                      falseTail.map (fun column => assignment column.id) := by
                simpa only [List.map_cons, List.cons.injEq] using joinedEqual
              exact ⟨
                muxRow_complete_false assignment
                  joined.id selector onTrue.id onFalse.id
                  selectorZero split.1,
                inductionHypothesis
                  (ordinal := ordinal + 1)
                  (onTrue := trueTail)
                  (onFalse := falseTail)
                  trueLength falseLength split.2⟩

/-- Exact physical data for one coordinate-wise branch join. -/
structure MuxRecipe (layout : Layout) where
  owner : PhysicalOwner
  firstOrdinal : Nat
  selector : ColumnId
  joined : ColumnBundle layout
  onTrue : ColumnBundle layout
  onFalse : ColumnBundle layout

namespace MuxRecipe

private theorem true_lengths_equal
    {layout : Layout}
    (recipe : MuxRecipe layout) :
    recipe.joined.columns.length = recipe.onTrue.columns.length := by
  rw [recipe.joined.length_eq, recipe.onTrue.length_eq]

private theorem false_lengths_equal
    {layout : Layout}
    (recipe : MuxRecipe layout) :
    recipe.joined.columns.length = recipe.onFalse.columns.length := by
  rw [recipe.joined.length_eq, recipe.onFalse.length_eq]

/-- Complete ordered mux row list. -/
def rows
    {layout : Layout}
    (recipe : MuxRecipe layout) : List OwnedRow :=
  muxRowsFrom recipe.owner recipe.selector recipe.firstOrdinal
    recipe.joined.columns recipe.onTrue.columns recipe.onFalse.columns

theorem row_count
    {layout : Layout}
    (recipe : MuxRecipe layout) :
    recipe.rows.length = layout.owners.length := by
  calc
    recipe.rows.length = recipe.joined.columns.length :=
      muxRowsFrom_length
        recipe.owner recipe.selector recipe.firstOrdinal
        recipe.joined.columns recipe.onTrue.columns recipe.onFalse.columns
        recipe.true_lengths_equal recipe.false_lengths_equal
    _ = layout.owners.length := recipe.joined.length_eq

/-- Every mux row is owned by the branch-join occurrence. -/
theorem rows_owned
    {layout : Layout}
    (recipe : MuxRecipe layout) :
    ∀ row, row ∈ recipe.rows -> row.id.owner = recipe.owner :=
  muxRowsFrom_owned recipe.owner recipe.selector recipe.firstOrdinal
    recipe.joined.columns recipe.onTrue.columns recipe.onFalse.columns

/-- Mux rows mention only the selector, the freshly joined coordinates, and
the exact two arm bundles. -/
theorem rows_supported
    {layout : Layout}
    (recipe : MuxRecipe layout) :
    ∀ row, row ∈ recipe.rows ->
      ∀ column, column ∈ row.columnIds ->
        column ∈
          [recipe.selector] ++ recipe.joined.ids ++
            recipe.onTrue.ids ++ recipe.onFalse.ids := by
  simpa only [ColumnBundle.ids] using
    muxRowsFrom_supported recipe.owner recipe.selector recipe.firstOrdinal
      recipe.joined.columns recipe.onTrue.columns recipe.onFalse.columns

/-- Mux row identities are the consecutive occurrence-local ordinals. -/
theorem row_ids_nodup
    {layout : Layout}
    (recipe : MuxRecipe layout) :
    (recipe.rows.map fun row => row.id).Nodup := by
  unfold rows
  rw [muxRowsFrom_row_ids
    recipe.owner recipe.selector recipe.firstOrdinal
    recipe.joined.columns recipe.onTrue.columns recipe.onFalse.columns
    recipe.true_lengths_equal recipe.false_lengths_equal]
  exact
    (List.nodup_range' :
      (List.range' recipe.firstOrdinal
        recipe.joined.columns.length).Nodup).map
    (fun index : Nat =>
      ({ owner := recipe.owner, ordinal := index } : RowId)) (by
      intro first second different equal
      apply different
      exact congrArg RowId.ordinal equal)

theorem selected_true_sound
    {layout : Layout}
    (recipe : MuxRecipe layout)
    (assignment : ColumnId -> Field)
    (selectorDecoded :
      boolCodec.decode [assignment recipe.selector] = some true)
    (holds : Satisfies recipe.rows assignment) :
    recipe.joined.values assignment =
      recipe.onTrue.values assignment := by
  have selectorOne :
      assignment recipe.selector = 1 :=
    (boolCodec_decode_true_iff
      (assignment recipe.selector)).1 selectorDecoded
  simpa only [ColumnBundle.values] using
    muxRowsFrom_selects_true
      recipe.owner recipe.selector recipe.firstOrdinal
      recipe.joined.columns recipe.onTrue.columns recipe.onFalse.columns
      assignment selectorOne recipe.true_lengths_equal
      recipe.false_lengths_equal holds

theorem selected_true_complete
    {layout : Layout}
    (recipe : MuxRecipe layout)
    (assignment : ColumnId -> Field)
    (selectorDecoded :
      boolCodec.decode [assignment recipe.selector] = some true)
    (selected :
      recipe.joined.values assignment =
        recipe.onTrue.values assignment) :
    Satisfies recipe.rows assignment := by
  have selectorOne :
      assignment recipe.selector = 1 :=
    (boolCodec_decode_true_iff
      (assignment recipe.selector)).1 selectorDecoded
  apply muxRowsFrom_complete_true
    recipe.owner recipe.selector recipe.firstOrdinal
    recipe.joined.columns recipe.onTrue.columns recipe.onFalse.columns
    assignment selectorOne recipe.true_lengths_equal
    recipe.false_lengths_equal
  simpa only [ColumnBundle.values] using selected

theorem selected_false_sound
    {layout : Layout}
    (recipe : MuxRecipe layout)
    (assignment : ColumnId -> Field)
    (selectorDecoded :
      boolCodec.decode [assignment recipe.selector] = some false)
    (holds : Satisfies recipe.rows assignment) :
    recipe.joined.values assignment =
      recipe.onFalse.values assignment := by
  have selectorZero :
      assignment recipe.selector = 0 :=
    (boolCodec_decode_false_iff
      (assignment recipe.selector)).1 selectorDecoded
  simpa only [ColumnBundle.values] using
    muxRowsFrom_selects_false
      recipe.owner recipe.selector recipe.firstOrdinal
      recipe.joined.columns recipe.onTrue.columns recipe.onFalse.columns
      assignment selectorZero recipe.true_lengths_equal
      recipe.false_lengths_equal holds

theorem selected_false_complete
    {layout : Layout}
    (recipe : MuxRecipe layout)
    (assignment : ColumnId -> Field)
    (selectorDecoded :
      boolCodec.decode [assignment recipe.selector] = some false)
    (selected :
      recipe.joined.values assignment =
        recipe.onFalse.values assignment) :
    Satisfies recipe.rows assignment := by
  have selectorZero :
      assignment recipe.selector = 0 :=
    (boolCodec_decode_false_iff
      (assignment recipe.selector)).1 selectorDecoded
  apply muxRowsFrom_complete_false
    recipe.owner recipe.selector recipe.firstOrdinal
    recipe.joined.columns recipe.onTrue.columns recipe.onFalse.columns
    assignment selectorZero recipe.true_lengths_equal
    recipe.false_lengths_equal
  simpa only [ColumnBundle.values] using selected

end MuxRecipe

end Nightstream.Implementation.Lowering.Goldilocks
