import Nightstream.Implementation.Lowering.Goldilocks.PrimitiveRecipes

/-!
Assignment-preservation lemmas used to compose exact branch receipts.
This leaf owns no branch semantics and emits no rows or columns.
-/

namespace Nightstream.Implementation.Lowering.Goldilocks

open Nightstream.Implementation.Lowering.Typed

universe u

def rowColumns (row : Row) : List ColumnId :=
  (row.a.map Term.column) ++
    (row.b.map Term.column) ++
    (row.c.map Term.column)

def rowsColumns (rows : List OwnedRow) : List ColumnId :=
  rows.flatMap fun row => rowColumns row.row

theorem idsDisjoint_symm {left right : List ColumnId}
    (disjoint : IdsDisjoint left right) :
    IdsDisjoint right left := by
  intro id rightMember leftMember
  exact disjoint id leftMember rightMember

theorem agreesOn_refl (ids : List ColumnId)
    (assignment : ColumnId -> Field) :
    AgreesOn ids assignment assignment := by
  intro _ _
  rfl

theorem agreesOn_trans {ids : List ColumnId}
    {first second third : ColumnId -> Field}
    (firstSecond : AgreesOn ids first second)
    (secondThird : AgreesOn ids second third) :
    AgreesOn ids first third := by
  intro id member
  rw [secondThird id member, firstSecond id member]

theorem agreesOn_of_subset {small large : List ColumnId}
    {before after : ColumnId -> Field}
    (subset : ∀ id, id ∈ small -> id ∈ large)
    (agrees : AgreesOn large before after) :
    AgreesOn small before after := by
  intro id member
  exact agrees id (subset id member)

theorem agreesOn_of_changesOnly
    {changed preserved : List ColumnId}
    {before after : ColumnId -> Field}
    (disjoint : IdsDisjoint changed preserved)
    (changes : ChangesOnly changed before after) :
    AgreesOn preserved before after := by
  intro id preservedMember
  apply changes id
  exact fun changedMember =>
    disjoint id changedMember preservedMember

private theorem linearCombination_eval_of_agrees
    (combination : LinearCombination)
    (before after : ColumnId -> Field)
    (agrees :
      AgreesOn (combination.map Term.column) before after) :
    combination.eval after = combination.eval before := by
  induction combination with
  | nil => rfl
  | cons term tail inductionHypothesis =>
      have headAgrees : after term.column = before term.column :=
        agrees term.column (by simp)
      have tailAgrees :
          AgreesOn (tail.map Term.column) before after := by
        intro id member
        exact agrees id (by simp [member])
      simp only [LinearCombination.eval_cons, headAgrees,
        inductionHypothesis tailAgrees]

private theorem row_holds_of_agrees
    (row : Row)
    (before after : ColumnId -> Field)
    (agrees : AgreesOn (rowColumns row) before after)
    (holds : row.Holds before) :
    row.Holds after := by
  have aAgrees :
      AgreesOn (row.a.map Term.column) before after := by
    apply agreesOn_of_subset _ agrees
    intro id member
    simp [rowColumns, member]
  have bAgrees :
      AgreesOn (row.b.map Term.column) before after := by
    apply agreesOn_of_subset _ agrees
    intro id member
    simp [rowColumns, member]
  have cAgrees :
      AgreesOn (row.c.map Term.column) before after := by
    apply agreesOn_of_subset _ agrees
    intro id member
    simp [rowColumns, member]
  unfold Row.Holds at holds ⊢
  rw [linearCombination_eval_of_agrees row.a before after aAgrees,
    linearCombination_eval_of_agrees row.b before after bAgrees,
    linearCombination_eval_of_agrees row.c before after cAgrees]
  exact holds

theorem satisfies_of_agrees
    (rows : List OwnedRow)
    (before after : ColumnId -> Field)
    (agrees : AgreesOn (rowsColumns rows) before after)
    (holds : Satisfies rows before) :
    Satisfies rows after := by
  induction rows with
  | nil => trivial
  | cons row tail inductionHypothesis =>
      have rowAgrees :
          AgreesOn (rowColumns row.row) before after := by
        apply agreesOn_of_subset _ agrees
        intro id member
        simp only [rowsColumns, List.flatMap_cons, List.mem_append]
        exact Or.inl member
      have tailAgrees :
          AgreesOn (rowsColumns tail) before after := by
        apply agreesOn_of_subset _ agrees
        intro id member
        simp only [rowsColumns, List.flatMap_cons, List.mem_append]
        exact Or.inr member
      exact ⟨row_holds_of_agrees row.row before after rowAgrees holds.1,
        inductionHypothesis tailAgrees holds.2⟩

theorem ColumnBundle.values_eq_of_agrees
    {layout : Layout}
    (bundle : ColumnBundle layout)
    (before after : ColumnId -> Field)
    (agrees : AgreesOn bundle.ids before after) :
    bundle.values after = bundle.values before := by
  change (bundle.columns.map fun column => after column.id) =
    bundle.columns.map fun column => before column.id
  change AgreesOn
    (bundle.columns.map fun column => column.id) before after at agrees
  generalize bundle.columns = columns at agrees ⊢
  revert agrees
  induction columns with
  | nil => intro; rfl
  | cons column tail inductionHypothesis =>
      intro agrees
      simp only [List.map_cons]
      have headAgrees : after column.id = before column.id :=
        agrees column.id (by simp)
      have tailAgrees :
          AgreesOn (tail.map fun item => item.id) before after := by
        intro id member
        exact agrees id (by simp [member])
      rw [headAgrees, inductionHypothesis tailAgrees]

theorem ColumnBundle.decodes_of_agrees
    {types : TypeSystem.{u}}
    (family : Family types)
    (kind : types.Kind)
    {layout : Layout}
    (bundle : ColumnBundle layout)
    (before after : ColumnId -> Field)
    (value : types.Value kind)
    (agrees : AgreesOn bundle.ids before after)
    (decoded : bundle.Decodes family kind before value) :
    bundle.Decodes family kind after value := by
  unfold ColumnBundle.Decodes at decoded ⊢
  rw [bundle.values_eq_of_agrees before after agrees]
  exact decoded

theorem ColumnBundle.encodes_of_agrees
    {types : TypeSystem.{u}}
    (family : Family types)
    (kind : types.Kind)
    {layout : Layout}
    (bundle : ColumnBundle layout)
    (before after : ColumnId -> Field)
    (value : types.Value kind)
    (agrees : AgreesOn bundle.ids before after)
    (encoded : bundle.Encodes family kind before value) :
    bundle.Encodes family kind after value := by
  rcases encoded with ⟨admissible, coordinates⟩
  exact ⟨admissible,
    (bundle.values_eq_of_agrees before after agrees).trans coordinates⟩

theorem SchemaBundles.encodes_of_agrees
    {types : TypeSystem.{u}}
    (family : Family types)
    (before after : ColumnId -> Field)
    {schema : Schema types}
    (bundles : SchemaBundles schema)
    (values : Schema.Values types schema)
    (agrees : AgreesOn bundles.ids before after)
    (encoded : bundles.Encodes family before values) :
    bundles.Encodes family after values := by
  induction bundles with
  | nil =>
      cases values
      trivial
  | @cons port tail head rest inductionHypothesis =>
      cases values with
      | cons value values =>
          rw [SchemaBundles.ids_cons head rest] at agrees
          have headAgrees :
              AgreesOn head.ids before after := by
            exact agreesOn_of_subset
              (fun _ member => List.mem_append_left _ member) agrees
          have tailAgrees :
              AgreesOn rest.ids before after := by
            exact agreesOn_of_subset
              (fun _ member => List.mem_append_right _ member) agrees
          exact ⟨
            head.encodes_of_agrees family port.kind before after value
              headAgrees encoded.1,
            inductionHypothesis values tailAgrees encoded.2⟩

theorem RefBundles.encodes_of_agrees
    {types : TypeSystem.{u}}
    (family : Family types)
    (before after : ColumnId -> Field)
    {context : Schema types}
    {sorts : List types.Kind}
    {references : Refs types context sorts}
    (bundles : RefBundles references)
    (values : HVec types.Value sorts)
    (agrees : AgreesOn bundles.ids before after)
    (encoded : bundles.Encodes family before values) :
    bundles.Encodes family after values := by
  induction bundles with
  | nil =>
      cases values
      trivial
  | @cons kind sorts reference references head tail inductionHypothesis =>
      cases values with
      | cons value values =>
          rw [RefBundles.ids_cons head tail] at agrees
          have headAgrees :
              AgreesOn head.ids before after := by
            exact agreesOn_of_subset
              (fun _ member => List.mem_append_left _ member) agrees
          have tailAgrees :
              AgreesOn tail.ids before after := by
            exact agreesOn_of_subset
              (fun _ member => List.mem_append_right _ member) agrees
          exact ⟨
            head.encodes_of_agrees family kind before after value
              headAgrees encoded.1,
            inductionHypothesis values tailAgrees encoded.2⟩

end Nightstream.Implementation.Lowering.Goldilocks
