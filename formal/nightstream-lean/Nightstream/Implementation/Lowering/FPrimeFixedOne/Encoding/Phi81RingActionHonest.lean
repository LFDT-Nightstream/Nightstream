import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RingActionSemantics

/-!
Contract: honest completion and freshness for the Lean-owned Phi81
ring-action rows.

Owns:
- the exact product-cell witness;
- preservation of every visible carried combination;
- satisfaction of every product and reduced-output equation;
- exact honest completeness of the owned row program.

Does not own: call-frame placement, activation, codecs, selected-NIFS
composition, Rust, or generated artifacts.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RingAction

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.SuperNeo.Concrete

/-- One exact product-cell write. -/
structure ProductEntry where
  column : ColumnId
  value : F

/-- The semantic value assigned to one product cell.  Natural indices are
used only to mirror the exact allocation order; out-of-range branches are
unreachable for `productEntries`. -/
def productEntry
    {count : Nat}
    (frame : Frame count)
    (assignment : ColumnId → F)
    (source left right : Nat) : ProductEntry where
  column := frame.productColumn source left right
  value :=
    if sourceLt : source < count then
      ringFCoeff
          (decoded assignment (frame.challenges ⟨source, sourceLt⟩))
          left *
        ringFCoeff
          (decoded assignment (frame.values ⟨source, sourceLt⟩))
          right
    else
      0

@[simp] theorem productEntry_column
    {count : Nat}
    (frame : Frame count)
    (assignment : ColumnId → F)
    (source left right : Nat) :
    (productEntry frame assignment source left right).column =
      frame.productColumn source left right :=
  rfl

/-- Exact row-major product witness. -/
def productEntries
    {count : Nat}
    (frame : Frame count)
    (assignment : ColumnId → F) : List ProductEntry :=
  (List.range count).flatMap fun source =>
    (List.range ringDegree).flatMap fun left =>
      (List.range ringDegree).map fun right =>
        productEntry frame assignment source left right

private theorem sum_map_const
    {α : Type} (items : List α) (value : Nat) :
    (items.map fun _ => value).sum = value * items.length := by
  induction items with
  | nil =>
      rfl
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, List.sum_cons, List.length_cons,
        inductionHypothesis, Nat.mul_succ]
      omega

@[simp] theorem productEntries_length
    {count : Nat}
    (frame : Frame count)
    (assignment : ColumnId → F) :
    (productEntries frame assignment).length = productWidth count := by
  simp only [productEntries, List.length_flatMap, List.length_map,
    List.length_range]
  rw [show
      (List.map
        (fun _ =>
          (List.map (fun _ => ringDegree)
            (List.range ringDegree)).sum)
        (List.range count)).sum =
        (List.map (fun _ => ringDegree * ringDegree)
          (List.range count)).sum by
      apply congrArg List.sum
      apply List.map_congr_left
      intro source sourceMember
      exact sum_map_const (List.range ringDegree) ringDegree]
  rw [sum_map_const]
  simp [productWidth, Nat.mul_comm, Nat.mul_left_comm]

theorem productEntries_columns
    {count : Nat}
    (frame : Frame count)
    (assignment : ColumnId → F) :
    (productEntries frame assignment).map ProductEntry.column =
      productIds frame := by
  simp [productEntries, productIds, productEntry_column,
    List.map_flatMap, Function.comp_def]

/-- Write exact entries, with earlier entries taking precedence. -/
def writeEntries
    (assignment : ColumnId → F) :
    List ProductEntry → ColumnId → F
  | [], column => assignment column
  | entry :: rest, column =>
      if column = entry.column then entry.value
      else writeEntries assignment rest column

theorem writeEntries_of_not_mem
    (assignment : ColumnId → F)
    (entries : List ProductEntry)
    (column : ColumnId)
    (notMember :
      column ∉ entries.map ProductEntry.column) :
    writeEntries assignment entries column = assignment column := by
  induction entries with
  | nil =>
      rfl
  | cons head tail inductionHypothesis =>
      have different : column ≠ head.column := by
        intro equal
        apply notMember
        simp [equal]
      have tailNotMember :
          column ∉ tail.map ProductEntry.column := by
        intro member
        exact notMember (by simp [member])
      simp [writeEntries, different,
        inductionHypothesis tailNotMember]

theorem writeEntries_exact
    (assignment : ColumnId → F)
    (entries : List ProductEntry)
    (nodup : (entries.map ProductEntry.column).Nodup)
    (entry : ProductEntry)
    (member : entry ∈ entries) :
    writeEntries assignment entries entry.column = entry.value := by
  induction entries with
  | nil =>
      simp at member
  | cons head tail inductionHypothesis =>
      have split :
          head.column ∉ tail.map ProductEntry.column ∧
            (tail.map ProductEntry.column).Nodup := by
        simpa only [List.map_cons, List.nodup_cons] using nodup
      rcases List.mem_cons.mp member with equal | tailMember
      · subst entry
        simp [writeEntries]
      · have different : entry.column ≠ head.column := by
          intro columnEqual
          apply split.1
          apply List.mem_map.2
          exact ⟨entry, tailMember, columnEqual⟩
        rw [writeEntries]
        simp only [if_neg different]
        exact inductionHypothesis split.2 tailMember

/-- Complete exactly the product-cell allocation. -/
def honestAssignment
    {count : Nat}
    (frame : Frame count)
    (assignment : ColumnId → F) : ColumnId → F :=
  writeEntries assignment (productEntries frame assignment)

/-- A carried combination is fresh when it reads no product cell allocated by
this occurrence. -/
def Fresh
    {count : Nat}
    (frame : Frame count)
    (combination : LinearCombination) : Prop :=
  ∀ term ∈ combination, term.column ∉ productIds frame

/-- Exact freshness contract required by honest completion. -/
structure WellFormed
    {count : Nat}
    (frame : Frame count) : Prop where
  productsNodup : (productIds frame).Nodup
  oneFresh : frame.one ∉ productIds frame
  challengesFresh :
    ∀ source : Fin count,
      ∀ lane : Fin ringDegree,
        Fresh frame (frame.challenges source lane)
  valuesFresh :
    ∀ source : Fin count,
      ∀ lane : Fin ringDegree,
        Fresh frame (frame.values source lane)
  outputFresh :
    ∀ lane : Fin ringDegree,
      Fresh frame (frame.output lane)

theorem honestAssignment_preserves_column
    {count : Nat}
    (frame : Frame count)
    (assignment : ColumnId → F)
    (column : ColumnId)
    (fresh : column ∉ productIds frame) :
    honestAssignment frame assignment column = assignment column := by
  unfold honestAssignment
  apply writeEntries_of_not_mem
  rw [productEntries_columns]
  exact fresh

theorem honestAssignment_preserves_combination
    {count : Nat}
    (frame : Frame count)
    (assignment : ColumnId → F)
    (combination : LinearCombination)
    (fresh : Fresh frame combination) :
    combination.eval (honestAssignment frame assignment) =
      combination.eval assignment := by
  induction combination with
  | nil =>
      rfl
  | cons term tail inductionHypothesis =>
      rw [LinearCombination.eval, LinearCombination.eval,
        honestAssignment_preserves_column frame assignment term.column
          (fresh term (by simp))]
      rw [inductionHypothesis (fun item member =>
        fresh item (by simp [member]))]

theorem honestAssignment_preserves_decoded
    {count : Nat}
    (frame : Frame count)
    (assignment : ColumnId → F)
    (value : CarriedRing)
    (fresh :
      ∀ lane : Fin ringDegree, Fresh frame (value lane)) :
    decoded (honestAssignment frame assignment) value =
      decoded assignment value := by
  funext lane
  exact honestAssignment_preserves_combination frame assignment
    (value lane) (fresh lane)

/-- Every product cell contains its exact semantic product after completion. -/
theorem honestAssignment_product
    {count : Nat}
    (frame : Frame count)
    (assignment : ColumnId → F)
    (wellFormed : WellFormed frame)
    (source : Fin count)
    (left right : Fin ringDegree) :
    honestAssignment frame assignment
        (frame.productColumn source.val left.val right.val) =
      decoded assignment (frame.challenges source) left *
        decoded assignment (frame.values source) right := by
  let entry :=
    productEntry frame assignment source.val left.val right.val
  have member :
      entry ∈ productEntries frame assignment := by
    unfold productEntries
    apply List.mem_flatMap.2
    refine ⟨source.val, List.mem_range.mpr source.isLt, ?_⟩
    apply List.mem_flatMap.2
    refine ⟨left.val, List.mem_range.mpr left.isLt, ?_⟩
    apply List.mem_map.2
    exact ⟨right.val, List.mem_range.mpr right.isLt, rfl⟩
  have exact :=
    writeEntries_exact assignment (productEntries frame assignment)
      (by
        rw [productEntries_columns]
        exact wellFormed.productsNodup)
      entry member
  simpa [honestAssignment, entry, productEntry, source.isLt,
    ringFCoeff, left.isLt, right.isLt] using exact

private theorem rawSatisfies_of_forall
    {source : List Row}
    {assignment : ColumnId → F}
    (holds : ∀ row, row ∈ source → row.Holds assignment) :
    RawSatisfies source assignment := by
  induction source with
  | nil =>
      trivial
  | cons head tail inductionHypothesis =>
      exact
        ⟨holds head (by simp),
          inductionHypothesis (fun row member =>
            holds row (by simp [member]))⟩

theorem honestAssignment_constantOne
    {count : Nat}
    (frame : Frame count)
    (assignment : ColumnId → F)
    (wellFormed : WellFormed frame)
    (constantOne : assignment frame.one = 1) :
    honestAssignment frame assignment frame.one = 1 := by
  rw [honestAssignment_preserves_column frame assignment frame.one
    wellFormed.oneFresh]
  exact constantOne

/-- Every schoolbook product row holds under the honest completion. -/
theorem productRow_holds_honest
    {count : Nat}
    (frame : Frame count)
    (assignment : ColumnId → F)
    (wellFormed : WellFormed frame)
    (source : Fin count)
    (left right : Fin ringDegree) :
    (productRow frame source.val left.val right.val).Holds
      (honestAssignment frame assignment) := by
  simp only [productRow, dif_pos source.isLt, dif_pos left.isLt,
    dif_pos right.isLt, Row.Holds, Goldilocks.singleton,
    LinearCombination.eval, Fin.one_mul, Fin.add_zero]
  rw [honestAssignment_preserves_combination frame assignment
      (frame.challenges source left)
      (wellFormed.challengesFresh source left),
    honestAssignment_preserves_combination frame assignment
      (frame.values source right)
      (wellFormed.valuesFresh source right),
    honestAssignment_product frame assignment wellFormed source left right]
  rfl

/-- Every reduced output equation holds when the visible output is the
semantic result. -/
theorem outputRow_holds_honest
    {count : Nat}
    (frame : Frame count)
    (assignment : ColumnId → F)
    (wellFormed : WellFormed frame)
    (constantOne : assignment frame.one = 1)
    (semantic :
      decoded assignment frame.output =
        combine
          (fun source => decoded assignment (frame.challenges source))
          (fun source => decoded assignment (frame.values source)))
    (output : Fin ringDegree) :
    (outputRow frame output.val).Holds
      (honestAssignment frame assignment) := by
  let completed := honestAssignment frame assignment
  have challengesPreserved :
      (fun source =>
          decoded completed (frame.challenges source)) =
        (fun source =>
          decoded assignment (frame.challenges source)) := by
    funext source
    exact honestAssignment_preserves_decoded frame assignment
      (frame.challenges source) (wellFormed.challengesFresh source)
  have valuesPreserved :
      (fun source =>
          decoded completed (frame.values source)) =
        (fun source =>
          decoded assignment (frame.values source)) := by
    funext source
    exact honestAssignment_preserves_decoded frame assignment
      (frame.values source) (wellFormed.valuesFresh source)
  have outputPreserved :
      decoded completed frame.output =
        decoded assignment frame.output :=
    honestAssignment_preserves_decoded frame assignment frame.output
      wellFormed.outputFresh
  have products :
      ∀ source : Fin count,
        ∀ left right : Fin ringDegree,
          completed
              (frame.productColumn source.val left.val right.val) =
            decoded completed (frame.challenges source) left *
              decoded completed (frame.values source) right := by
    intro source left right
    rw [congrFun (congrFun challengesPreserved source) left,
      congrFun (congrFun valuesPreserved source) right]
    exact honestAssignment_product frame assignment wellFormed
      source left right
  have reducedExact :=
    reducedCombination_eval frame completed products output
  have expected :
      combine
          (fun source => decoded completed (frame.challenges source))
          (fun source => decoded completed (frame.values source))
          output =
        decoded completed frame.output output := by
    rw [challengesPreserved, valuesPreserved, outputPreserved, semantic]
  simp only [outputRow, dif_pos output.isLt, Row.Holds,
    Goldilocks.singleton, LinearCombination.eval,
    honestAssignment_constantOne frame assignment wellFormed constantOne,
    Fin.mul_one, Fin.add_zero]
  rw [reducedExact]
  exact expected

/-- The complete raw program holds under the honest product-cell witness. -/
theorem rawRows_honest
    {count : Nat}
    (frame : Frame count)
    (assignment : ColumnId → F)
    (wellFormed : WellFormed frame)
    (constantOne : assignment frame.one = 1)
    (semantic :
      decoded assignment frame.output =
        combine
          (fun source => decoded assignment (frame.challenges source))
          (fun source => decoded assignment (frame.values source))) :
    RawSatisfies (rawRows frame)
      (honestAssignment frame assignment) := by
  apply rawSatisfies_of_forall
  intro row member
  rcases List.mem_append.mp member with productMember | outputMember
  · unfold productRows at productMember
    rcases List.mem_flatMap.1 productMember with
      ⟨source, sourceMember, sourceRows⟩
    rcases List.mem_flatMap.1 sourceRows with
      ⟨left, leftMember, leftRows⟩
    rcases List.mem_map.1 leftRows with
      ⟨right, rightMember, rowExact⟩
    subst row
    exact productRow_holds_honest frame assignment wellFormed
      ⟨source, List.mem_range.mp sourceMember⟩
      ⟨left, List.mem_range.mp leftMember⟩
      ⟨right, List.mem_range.mp rightMember⟩
  · unfold outputRows at outputMember
    rcases List.mem_map.1 outputMember with
      ⟨output, outputInRange, rowExact⟩
    rw [← rowExact]
    exact outputRow_holds_honest frame assignment wellFormed
      constantOne semantic
      ⟨output, List.mem_range.mp outputInRange⟩

/-- **Headline honest completeness.** Every semantically correct visible
output extends, only on the declared product cells, to a satisfying assignment
for all owned rows. -/
theorem rows_honest
    {count : Nat}
    (frame : Frame count)
    (assignment : ColumnId → F)
    (wellFormed : WellFormed frame)
    (constantOne : assignment frame.one = 1)
    (semantic :
      decoded assignment frame.output =
        combine
          (fun source => decoded assignment (frame.challenges source))
          (fun source => decoded assignment (frame.values source))) :
    Satisfies (rows frame) (honestAssignment frame assignment) := by
  exact (satisfies_rows_iff frame
    (honestAssignment frame assignment)).mpr
      (rawRows_honest frame assignment wellFormed constantOne semantic)

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RingAction
