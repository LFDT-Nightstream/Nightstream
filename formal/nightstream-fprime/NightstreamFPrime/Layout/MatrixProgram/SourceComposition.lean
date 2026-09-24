import NightstreamFPrime.Layout.MatrixProgram.SourceProjection

/-! Compose affine source ranges without adding a wire constructor. The
first projection must have unique source ownership, so composition cannot
turn an ambiguous source into an accepted one. -/

namespace NightstreamFPrime.Layout.MatrixProgram

def SourceProjectionRange.compose (left right : SourceProjectionRange) : SourceProjectionRange :=
  let first := max left.sourceStart right.packageStart
  let last := min (left.sourceStart + left.count) (right.packageStart + right.count)
  ⟨left.packageStart + (first - left.sourceStart),
    right.sourceStart + (first - right.packageStart), last - first⟩

theorem SourceProjectionRange.compose_column (left right : SourceProjectionRange) (source : Nat) :
    (left.compose right).column? source = (left.column? source).bind right.column? := by
  unfold compose column?
  by_cases a : left.packageStart + (max left.sourceStart right.packageStart - left.sourceStart) ≤ source <;>
    by_cases b : source - (left.packageStart + (max left.sourceStart right.packageStart - left.sourceStart)) <
      min (left.sourceStart + left.count) (right.packageStart + right.count) - max left.sourceStart right.packageStart <;>
    by_cases c : left.packageStart ≤ source <;>
    by_cases d : source - left.packageStart < left.count <;>
    by_cases e : right.packageStart ≤ left.sourceStart + (source - left.packageStart) <;>
    by_cases f : left.sourceStart + (source - left.packageStart) - right.packageStart < right.count <;>
    simp only [a, b, c, d, e, f, ite_true, ite_false, Option.bind_some, Option.bind_none]
  all_goals first | omega | exact congrArg some (by omega)

def SourceProjection.compose (left right : SourceProjection) : SourceProjection :=
  match left, right with
  | .identity, _ => right
  | _, .identity => left
  | .mapped first, .mapped second =>
    .mapped (first.flatMap fun a => second.map a.compose)

def SourceProjection.Unique : SourceProjection → Prop
  | .identity => True
  | .mapped ranges => ∀ source, (ranges.filterMap (fun range => range.column? source)).length ≤ 1

private theorem composed_matches (left right : List SourceProjectionRange) (source : Nat) :
    ((left.flatMap fun a => right.map a.compose).filterMap (fun range => range.column? source)) =
      (left.filterMap (fun range => range.column? source)).flatMap
        (fun intermediate => right.filterMap (fun range => range.column? intermediate)) := by
  induction left with
  | nil => rfl
  | cons first rest ih =>
    simp only [List.flatMap_cons, List.filterMap_append, List.filterMap_cons, ih,
      List.filterMap_map, Function.comp_def, SourceProjectionRange.compose_column]
    cases first.column? source <;> simp

theorem SourceProjection.compose_column (left right : SourceProjection) (unique : left.Unique)
    (source : Nat) :
    (left.compose right).column? source = (left.column? source).bind right.column? := by
  cases left with
  | identity => simp [compose, column?]
  | mapped first =>
    cases right with
    | identity => simp [compose, column?]
    | mapped second =>
      have size := unique source
      simp only [compose, column?, composed_matches]
      cases selected : first.filterMap (fun range => range.column? source) with
      | nil => rfl
      | cons value rest =>
        have empty : rest = [] := by
          rw [selected, List.length_cons] at size
          exact List.eq_nil_of_length_eq_zero (by omega)
        subst rest
        simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, Option.bind_some]

end NightstreamFPrime.Layout.MatrixProgram
