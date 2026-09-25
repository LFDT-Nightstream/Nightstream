import NightstreamFPrime.Layout.R1CS.Completeness

/-!
Owns structural lowering over an outer list of opaque constraint segments.
The interface exposes satisfaction per segment without materializing or
comparing one flattened production row list.
-/

namespace NightstreamFPrime.Layout.R1CS

open NightstreamFPrime.Circuit

def SegmentsHold (env : Env) : List (List Expr) → Nat → Prop
  | [], _ => True
  | segment :: rest, start =>
      RowsHold env (lowerConstraints segment start).rows ∧
        SegmentsHold env rest (start + totalFreshCount segment)

/-- Lowering a flattened outer list holds exactly when each opaque segment
holds at the fresh-column endpoint of its predecessor. -/
theorem rowsHold_flatten_iff (env : Env) (segments : List (List Expr))
    (start : Nat) :
    RowsHold env (lowerConstraints segments.flatten start).rows ↔
      SegmentsHold env segments start := by
  induction segments generalizing start with
  | nil => simp [SegmentsHold, lowerConstraints, RowsHold]
  | cons segment rest inductionHypothesis =>
      rw [List.flatten_cons, lowerConstraints_append_rows,
        rowsHold_append]
      simp only [SegmentsHold]
      exact and_congr Iff.rfl (inductionHypothesis _)

/-- Complete an ordered list of opaque constraint segments without reducing
the full flattened list in the caller's proof term. Each segment starts at
the exact fresh-column endpoint of its predecessor. -/
theorem lowerSegments_complete (env : Env) (segments : List (List Expr))
    (start : Nat)
    (scope : ∀ expression ∈ segments.flatten,
      expression.VarsBelow start)
    (logical : ConstraintsHold env segments.flatten) :
    ∃ completed,
      AgreesOutside env completed start
          (totalFreshCount segments.flatten) ∧
        SegmentsHold completed segments start := by
  induction segments generalizing env start with
  | nil =>
      refine ⟨env, ?_, trivial⟩
      intro index _outside
      rfl
  | cons segment rest inductionHypothesis =>
      have firstScope : ∀ expression ∈ segment,
          expression.VarsBelow start := by
        intro expression member
        exact scope expression (by simp [member])
      have restScope : ∀ expression ∈ rest.flatten,
          expression.VarsBelow start := by
        intro expression member
        exact scope expression (by simp [member])
      have firstLogical : ConstraintsHold env segment := by
        intro expression member
        exact logical expression (by simp [member])
      rcases lowerConstraints_complete env segment start firstScope firstLogical
          with ⟨afterFirst, firstAgrees, firstRows⟩
      let next := start + totalFreshCount segment
      have restScopeAtNext : ∀ expression ∈ rest.flatten,
          expression.VarsBelow next := by
        intro expression member
        exact Expr.VarsBelow.mono expression (restScope expression member) (by
          unfold next
          omega)
      have restLogical : ConstraintsHold afterFirst rest.flatten := by
        apply constraintsHold_of_agree_below env afterFirst rest.flatten start
          restScope
        · intro index below
          exact firstAgrees index (Or.inl below)
        · intro expression member
          exact logical expression (by simp [member])
      rcases inductionHypothesis afterFirst next restScopeAtNext restLogical with
        ⟨completed, restAgrees, restRows⟩
      have firstRowsAtEnd :
          RowsHold completed (lowerConstraints segment start).rows := by
        apply rowsHold_of_agree_below
          (lowerConstraints segment start).rows next afterFirst completed
        · simpa [next] using
            lowerConstraints_rows_varsBelow segment start firstScope
        · intro index below
          exact restAgrees index (Or.inl below)
        · exact firstRows
      refine ⟨completed, ?_, ⟨firstRowsAtEnd, restRows⟩⟩
      have combined := firstAgrees.append restAgrees
      simpa [List.flatten_cons, next] using combined

/-- Held lowering of an indexed segment list projects to one segment at the
sum of the exact preceding segment fresh counts. The proof is structural in
the outer list and does not inspect a segment's constraints. -/
theorem segmentsHold_ofFn_get (env : Env) {count : Nat}
    (segments : Fin count → List Expr) (start : Nat)
    (holds : SegmentsHold env (List.ofFn segments) start)
    (index : Fin count) :
    RowsHold env
      (lowerConstraints (segments index)
        (start + ((List.ofFn fun current : Fin count =>
          totalFreshCount (segments current)).take index.val).sum)).rows := by
  induction count generalizing start with
  | zero => exact Fin.elim0 index
  | succ count inductionHypothesis =>
      rw [List.ofFn_succ] at holds
      simp only [SegmentsHold] at holds
      refine Fin.cases ?_ (fun tailIndex => ?_) index
      · simpa using holds.1
      · have tail := inductionHypothesis
          (fun current : Fin count => segments current.succ)
          (start + totalFreshCount (segments 0)) holds.2 tailIndex
        simpa [List.ofFn_succ, Nat.add_assoc] using tail

/-- Held lowering of an arbitrary segment list projects to one segment at the
sum of the exact preceding segment fresh counts. -/
theorem segmentsHold_get (env : Env) (segments : List (List Expr))
    (start : Nat) (holds : SegmentsHold env segments start)
    (index : Fin segments.length) :
    RowsHold env
      (lowerConstraints (segments.get index)
        (start + ((List.ofFn fun current : Fin segments.length =>
          totalFreshCount (segments.get current)).take index.val).sum)).rows := by
  have normalized :
      SegmentsHold env
        (List.ofFn fun current : Fin segments.length => segments.get current)
        start := by
    simpa only [List.ofFn_get] using holds
  exact segmentsHold_ofFn_get env
    (fun current : Fin segments.length => segments.get current)
    start normalized index

/-- Held lowering of an indexed constraint list projects to one optimized
constraint lowering at the sum of its exact preceding fresh counts. -/
theorem rowsHold_lowerConstraints_ofFn_get (env : Env) {count : Nat}
    (constraints : Fin count → Expr) (start : Nat)
    (holds : RowsHold env (lowerConstraints (List.ofFn constraints) start).rows)
    (index : Fin count) :
    RowsHold env
      (lowerConstraint (constraints index)
        (start + ((List.ofFn fun current : Fin count =>
          constraintFreshCount (constraints current)).take index.val).sum)).rows := by
  induction count generalizing start with
  | zero => exact Fin.elim0 index
  | succ count inductionHypothesis =>
      rw [List.ofFn_succ] at holds
      have separated :
          RowsHold env (lowerConstraint (constraints 0) start).rows ∧
            RowsHold env
              (lowerConstraints
                (List.ofFn fun current : Fin count => constraints current.succ)
                (start + constraintFreshCount (constraints 0))).rows := by
        apply (rowsHold_append env _ _).mp
        simpa [lowerConstraints] using holds
      refine Fin.cases ?_ (fun tailIndex => ?_) index
      · simpa using separated.1
      · have tail := inductionHypothesis
          (fun current : Fin count => constraints current.succ)
          (start + constraintFreshCount (constraints 0)) separated.2 tailIndex
        simpa [List.ofFn_succ, Nat.add_assoc] using tail

/-- A constraint with a positive optimized fresh count cannot use a direct
row, so its optimized lowering is its generic lowering. -/
theorem lowerConstraint_eq_lowerGenericConstraint_of_fresh_pos
    (expression : Expr) (start : Nat)
    (positive : 0 < constraintFreshCount expression) :
    lowerConstraint expression start = lowerGenericConstraint expression start := by
  cases result : directConstraint expression with
  | none => simp [lowerConstraint, result]
  | some direct => simp [constraintFreshCount, result] at positive

namespace LoweringPlan

/-- A plan whose constraints are a segmented flatten has exactly the same row
predicate as ordered segment satisfaction. The caller supplies only the
constraint-list equality, so its concrete rows stay opaque. -/
theorem rowsHold_iff_segments_of_constraints (plan : LoweringPlan) (env : Env)
    (segments : List (List Expr))
    (constraintsEq : plan.constraints = segments.flatten) :
    RowsHold env plan.rows ↔ SegmentsHold env segments plan.firstFresh := by
  change RowsHold env
    (lowerConstraints plan.constraints plan.firstFresh).rows ↔ _
  rw [constraintsEq]
  exact rowsHold_flatten_iff env segments plan.firstFresh

/-- A lowering plan whose constraint list is a segmented flatten projects
held physical rows to the ordered segment predicate. The plan stays symbolic,
so this theorem does not reduce any caller-owned production row list. -/
theorem rowsHold_segments_of_constraints (plan : LoweringPlan) (env : Env)
    (segments : List (List Expr))
    (constraintsEq : plan.constraints = segments.flatten)
    (physical : RowsHold env plan.rows) :
    SegmentsHold env segments plan.firstFresh := by
  exact (rowsHold_iff_segments_of_constraints plan env segments
    constraintsEq).mp physical

end LoweringPlan

end NightstreamFPrime.Layout.R1CS
