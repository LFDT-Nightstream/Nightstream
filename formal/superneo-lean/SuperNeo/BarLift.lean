import SuperNeo.Embedding

/-!
Definition-8 bar-lift layer (compact scaffold).

Reading guide:
1. `barLiftVector` / `barLiftMatrix` are the core operators.
2. `barLiftLinearityAssumption` is the theorem-facing contract used by downstream files.
3. `barLiftLinearityCheckAssumption` is the Bool-facing compatibility surface.
4. `_of_assumption`, `_of_checkAssumption`, and `_iff_...` are conversion bridges.

This file is intentionally small: it exposes one clear semantic boundary plus
check/prop interop, without wrapper quadrants.
-/

namespace SuperNeo

open F

/-- Vector bar-lift operator. In the compact scaffold it is identity. -/
def barLiftVector (_bar : Array (Array F)) (v : Array F) : Array F :=
  v

/-- Matrix bar-lift operator (row-wise). -/
def barLiftMatrix (bar : Array (Array F)) (m : Array (Array F)) : Array (Array F) :=
  m.map (barLiftVector bar)

@[simp] theorem barLiftVector_eq (bar : Array (Array F)) (v : Array F) :
    barLiftVector bar v = v := by
  rfl

@[simp] theorem barLiftMatrix_eq (bar : Array (Array F)) (m : Array (Array F)) :
    barLiftMatrix bar m = m := by
  apply Array.ext
  · simp [barLiftMatrix]
  · intro i hiL hiR
    simp [barLiftMatrix, barLiftVector]

theorem barLiftVector_add (bar : Array (Array F)) (v w : Array F) :
    barLiftVector bar (vecAdd v w) = vecAdd (barLiftVector bar v) (barLiftVector bar w) := by
  rfl

theorem barLiftVector_scale (bar : Array (Array F)) (s : F) (v : Array F) :
    barLiftVector bar (vecScale s v) = vecScale s (barLiftVector bar v) := by
  rfl

/--
Theorem-facing linearity contract for bar-lift.

Downstream theorem files should depend on this Prop boundary, not on Bool checks.
-/
def barLiftLinearityAssumption (bar : Array (Array F)) : Prop :=
  (∀ v w : Array F, v.size = w.size →
    barLiftVector bar (vecAdd v w) = vecAdd (barLiftVector bar v) (barLiftVector bar w)) ∧
  (∀ s : F, ∀ v : Array F,
    barLiftVector bar (vecScale s v) = vecScale s (barLiftVector bar v))

/--
Check-facing universal contract for bar-lift linearity.

This is retained for executable compatibility, then bridged back to Prop.
-/
def barLiftLinearityCheckAssumption (bar : Array (Array F)) : Prop :=
  ∀ s : F, ∀ v w : Array F, v.size = w.size →
    decide (barLiftVector bar (vecAdd v w) = vecAdd (barLiftVector bar v) (barLiftVector bar w)) = true ∧
    decide (barLiftVector bar (vecScale s v) = vecScale s (barLiftVector bar v)) = true

theorem barLiftLinearityAssumption_native
  (bar : Array (Array F)) :
  barLiftLinearityAssumption bar := by
  constructor
  · intro v w _hSize
    exact barLiftVector_add bar v w
  · intro s v
    exact barLiftVector_scale bar s v

/-- Convert theorem-facing bar-lift linearity into the check-facing contract. -/
theorem barLiftLinearityCheckAssumption_of_assumption
  {bar : Array (Array F)}
  (hAssm : barLiftLinearityAssumption bar) :
  barLiftLinearityCheckAssumption bar := by
  intro s v w hSize
  refine ⟨?_, ?_⟩
  · exact decide_eq_true (hAssm.1 v w hSize)
  · exact decide_eq_true (hAssm.2 s v)

/-- Convert check-facing bar-lift linearity into the theorem-facing contract. -/
theorem barLiftLinearityAssumption_of_checkAssumption
  {bar : Array (Array F)}
  (hCheck : barLiftLinearityCheckAssumption bar) :
  barLiftLinearityAssumption bar := by
  constructor
  · intro v w hSize
    exact decide_eq_true_eq.mp (hCheck 0 v w hSize).1
  · intro s v
    exact decide_eq_true_eq.mp (hCheck s v v rfl).2

/-- Equivalence between theorem-facing and check-facing bar-lift contracts. -/
theorem barLiftLinearityAssumption_iff_checkAssumption
  {bar : Array (Array F)} :
  barLiftLinearityAssumption bar ↔ barLiftLinearityCheckAssumption bar := by
  constructor
  · exact barLiftLinearityCheckAssumption_of_assumption
  · exact barLiftLinearityAssumption_of_checkAssumption


end SuperNeo
