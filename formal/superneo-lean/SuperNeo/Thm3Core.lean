import SuperNeo.Ring
import SuperNeo.Dimensions

namespace SuperNeo

open F

def IsDVec (v : Array F) : Prop := v.size = d

def IsDBarMatrix (bar : Array (Array F)) : Prop :=
  bar.size = d ∧ bar.all (fun row => row.size = d) = true

/-- P10 (Theorem 3 core) as a reusable proposition on one pair of vectors. -/
def p10CoreProp (bar : Array (Array F)) (a b : Array F) : Prop :=
  ct (mulRq (superneoBarBlock bar a) b) = dot a b

/--
Theorem-native assumption interface for P10.
This records the core identity directly, separated from executable checks.
-/
def thm3CoreAssumption (bar : Array (Array F)) : Prop :=
  ∀ {a b : Array F}, IsDVec a -> IsDVec b -> p10CoreProp bar a b

def p10CoreCheck (bar : Array (Array F)) (a b : Array F) : Bool :=
  decide (ct (mulRq (superneoBarBlock bar a) b) = dot a b)

theorem p10CoreCheck_sound
  {bar : Array (Array F)} {a b : Array F}
  (hOk : p10CoreCheck bar a b = true) :
  p10CoreProp bar a b := by
  unfold p10CoreCheck at hOk
  unfold p10CoreProp
  exact decide_eq_true_eq.mp hOk

theorem p10CoreCheck_complete
  {bar : Array (Array F)} {a b : Array F}
  (hProp : p10CoreProp bar a b) :
  p10CoreCheck bar a b = true := by
  unfold p10CoreCheck
  exact decide_eq_true hProp

theorem p10Core_of_assumption
  {bar : Array (Array F)} {a b : Array F}
  (hThm3 : thm3CoreAssumption bar)
  (ha : IsDVec a)
  (hb : IsDVec b) :
  p10CoreProp bar a b := by
  exact hThm3 ha hb

theorem p10CoreCheck_true_of_assumption
  {bar : Array (Array F)} {a b : Array F}
  (hThm3 : thm3CoreAssumption bar)
  (ha : IsDVec a)
  (hb : IsDVec b) :
  p10CoreCheck bar a b = true := by
  exact p10CoreCheck_complete (p10Core_of_assumption hThm3 ha hb)

theorem thm3CoreAssumption_of_check_family
  {bar : Array (Array F)}
  (hChecks : ∀ {a b : Array F}, IsDVec a -> IsDVec b -> p10CoreCheck bar a b = true) :
  thm3CoreAssumption bar := by
  intro a b ha hb
  exact p10CoreCheck_sound (hChecks ha hb)

/--
Theorem-3 family with explicit dimensional preconditions and a check-backed witness.
The dimensional wrappers make assumptions explicit for later theorem composition.
-/
theorem p10Core_of_preconditions
  {bar : Array (Array F)} {a b : Array F}
  (hBar : IsDBarMatrix bar)
  (ha : IsDVec a)
  (hb : IsDVec b)
  (hCheck : p10CoreCheck bar a b = true) :
  p10CoreProp bar a b := by
  let _ := hBar
  let _ := ha
  let _ := hb
  exact p10CoreCheck_sound hCheck

/--
Theorem-3 family with explicit dimensional preconditions and theorem-native assumption.
-/
theorem p10Core_of_preconditions_props
  {bar : Array (Array F)} {a b : Array F}
  (hBar : IsDBarMatrix bar)
  (ha : IsDVec a)
  (hb : IsDVec b)
  (hThm3 : thm3CoreAssumption bar) :
  p10CoreProp bar a b := by
  let _ := hBar
  exact p10Core_of_assumption hThm3 ha hb

end SuperNeo
