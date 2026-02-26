import SuperNeo.Field

namespace SuperNeo

open F

def oneMinus (x : F) : F := (1 : F) - x

def eqTerm (x y : F) : F :=
  x * y + oneMinus x * oneMinus y

/-- eq(x,y) = Π_i (x_i y_i + (1-x_i)(1-y_i)). -/
def eqPoly (x y : Array F) : F :=
  if x.size != y.size then
    0
  else
    Id.run do
      let mut acc : F := 1
      for i in [0:x.size] do
        acc := acc * eqTerm x[i]! y[i]!
      return acc

def bitsToFArray (width mask : Nat) : Array F :=
  Id.run do
    let mut out := Array.replicate width (0 : F)
    for i in [0:width] do
      let bit := (mask / (2 ^ i)) % 2
      out := out.set! i (F.ofNat bit)
    return out

def isBoolF (x : F) : Bool :=
  decide (x = 0 ∨ x = 1)

def isBoolFProp (x : F) : Prop :=
  x = 0 ∨ x = 1

theorem isBoolF_sound
  {x : F}
  (hOk : isBoolF x = true) :
  isBoolFProp x := by
  unfold isBoolF at hOk
  unfold isBoolFProp
  exact decide_eq_true_eq.mp hOk

theorem isBoolF_complete
  {x : F}
  (hProp : isBoolFProp x) :
  isBoolF x = true := by
  unfold isBoolF
  exact decide_eq_true hProp

theorem isBoolF_iff_prop
  {x : F} :
  isBoolF x = true ↔ isBoolFProp x := by
  constructor
  · exact isBoolF_sound
  · exact isBoolF_complete

/-- Indicator behavior on Boolean points: eq(x,y)=1 iff x=y, else 0. -/
def eqHypercubeIndicator (x y : Array F) : Bool :=
  if x.size != y.size then
    false
  else if !(x.all isBoolF && y.all isBoolF) then
    false
  else
    let e := eqPoly x y
    if decide (x = y) then
      decide (e = 1)
    else
      decide (e = 0)

def eqHypercubeIndicatorProp (x y : Array F) : Prop :=
  (x.size != y.size) = false ∧
  (x.all isBoolF && y.all isBoolF) = true ∧
  let e := eqPoly x y
  if x = y then e = 1 else e = 0

theorem eqHypercubeIndicator_sound
  {x y : Array F}
  (hOk : eqHypercubeIndicator x y = true) :
  eqHypercubeIndicatorProp x y := by
  unfold eqHypercubeIndicator at hOk
  by_cases hSize : x.size != y.size
  · simp [hSize] at hOk
  · have hSizeFalse : (x.size != y.size) = false := by simp [hSize]
    by_cases hBool : (x.all isBoolF && y.all isBoolF) = true
    · simp [hSize, hBool] at hOk
      refine ⟨hSizeFalse, hBool, ?_⟩
      by_cases hEq : x = y
      · have hEqPoly : eqPoly x y = 1 := by simpa [hEq] using hOk
        simpa [hEq] using hEqPoly
      · have hEqPoly : eqPoly x y = 0 := by simpa [hEq] using hOk
        simpa [hEq] using hEqPoly
    · simp [hSize, hBool] at hOk

theorem eqHypercubeIndicator_complete
  {x y : Array F}
  (hProp : eqHypercubeIndicatorProp x y) :
  eqHypercubeIndicator x y = true := by
  rcases hProp with ⟨hSize, hBool, hCase⟩
  unfold eqHypercubeIndicator
  by_cases hEq : x = y
  · have hEqPoly : eqPoly x y = 1 := by simpa [hEq] using hCase
    have hDecEq : decide (x = y) = true := decide_eq_true hEq
    simp [hSize, hBool, hDecEq, hEqPoly]
  · have hEqPoly : eqPoly x y = 0 := by simpa [hEq] using hCase
    have hDecEq : decide (x = y) = false := decide_eq_false hEq
    simp [hSize, hBool, hDecEq, hEqPoly]

theorem eqHypercubeIndicator_iff_prop
  {x y : Array F} :
  eqHypercubeIndicator x y = true ↔ eqHypercubeIndicatorProp x y := by
  constructor
  · exact eqHypercubeIndicator_sound
  · exact eqHypercubeIndicator_complete

def eqPolySanity : Bool :=
  let x := #[0, 1, 0, 1]
  let y := #[0, 1, 0, 1]
  let z := #[1, 0, 1, 0]
  decide (eqPoly x y = 1 ∧ eqPoly x z = 0)

end SuperNeo
