import SuperNeo.EqPoly

namespace SuperNeo

open F

/-- Sum_x eq(x,z) * q(x) over the Boolean hypercube table q. -/
def eqLiftFromTable (qVals z : Array F) : F :=
  let ell := z.size
  let n := 2 ^ ell
  if qVals.size != n then
    0
  else
    Id.run do
      let mut acc : F := 0
      for mask in [0:n] do
        let x := bitsToFArray ell mask
        acc := acc + eqPoly x z * qVals[mask]!
      return acc

def eqLiftBooleanIndicator (qVals : Array F) (ell mask : Nat) : Bool :=
  if qVals.size != 2 ^ ell then
    false
  else if mask >= 2 ^ ell then
    false
  else
    let z := bitsToFArray ell mask
    decide (eqLiftFromTable qVals z = qVals[mask]!)

def eqLiftAllBoolean (qVals : Array F) (ell : Nat) : Bool :=
  if qVals.size != 2 ^ ell then
    false
  else
    Id.run do
      let mut ok := true
      for mask in [0:(2 ^ ell)] do
        ok := ok && eqLiftBooleanIndicator qVals ell mask
      return ok

/-- Schwartz-Zippel failure bound interface: d / |S|. -/
def schwartzZippelBoundLeOne (totalDegree setSize : Nat) : Bool :=
  if setSize = 0 then
    false
  else
    decide (totalDegree <= setSize)

theorem schwartzZippelBoundLeOne_sound
  {totalDegree setSize : Nat}
  (hOk : schwartzZippelBoundLeOne totalDegree setSize = true) :
  setSize ≠ 0 ∧ totalDegree <= setSize := by
  unfold schwartzZippelBoundLeOne at hOk
  by_cases hzero : setSize = 0
  · simp [hzero] at hOk
  · have hDec : decide (totalDegree <= setSize) = true := by
      simpa [hzero] using hOk
    exact ⟨hzero, decide_eq_true_eq.mp hDec⟩

theorem schwartzZippelBoundLeOne_complete
  {totalDegree setSize : Nat}
  (hNonzero : setSize ≠ 0)
  (hBound : totalDegree <= setSize) :
  schwartzZippelBoundLeOne totalDegree setSize = true := by
  unfold schwartzZippelBoundLeOne
  simp [hNonzero, decide_eq_true hBound]

def polyLemmaSanity : Bool :=
  let qVals : Array F := #[3, 1, 4, 1, 5, 9, 2, 6]
  eqLiftAllBoolean qVals 3 && schwartzZippelBoundLeOne 5 17

end SuperNeo
