import SuperNeo.Norm

namespace SuperNeo

open F

/-- Centered integer representative in [-(q-1)/2, (q-1)/2]. -/
def centeredInt (a : F) : Int :=
  if a.val <= halfQ then
    Int.ofNat a.val
  else
    Int.ofNat a.val - Int.ofNat q

private def balancedResidue (a : Int) (b : Nat) : Int :=
  let bi := Int.ofNat b
  let half := Int.ofNat (b / 2)
  let q0 :=
    if a >= 0 then
      a / bi
    else
      - ((-a) / bi)
  let r0 := a - q0 * bi
  let r1 := if r0 > half then r0 - bi else r0
  if r1 < -half then r1 + bi else r1

/-- Balanced base-b split of one field element into k digits. -/
def splitBalancedScalar (a : F) (b k : Nat) : Array F :=
  if b < 2 then
    Array.replicate k (0 : F)
  else
    Id.run do
      let mut out : Array F := #[]
      let mut cur := centeredInt a
      let bi := Int.ofNat b
      for _ in [0:k] do
        let r := balancedResidue cur b
        out := out.push (F.ofInt r)
        cur :=
          if cur - r >= 0 then
            (cur - r) / bi
          else
            - ((-(cur - r)) / bi)
      return out

/-- Balanced base-b split of a vector into k digit-vectors. -/
def splitBalancedVec (z : Array F) (b k : Nat) : Array (Array F) :=
  if b < 2 then
    Array.replicate k (Array.replicate z.size (0 : F))
  else
    Id.run do
      let mut digits := Array.replicate k (Array.replicate z.size (0 : F))
      for j in [0:z.size] do
        let ds := splitBalancedScalar z[j]! b k
        for i in [0:k] do
          let row := digits[i]!
          digits := digits.set! i (row.set! j ds[i]!)
      return digits

/-- Recompose z = Σ b^i z_i from split digits. -/
def recomposeSplitDigits (digits : Array (Array F)) (b : Nat) : Array F :=
  if digits.isEmpty then
    #[]
  else
    let m := (digits[0]!).size
    Id.run do
      let mut out := Array.replicate m (0 : F)
      let mut scale : F := 1
      let bF := F.ofNat b
      for i in [0:digits.size] do
        let row := digits[i]!
        for j in [0:m] do
          out := out.set! j (out[j]! + scale * row[j]!)
        scale := scale * bF
      return out

def digitsWithinBase (digits : Array (Array F)) (b : Nat) : Bool :=
  digits.all (fun row => row.all (fun x => normInfF x < b))

def splitRoundTrip (z : Array F) (b k : Nat) : Bool :=
  if b < 2 then
    false
  else
    let digits := splitBalancedVec z b k
    decide (recomposeSplitDigits digits b = z) && digitsWithinBase digits b

theorem splitRoundTrip_sound
  {z : Array F} {b k : Nat}
  (hOk : splitRoundTrip z b k = true) :
  b ≥ 2 ∧
    let digits := splitBalancedVec z b k
    recomposeSplitDigits digits b = z ∧ digitsWithinBase digits b = true := by
  unfold splitRoundTrip at hOk
  by_cases hb : b < 2
  · simp [hb] at hOk
  · have hbGe : b ≥ 2 := Nat.le_of_not_lt hb
    simp [hb] at hOk
    have hAnd :
      decide (recomposeSplitDigits (splitBalancedVec z b k) b = z) = true ∧
        digitsWithinBase (splitBalancedVec z b k) b = true := by
      simpa [Bool.and_eq_true] using hOk
    refine ⟨hbGe, ?_⟩
    exact ⟨decide_eq_true_eq.mp hAnd.1, hAnd.2⟩

theorem splitRoundTrip_complete
  {z : Array F} {b k : Nat}
  (hProp : b ≥ 2 ∧
    let digits := splitBalancedVec z b k
    recomposeSplitDigits digits b = z ∧ digitsWithinBase digits b = true) :
  splitRoundTrip z b k = true := by
  rcases hProp with ⟨hbGe, hRest⟩
  have hbNotLt : ¬ b < 2 := Nat.not_lt.mpr hbGe
  rcases hRest with ⟨hRec, hDigits⟩
  unfold splitRoundTrip
  simp [hbNotLt, hRec, hDigits]

/-- Tiny sanity check for Definition 3 + split_b style decomposition. -/
def decompSanity : Bool :=
  let z := #[F.ofNat 3, F.ofInt (-2), F.ofNat (q - 1)]
  splitRoundTrip z 2 8

end SuperNeo
