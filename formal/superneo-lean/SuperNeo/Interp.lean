import SuperNeo.Field

namespace SuperNeo

open F

/-- Evaluate polynomial with low->high coefficients via Horner. -/
def polyEval (coeffs : Array F) (x : F) : F :=
  match coeffs.toList.reverse with
  | [] => 0
  | c :: cs => cs.foldl (fun acc coeff => acc * x + coeff) c

/-- Lagrange interpolation from evaluations (xs, ys), low->high coefficient output. -/
def interpolateFromEvals (xs ys : Array F) : Array F :=
  Id.run do
    let n := xs.size
    if n != ys.size then
      return #[]
    let mut coeffs := Array.replicate n (0 : F)

    for i in [0:n] do
      let mut numer := Array.replicate n (0 : F)
      numer := numer.set! 0 (1 : F)
      let mut curDeg := 0

      for j in [0:n] do
        if i != j then
          let xj := xs[j]!
          let mut next := Array.replicate n (0 : F)
          for d in [0:(curDeg + 1)] do
            next := next.set! (d + 1) (next[d + 1]! + numer[d]!)
            next := next.set! d (next[d]! - xj * numer[d]!)
          numer := next
          curDeg := curDeg + 1

      let mut denom : F := 1
      for j in [0:n] do
        if i != j then
          denom := denom * (xs[i]! - xs[j]!)

      let scale := ys[i]! * denom⁻¹
      for d in [0:(curDeg + 1)] do
        coeffs := coeffs.set! d (coeffs[d]! + scale * numer[d]!)

    return coeffs

def interpolationCase
  (xs ys expectedCoeffs : Array F)
  (evalPoint expectedEval : F) : Bool :=
  let coeffs := interpolateFromEvals xs ys
  decide (coeffs = expectedCoeffs ∧ polyEval coeffs evalPoint = expectedEval)

theorem interpolationCase_sound
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  (hOk : interpolationCase xs ys expectedCoeffs evalPoint expectedEval = true) :
  let coeffs := interpolateFromEvals xs ys
  coeffs = expectedCoeffs ∧ polyEval coeffs evalPoint = expectedEval := by
  unfold interpolationCase at hOk
  exact decide_eq_true_eq.mp hOk

theorem interpolationCase_complete
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  (h :
    let coeffs := interpolateFromEvals xs ys
    coeffs = expectedCoeffs ∧ polyEval coeffs evalPoint = expectedEval) :
  interpolationCase xs ys expectedCoeffs evalPoint expectedEval = true := by
  unfold interpolationCase
  exact decide_eq_true h

end SuperNeo
