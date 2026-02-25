import SuperNeo.InvertibilityAxioms

namespace SuperNeo

open F

def pairwiseWithinBound (cset : Array Coeffs) (bound : Nat) : Bool :=
  Id.run do
    let mut ok := true
    for i in [0:cset.size] do
      for j in [0:cset.size] do
        if i < j then
          let diff := coeffSub cset[i]! cset[j]!
          ok := ok && decide (normInfCoeffs diff < bound)
    return ok

def strongSamplingSet (cset : Array Coeffs) : Bool :=
  pairwiseWithinBound cset bInvApprox

def maxRhoNorm (cset : Array Coeffs) : Nat :=
  cset.foldl (fun m rho => Nat.max m (normInfCoeffs rho)) 0

/-- Theorem 9 interface: expansion factor upper bound 2 * phi(eta) * max||rho||∞. -/
def theorem9UpperBound (maxNorm : Nat) : Nat :=
  2 * d * maxNorm

private def mulRatio (rho v : Coeffs) : Nat :=
  let denom := normInfCoeffs v
  if denom = 0 then
    0
  else
    normInfCoeffs (mulRq rho v) / denom

def empiricalExpansionFactor (cset : Array Coeffs) (samples : Array Coeffs) : Nat :=
  cset.foldl
    (fun outer rho =>
      samples.foldl (fun inner v => Nat.max inner (mulRatio rho v)) outer)
    0

def samplingSetBoundCheck (cset : Array Coeffs) (samples : Array Coeffs) : Bool :=
  let empirical := empiricalExpansionFactor cset samples
  let bound := theorem9UpperBound (maxRhoNorm cset)
  decide (empirical <= bound)

theorem samplingSetBoundCheck_sound
  {cset : Array Coeffs} {samples : Array Coeffs}
  (hOk : samplingSetBoundCheck cset samples = true) :
  empiricalExpansionFactor cset samples <= theorem9UpperBound (maxRhoNorm cset) := by
  unfold samplingSetBoundCheck at hOk
  exact decide_eq_true_eq.mp hOk

theorem samplingSetBoundCheck_complete
  {cset : Array Coeffs} {samples : Array Coeffs}
  (hBound : empiricalExpansionFactor cset samples <= theorem9UpperBound (maxRhoNorm cset)) :
  samplingSetBoundCheck cset samples = true := by
  unfold samplingSetBoundCheck
  exact decide_eq_true hBound

def samplingSetSanity : Bool :=
  decide (theorem9UpperBound 2 = 216)

end SuperNeo
