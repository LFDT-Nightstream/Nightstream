import SuperNeo.MatrixTransform
import SuperNeo.MLE

namespace SuperNeo

open F

private def dotF (a b : Array F) : F :=
  if a.size != b.size then
    0
  else
    Id.run do
      let mut acc : F := 0
      for i in [0:a.size] do
        acc := acc + a[i]! * b[i]!
      return acc

/-- One row ring value of bar(Mz), prior to taking ct. -/
def rowBarMzRing (bar : Array (Array F)) (row z : Array F) : Coeffs :=
  if row.size != z.size then
    #[]
  else if row.size % d != 0 then
    #[]
  else
    let nBlocks := row.size / d
    Id.run do
      let mut acc := Array.replicate d (0 : F)
      for t in [0:nBlocks] do
        let start := t * d
        let stop := start + d
        let aBlk := row.extract start stop
        let zBlk := z.extract start stop
        let term := mulRq (superneoBarBlock bar aBlk) zBlk
        acc := vecAdd acc term
      return acc

/-- Ring-valued vector bar(Mz) over matrix rows. -/
def barMzRing (bar : Array (Array F)) (m : Array (Array F)) (z : Array F) : Array Coeffs :=
  m.map (fun row => rowBarMzRing bar row z)

/-- Evaluate a ring-valued vector with scalar weights (inner product over rows). -/
def evalRingVec (ys : Array Coeffs) (weights : Array F) : Coeffs :=
  if ys.size != weights.size then
    #[]
  else
    Id.run do
      let mut acc := Array.replicate d (0 : F)
      for i in [0:ys.size] do
        acc := vecAdd acc (vecScale weights[i]! ys[i]!)
      return acc

/-- Coefficient-row view cf(ys)_ell of a ring vector ys. -/
def coeffRowsOfRingVec (ys : Array Coeffs) : Array (Array F) :=
  Id.run do
    let mut rows := Array.replicate d (Array.replicate ys.size (0 : F))
    for i in [0:ys.size] do
      let yi := ys[i]!
      for ell in [0:d] do
        let row := rows[ell]!
        rows := rows.set! ell (row.set! i yi[ell]!)
    return rows

/-- Evaluate each coefficient row independently with the same weights. -/
def evalCoeffRows (rows : Array (Array F)) (weights : Array F) : Array F :=
  rows.map (fun row => dotF row weights)

/-- Constant-term row projection ct(ys). -/
def ctRow (ys : Array Coeffs) : Array F := ys.map ct

/-- Remark 2 computational identity for an already-built ring vector ys. -/
def evalLinkIdentity (ys : Array Coeffs) (weights : Array F) : Bool :=
  if ys.size != weights.size then
    false
  else
    let y := evalRingVec ys weights
    let coeffSide := evalCoeffRows (coeffRowsOfRingVec ys) weights
    let ctSide := dotF (ctRow ys) weights
    decide (y = coeffSide ∧ ct y = ctSide)

/-- Remark 2 identity specialized to bar(Mz) with MLE-derived r_hat weights. -/
def evalLinkForMatrix (bar : Array (Array F)) (m : Array (Array F)) (z r : Array F) : Bool :=
  let ys := barMzRing bar m z
  let weights := rHat r ys.size
  evalLinkIdentity ys weights

theorem evalLinkIdentity_sound
  {ys : Array Coeffs} {weights : Array F}
  (hOk : evalLinkIdentity ys weights = true) :
  let y := evalRingVec ys weights
  let coeffSide := evalCoeffRows (coeffRowsOfRingVec ys) weights
  let ctSide := dotF (ctRow ys) weights
  y = coeffSide ∧ ct y = ctSide := by
  unfold evalLinkIdentity at hOk
  by_cases hsz : ys.size != weights.size
  · simp [hsz] at hOk
  · simp [hsz] at hOk
    exact hOk

theorem evalLinkForMatrix_sound
  {bar : Array (Array F)} {m : Array (Array F)} {z r : Array F}
  (hOk : evalLinkForMatrix bar m z r = true) :
  let ys := barMzRing bar m z
  let weights := rHat r ys.size
  let y := evalRingVec ys weights
  let coeffSide := evalCoeffRows (coeffRowsOfRingVec ys) weights
  let ctSide := dotF (ctRow ys) weights
  y = coeffSide ∧ ct y = ctSide := by
  unfold evalLinkForMatrix at hOk
  exact evalLinkIdentity_sound hOk

end SuperNeo
