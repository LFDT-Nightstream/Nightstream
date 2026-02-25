import SuperNeo.EvalLink

namespace SuperNeo

open F

/-- Compute y = M~z(r) in ring form using Remark 2 machinery. -/
def evalBarMzAt (bar : Array (Array F)) (m : Array (Array F)) (z r : Array F) : Coeffs :=
  let ys := barMzRing bar m z
  let weights := rHat r ys.size
  evalRingVec ys weights

/-- Linear combination of two field vectors with scalar coefficients. -/
def linComb2Vec (ρ1 ρ2 : F) (z1 z2 : Array F) : Array F :=
  vecAdd (vecScale ρ1 z1) (vecScale ρ2 z2)

/-- Theorem 5 computational check for two inputs over base-field scalars. -/
def evalHom2
  (bar : Array (Array F))
  (m : Array (Array F))
  (z1 z2 r : Array F)
  (ρ1 ρ2 : F) : Bool :=
  if z1.size != z2.size then
    false
  else if !(m.all (fun row => row.size = z1.size ∧ row.size % d = 0)) then
    false
  else
    let y1 := evalBarMzAt bar m z1 r
    let y2 := evalBarMzAt bar m z2 r
    let zStar := linComb2Vec ρ1 ρ2 z1 z2
    let yLin := vecAdd (vecScale ρ1 y1) (vecScale ρ2 y2)
    let yDirect := evalBarMzAt bar m zStar r
    decide (yLin = yDirect ∧ ct yLin = ρ1 * ct y1 + ρ2 * ct y2)

theorem evalHom2_sound
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z1 z2 r : Array F}
  {ρ1 ρ2 : F}
  (hOk : evalHom2 bar m z1 z2 r ρ1 ρ2 = true) :
  let y1 := evalBarMzAt bar m z1 r
  let y2 := evalBarMzAt bar m z2 r
  let yLin := vecAdd (vecScale ρ1 y1) (vecScale ρ2 y2)
  let yDirect := evalBarMzAt bar m (linComb2Vec ρ1 ρ2 z1 z2) r
  yLin = yDirect ∧ ct yLin = ρ1 * ct y1 + ρ2 * ct y2 := by
  unfold evalHom2 at hOk
  by_cases hsz : z1.size != z2.size
  · simp [hsz] at hOk
  · by_cases hall : m.all (fun row => row.size = z1.size ∧ row.size % d = 0) = true
    · simp [hsz] at hOk
      exact hOk.2
    · simp [hsz] at hOk
      exact hOk.2

end SuperNeo
