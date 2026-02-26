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

/-- Proposition-level counterpart of `evalHom2`. -/
def evalHom2Prop
  (bar : Array (Array F))
  (m : Array (Array F))
  (z1 z2 r : Array F)
  (ρ1 ρ2 : F) : Prop :=
  (z1.size != z2.size) = false ∧
    MatrixRowsCompatible m z1 ∧
    let y1 := evalBarMzAt bar m z1 r
    let y2 := evalBarMzAt bar m z2 r
    let yLin := vecAdd (vecScale ρ1 y1) (vecScale ρ2 y2)
    let yDirect := evalBarMzAt bar m (linComb2Vec ρ1 ρ2 z1 z2) r
    yLin = yDirect ∧ ct yLin = ρ1 * ct y1 + ρ2 * ct y2

/--
Theorem-native assumption interface for P14/Theorem 5:
for size-compatible inputs with row-shape compatibility, evaluation is linear.
-/
def p14EvalHomAssumption
  (bar : Array (Array F))
  (m : Array (Array F))
  (r : Array F)
  (ρ1 ρ2 : F) : Prop :=
  ∀ z1 z2 : Array F, z1.size = z2.size -> MatrixRowsCompatible m z1 ->
    evalHom2Prop bar m z1 z2 r ρ1 ρ2

/--
Check-oriented universal assumption interface for P14, conditioned on the same
shape preconditions as the theorem-native form.
-/
def p14EvalHomCheckAssumption
  (bar : Array (Array F))
  (m : Array (Array F))
  (r : Array F)
  (ρ1 ρ2 : F) : Prop :=
  ∀ z1 z2 : Array F, z1.size = z2.size -> MatrixRowsCompatible m z1 ->
    evalHom2 bar m z1 z2 r ρ1 ρ2 = true

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

theorem evalHom2_sound_full
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z1 z2 r : Array F}
  {ρ1 ρ2 : F}
  (hOk : evalHom2 bar m z1 z2 r ρ1 ρ2 = true) :
  evalHom2Prop bar m z1 z2 r ρ1 ρ2 := by
  unfold evalHom2 at hOk
  cases hsz : (z1.size != z2.size) with
  | true =>
      simp [hsz] at hOk
  | false =>
      simp [hsz] at hOk
      exact ⟨hsz, hOk.1, hOk.2⟩

theorem evalHom2_complete
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z1 z2 r : Array F}
  {ρ1 ρ2 : F}
  (hProp : evalHom2Prop bar m z1 z2 r ρ1 ρ2) :
  evalHom2 bar m z1 z2 r ρ1 ρ2 = true := by
  rcases hProp with ⟨hsz, hRows, hEq⟩
  have hRowsAll : m.all (fun row => row.size = z1.size ∧ row.size % d = 0) = true :=
    by
      apply (Array.all_eq_true).2
      intro i hi
      exact decide_eq_true (hRows i hi)
  unfold evalHom2
  rw [hsz]
  rw [hRowsAll]
  exact decide_eq_true hEq

theorem evalHom2Prop_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z1 z2 r : Array F}
  {ρ1 ρ2 : F}
  (hAssm : p14EvalHomAssumption bar m r ρ1 ρ2)
  (hSize : z1.size = z2.size)
  (hRows : MatrixRowsCompatible m z1) :
  evalHom2Prop bar m z1 z2 r ρ1 ρ2 := by
  exact hAssm z1 z2 hSize hRows

theorem evalHom2_true_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z1 z2 r : Array F}
  {ρ1 ρ2 : F}
  (hAssm : p14EvalHomAssumption bar m r ρ1 ρ2)
  (hSize : z1.size = z2.size)
  (hRows : MatrixRowsCompatible m z1) :
  evalHom2 bar m z1 z2 r ρ1 ρ2 = true := by
  exact evalHom2_complete (evalHom2Prop_of_assumption hAssm hSize hRows)

theorem p14EvalHomAssumption_of_checkAssumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {r : Array F}
  {ρ1 ρ2 : F}
  (hCheck : p14EvalHomCheckAssumption bar m r ρ1 ρ2) :
  p14EvalHomAssumption bar m r ρ1 ρ2 := by
  intro z1 z2 hSize hRows
  exact evalHom2_sound_full (hCheck z1 z2 hSize hRows)

theorem p14EvalHomCheckAssumption_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {r : Array F}
  {ρ1 ρ2 : F}
  (hAssm : p14EvalHomAssumption bar m r ρ1 ρ2) :
  p14EvalHomCheckAssumption bar m r ρ1 ρ2 := by
  intro z1 z2 hSize hRows
  exact evalHom2_complete (hAssm z1 z2 hSize hRows)

theorem evalHom2Prop_size_eq
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z1 z2 r : Array F}
  {ρ1 ρ2 : F}
  (hProp : evalHom2Prop bar m z1 z2 r ρ1 ρ2) :
  z1.size = z2.size := by
  have hSizeFalse : (z1.size != z2.size) = false := hProp.1
  by_cases hEq : z1.size = z2.size
  · exact hEq
  · have hNeTrue : (z1.size != z2.size) = true := by
      simp [hEq]
    rw [hNeTrue] at hSizeFalse
    cases hSizeFalse

theorem evalHom2_iff_prop
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z1 z2 r : Array F}
  {ρ1 ρ2 : F} :
  evalHom2 bar m z1 z2 r ρ1 ρ2 = true ↔ evalHom2Prop bar m z1 z2 r ρ1 ρ2 := by
  constructor
  · exact evalHom2_sound_full
  · exact evalHom2_complete

end SuperNeo
