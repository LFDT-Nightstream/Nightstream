import SuperNeo.EvalLink

/-! Evaluation-hom layer for Remark-2-style identities (P14). -/


namespace SuperNeo

open F

/-- Compute y = M~z(r) in ring form using Remark 2 machinery. -/
def evalBarMzAt (bar : Array (Array F)) (m : Array (Array F)) (z r : Array F) : Coeffs :=
  let ys := barMzRing bar m z
  let weights := rHat r ys.size
  evalRingVec ys weights

theorem evalBarMzAt_size
  (bar : Array (Array F)) (m : Array (Array F)) (z r : Array F) :
  (evalBarMzAt bar m z r).size = d := by
  unfold evalBarMzAt
  dsimp
  apply evalRingVec_size_of_size_eq
  simp [rHat_size]

/-- Linear combination of two field vectors with scalar coefficients. -/
def linComb2Vec (ρ1 ρ2 : F) (z1 z2 : Array F) : Array F :=
  vecAdd (vecScale ρ1 z1) (vecScale ρ2 z2)

theorem linComb2Vec_size_of_eq
  {ρ1 ρ2 : F} {z1 z2 : Array F}
  (hSize : z1.size = z2.size) :
  (linComb2Vec ρ1 ρ2 z1 z2).size = z1.size := by
  unfold linComb2Vec
  simpa [vecScale_size] using vecAdd_size_of_eq
    (a := vecScale ρ1 z1) (b := vecScale ρ2 z2) (by simpa [vecScale_size] using hSize)

theorem linComb2Vec_size_of_ne
  {ρ1 ρ2 : F} {z1 z2 : Array F}
  (hSize : z1.size ≠ z2.size) :
  (linComb2Vec ρ1 ρ2 z1 z2).size = 0 := by
  unfold linComb2Vec
  simpa [vecScale_size] using vecAdd_size_of_ne
    (a := vecScale ρ1 z1) (b := vecScale ρ2 z2) (by simpa [vecScale_size] using hSize)

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

theorem evalHom2Prop_rows_compatible
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z1 z2 r : Array F}
  {ρ1 ρ2 : F}
  (hProp : evalHom2Prop bar m z1 z2 r ρ1 ρ2) :
  MatrixRowsCompatible m z1 := by
  exact hProp.2.1

theorem evalHom2Prop_rows_compatible_z2
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z1 z2 r : Array F}
  {ρ1 ρ2 : F}
  (hProp : evalHom2Prop bar m z1 z2 r ρ1 ρ2) :
  MatrixRowsCompatible m z2 := by
  have hRows : MatrixRowsCompatible m z1 :=
    evalHom2Prop_rows_compatible
      (bar := bar) (m := m) (z1 := z1) (z2 := z2) (r := r) (ρ1 := ρ1) (ρ2 := ρ2) hProp
  have hSize : z1.size = z2.size :=
    evalHom2Prop_size_eq
      (bar := bar) (m := m) (z1 := z1) (z2 := z2) (r := r) (ρ1 := ρ1) (ρ2 := ρ2) hProp
  exact matrixRowsCompatible_of_size_eq hRows hSize

theorem evalHom2Prop_rows_compatible_linComb2Vec
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z1 z2 r : Array F}
  {ρ1 ρ2 : F}
  (hProp : evalHom2Prop bar m z1 z2 r ρ1 ρ2) :
  MatrixRowsCompatible m (linComb2Vec ρ1 ρ2 z1 z2) := by
  have hRows : MatrixRowsCompatible m z1 :=
    evalHom2Prop_rows_compatible
      (bar := bar) (m := m) (z1 := z1) (z2 := z2) (r := r) (ρ1 := ρ1) (ρ2 := ρ2) hProp
  have hSize : z1.size = z2.size :=
    evalHom2Prop_size_eq
      (bar := bar) (m := m) (z1 := z1) (z2 := z2) (r := r) (ρ1 := ρ1) (ρ2 := ρ2) hProp
  have hLin : z1.size = (linComb2Vec ρ1 ρ2 z1 z2).size := by
    simpa using
      (linComb2Vec_size_of_eq (ρ1 := ρ1) (ρ2 := ρ2) (z1 := z1) (z2 := z2) hSize).symm
  exact matrixRowsCompatible_of_size_eq hRows hLin

theorem evalHom2Prop_eval_eq
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z1 z2 r : Array F}
  {ρ1 ρ2 : F}
  (hProp : evalHom2Prop bar m z1 z2 r ρ1 ρ2) :
  let y1 := evalBarMzAt bar m z1 r
  let y2 := evalBarMzAt bar m z2 r
  let yLin := vecAdd (vecScale ρ1 y1) (vecScale ρ2 y2)
  let yDirect := evalBarMzAt bar m (linComb2Vec ρ1 ρ2 z1 z2) r
  yLin = yDirect := by
  exact And.left hProp.2.2

theorem evalHom2Prop_ct_eq
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z1 z2 r : Array F}
  {ρ1 ρ2 : F}
  (hProp : evalHom2Prop bar m z1 z2 r ρ1 ρ2) :
  let y1 := evalBarMzAt bar m z1 r
  let y2 := evalBarMzAt bar m z2 r
  let yLin := vecAdd (vecScale ρ1 y1) (vecScale ρ2 y2)
  ct yLin = ρ1 * ct y1 + ρ2 * ct y2 := by
  exact And.right hProp.2.2

theorem evalHom2Prop_eval_eq_explicit
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z1 z2 r : Array F}
  {ρ1 ρ2 : F}
  (hProp : evalHom2Prop bar m z1 z2 r ρ1 ρ2) :
  vecAdd (vecScale ρ1 (evalBarMzAt bar m z1 r))
    (vecScale ρ2 (evalBarMzAt bar m z2 r))
    = evalBarMzAt bar m (linComb2Vec ρ1 ρ2 z1 z2) r := by
  simpa using
    (evalHom2Prop_eval_eq
      (bar := bar) (m := m) (z1 := z1) (z2 := z2) (r := r) (ρ1 := ρ1) (ρ2 := ρ2) hProp)

theorem evalHom2Prop_ct_eq_explicit
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z1 z2 r : Array F}
  {ρ1 ρ2 : F}
  (hProp : evalHom2Prop bar m z1 z2 r ρ1 ρ2) :
  ct (vecAdd (vecScale ρ1 (evalBarMzAt bar m z1 r))
      (vecScale ρ2 (evalBarMzAt bar m z2 r)))
    = ρ1 * ct (evalBarMzAt bar m z1 r) + ρ2 * ct (evalBarMzAt bar m z2 r) := by
  simpa using
    (evalHom2Prop_ct_eq
      (bar := bar) (m := m) (z1 := z1) (z2 := z2) (r := r) (ρ1 := ρ1) (ρ2 := ρ2) hProp)

theorem evalHom2Prop_ct_eq_direct_explicit
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z1 z2 r : Array F}
  {ρ1 ρ2 : F}
  (hProp : evalHom2Prop bar m z1 z2 r ρ1 ρ2) :
  ct (evalBarMzAt bar m (linComb2Vec ρ1 ρ2 z1 z2) r)
    = ρ1 * ct (evalBarMzAt bar m z1 r) + ρ2 * ct (evalBarMzAt bar m z2 r) := by
  have hEq :
      vecAdd (vecScale ρ1 (evalBarMzAt bar m z1 r))
        (vecScale ρ2 (evalBarMzAt bar m z2 r))
        = evalBarMzAt bar m (linComb2Vec ρ1 ρ2 z1 z2) r := by
    simpa using
      (evalHom2Prop_eval_eq
        (bar := bar) (m := m) (z1 := z1) (z2 := z2) (r := r) (ρ1 := ρ1) (ρ2 := ρ2) hProp)
  have hCt :
      ct (vecAdd (vecScale ρ1 (evalBarMzAt bar m z1 r))
        (vecScale ρ2 (evalBarMzAt bar m z2 r)))
        = ρ1 * ct (evalBarMzAt bar m z1 r) + ρ2 * ct (evalBarMzAt bar m z2 r) := by
    simpa using
      (evalHom2Prop_ct_eq
        (bar := bar) (m := m) (z1 := z1) (z2 := z2) (r := r) (ρ1 := ρ1) (ρ2 := ρ2) hProp)
  have hCtEq :
      ct (vecAdd (vecScale ρ1 (evalBarMzAt bar m z1 r))
        (vecScale ρ2 (evalBarMzAt bar m z2 r)))
        = ct (evalBarMzAt bar m (linComb2Vec ρ1 ρ2 z1 z2) r) := by
    simpa using congrArg ct hEq
  calc
    ct (evalBarMzAt bar m (linComb2Vec ρ1 ρ2 z1 z2) r)
        = ct (vecAdd (vecScale ρ1 (evalBarMzAt bar m z1 r))
            (vecScale ρ2 (evalBarMzAt bar m z2 r))) := by
          symm
          exact hCtEq
    _ = ρ1 * ct (evalBarMzAt bar m z1 r) + ρ2 * ct (evalBarMzAt bar m z2 r) := hCt

theorem evalHom2_size_eq
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z1 z2 r : Array F}
  {ρ1 ρ2 : F}
  (hOk : evalHom2 bar m z1 z2 r ρ1 ρ2 = true) :
  z1.size = z2.size := by
  exact evalHom2Prop_size_eq
    (evalHom2_sound_full
      (bar := bar) (m := m) (z1 := z1) (z2 := z2) (r := r) (ρ1 := ρ1) (ρ2 := ρ2) hOk)

theorem evalHom2_iff_prop
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z1 z2 r : Array F}
  {ρ1 ρ2 : F} :
  evalHom2 bar m z1 z2 r ρ1 ρ2 = true ↔ evalHom2Prop bar m z1 z2 r ρ1 ρ2 := by
  constructor
  · exact evalHom2_sound_full
  · exact evalHom2_complete

private theorem mul_zero_F (x : F) : x * (0 : F) = 0 := by
  cases x with
  | mk xv =>
      change F.ofNat (xv * (F.ofNat 0).val) = F.ofNat 0
      simp [F.ofNat]

/-- `ct` commutes with `vecScale` at coefficient index `0`. -/
theorem ct_vecScale (s : F) (a : Array F) :
  ct (vecScale s a) = s * ct a := by
  by_cases h0 : a.size = 0
  · have hEq : a = #[] := Array.eq_empty_of_size_eq_zero h0
    subst hEq
    simp [ct, vecScale, mul_zero_F]
  · have hPos : 0 < a.size := Nat.pos_of_ne_zero h0
    have hAFalse : a.isEmpty = false := by
      apply eq_false_of_ne_true
      intro hT
      exact h0 ((Array.isEmpty_iff_size_eq_zero).1 hT)
    have hScalePos : 0 < (vecScale s a).size := by
      simpa [vecScale_size] using hPos
    have hScaleFalse : (vecScale s a).isEmpty = false := by
      apply eq_false_of_ne_true
      intro hT
      exact (Nat.ne_of_gt hScalePos) ((Array.isEmpty_iff_size_eq_zero).1 hT)
    rw [ct_of_not_isEmpty hScaleFalse, ct_of_not_isEmpty hAFalse]
    simp [vecScale, hPos]

/-- `ct` commutes with `vecAdd` when sizes are compatible. -/
theorem ct_vecAdd_of_size_eq
  {a b : Array F}
  (hSize : a.size = b.size) :
  ct (vecAdd a b) = ct a + ct b := by
  by_cases h0 : a.size = 0
  · have hEqA : a = #[] := Array.eq_empty_of_size_eq_zero h0
    have h0b : b.size = 0 := by simpa [hEqA] using hSize.symm
    have hEqB : b = #[] := Array.eq_empty_of_size_eq_zero h0b
    subst hEqA
    subst hEqB
    have h00 : (0 : F) + (0 : F) = 0 :=
      F.zero_add_of_canonical (a := (0 : F)) F.canonical_zero
    simp [ct, vecAdd, h00]
  · have hPos : 0 < a.size := Nat.pos_of_ne_zero h0
    have h0b : b.size ≠ 0 := by simpa [hSize] using h0
    have hPosB : 0 < b.size := Nat.pos_of_ne_zero h0b
    have hAFalse : a.isEmpty = false := by
      apply eq_false_of_ne_true
      intro hT
      exact h0 ((Array.isEmpty_iff_size_eq_zero).1 hT)
    have hBFalse : b.isEmpty = false := by
      apply eq_false_of_ne_true
      intro hT
      exact h0b ((Array.isEmpty_iff_size_eq_zero).1 hT)
    have hVsz : (vecAdd a b).size = a.size := vecAdd_size_of_eq hSize
    have hVPos : 0 < (vecAdd a b).size := by
      simpa [hVsz] using hPos
    have hVFalse : (vecAdd a b).isEmpty = false := by
      apply eq_false_of_ne_true
      intro hT
      exact (Nat.ne_of_gt hVPos) ((Array.isEmpty_iff_size_eq_zero).1 hT)
    rw [ct_of_not_isEmpty hVFalse, ct_of_not_isEmpty hAFalse, ct_of_not_isEmpty hBFalse]
    unfold vecAdd
    simp [hSize, hPosB]

end SuperNeo
