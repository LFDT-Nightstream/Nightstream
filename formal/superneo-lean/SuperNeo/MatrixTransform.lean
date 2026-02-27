import SuperNeo.BarLift
import SuperNeo.Thm3Core

/-! Matrix-transform identities and theorem/check interfaces (P12). -/


namespace SuperNeo

open F

private def sumBlocks (n : Nat) (f : Nat → F) : F :=
  Nat.rec (motive := fun _ => F) 0 (fun t acc => acc + f t) n

private theorem sumBlocks_congr
  {n : Nat} {f g : Nat → F}
  (hEq : ∀ t, t < n → f t = g t) :
  sumBlocks n f = sumBlocks n g := by
  induction n generalizing f g with
  | zero =>
      rfl
  | succ n ih =>
      have hInit : sumBlocks n f = sumBlocks n g := ih (by
        intro t ht
        exact hEq t (Nat.lt_trans ht (Nat.lt_succ_self n)))
      have hLast : f n = g n := hEq n (Nat.lt_succ_self n)
      have hInit' := hInit
      unfold sumBlocks at hInit'
      unfold sumBlocks
      simp [hInit', hLast]

private def dotVec (a b : Array F) : F :=
  if a.size != b.size then
    0
  else
    Id.run do
      let mut acc : F := 0
      for i in [0:a.size] do
        acc := acc + a[i]! * b[i]!
      return acc

private def dotVecBlocks (a b : Array F) : F :=
  if a.size != b.size then
    0
  else if a.size % d != 0 then
    0
  else
    let nBlocks := a.size / d
    sumBlocks nBlocks (fun t =>
      let start := t * d
      let stop := start + d
      dot (a.extract start stop) (b.extract start stop))

private def rowCtBarProduct (bar : Array (Array F)) (row z : Array F) : F :=
  if row.size != z.size then
    0
  else if row.size % d != 0 then
    0
  else
    let nBlocks := row.size / d
    sumBlocks nBlocks (fun t =>
      let start := t * d
      let stop := start + d
      let aBlk := row.extract start stop
      let zBlk := z.extract start stop
      ct (mulRq (superneoBarBlock bar aBlk) zBlk))

/-- Direct field matrix-vector product Mz over rows. -/
def matrixVecDirect (m : Array (Array F)) (z : Array F) : Array F :=
  m.map (fun row => dotVecBlocks row z)

/-- ct(bar(M)z) row values via blockwise ring products. -/
def matrixVecCtBar (bar : Array (Array F)) (m : Array (Array F)) (z : Array F) : Array F :=
  m.map (fun row => rowCtBarProduct bar row z)

theorem matrixVecDirect_size (m : Array (Array F)) (z : Array F) :
  (matrixVecDirect m z).size = m.size := by
  unfold matrixVecDirect
  simp

theorem matrixVecCtBar_size (bar : Array (Array F)) (m : Array (Array F)) (z : Array F) :
  (matrixVecCtBar bar m z).size = m.size := by
  unfold matrixVecCtBar
  simp

/-- Theorem 4 computational check: Mz = ct(bar(M)z). -/
def matrixTransformIdentity (bar : Array (Array F)) (m : Array (Array F)) (z : Array F) : Bool :=
  if !(m.all (fun row => row.size = z.size ∧ row.size % d = 0)) then
    false
  else
    decide (matrixVecDirect m z = matrixVecCtBar bar m z)

/-- Proposition-level row-shape preconditions used by Theorem 4 wrappers. -/
def MatrixRowsCompatible (m : Array (Array F)) (z : Array F) : Prop :=
  ∀ i (hi : i < m.size), (m[i]'hi).size = z.size ∧ (m[i]'hi).size % d = 0

instance matrixRowsCompatible_decidable (m : Array (Array F)) (z : Array F) :
    Decidable (MatrixRowsCompatible m z) := by
  unfold MatrixRowsCompatible
  infer_instance

theorem matrixRowsCompatible_row_size
  {m : Array (Array F)} {z : Array F}
  (hRows : MatrixRowsCompatible m z)
  (i : Nat) (hi : i < m.size) :
  (m[i]'hi).size = z.size :=
  (hRows i hi).1

theorem matrixRowsCompatible_row_mod
  {m : Array (Array F)} {z : Array F}
  (hRows : MatrixRowsCompatible m z)
  (i : Nat) (hi : i < m.size) :
  (m[i]'hi).size % d = 0 :=
  (hRows i hi).2

theorem matrixRowsCompatible_of_size_eq
  {m : Array (Array F)} {z1 z2 : Array F}
  (hRows : MatrixRowsCompatible m z1)
  (hSize : z1.size = z2.size) :
  MatrixRowsCompatible m z2 := by
  intro i hi
  have hRow := hRows i hi
  refine ⟨?_, hRow.2⟩
  calc
    (m[i]'hi).size = z1.size := hRow.1
    _ = z2.size := hSize

/-- Proposition-level counterpart of `matrixTransformIdentity`. -/
def matrixTransformIdentityProp (bar : Array (Array F)) (m : Array (Array F)) (z : Array F) : Prop :=
  MatrixRowsCompatible m z ∧ matrixVecDirect m z = matrixVecCtBar bar m z

/--
Theorem-native assumption interface for P12/Theorem 4:
for row-shape-compatible vectors, `Mz = ct(bar(M)z)`.
-/
def p12MatrixTransformAssumption (bar : Array (Array F)) (m : Array (Array F)) : Prop :=
  ∀ z : Array F, MatrixRowsCompatible m z ->
    matrixVecDirect m z = matrixVecCtBar bar m z

/--
Check-oriented universal assumption interface for P12, conditioned on the same
row-shape preconditions as the theorem-native form.
-/
def p12MatrixTransformCheckAssumption (bar : Array (Array F)) (m : Array (Array F)) : Prop :=
  ∀ z : Array F, MatrixRowsCompatible m z ->
    matrixTransformIdentity bar m z = true

theorem matrixRowsCompatible_of_all
  {m : Array (Array F)} {z : Array F}
  (hAll : m.all (fun row => row.size = z.size ∧ row.size % d = 0) = true) :
  MatrixRowsCompatible m z := by
  intro i hi
  have hDec :
      decide ((m[i]'hi).size = z.size ∧ (m[i]'hi).size % d = 0) = true :=
    (Array.all_eq_true.mp hAll) i hi
  exact decide_eq_true_eq.mp hDec

theorem all_true_of_matrixRowsCompatible
  {m : Array (Array F)} {z : Array F}
  (hRows : MatrixRowsCompatible m z) :
  m.all (fun row => row.size = z.size ∧ row.size % d = 0) = true := by
  apply (Array.all_eq_true).2
  intro i hi
  exact decide_eq_true (hRows i hi)

theorem dotVec_eq_dot_of_isDVec
  {a b : Array F}
  (ha : a.size = d)
  (hb : b.size = d) :
  dotVec a b = dot a b := by
  unfold dotVec dot
  have hEqSz : a.size = b.size := by
    calc
      a.size = d := ha
      _ = b.size := hb.symm
  simp [hEqSz, hb, D_eq_d]

private theorem extract_size_eq_d_of_lt_div
  {v : Array F} {t : Nat}
  (hMod : v.size % d = 0)
  (ht : t < v.size / d) :
  (v.extract (t * d) (t * d + d)).size = d := by
  let start := t * d
  let stop := start + d
  have hDvd : d ∣ v.size := Nat.dvd_of_mod_eq_zero hMod
  have hSizeMul : (v.size / d) * d = v.size := Nat.div_mul_cancel hDvd
  have htLe : t + 1 ≤ v.size / d := Nat.succ_le_of_lt ht
  have hStop : stop ≤ v.size := by
    have hMul : (t + 1) * d ≤ (v.size / d) * d := Nat.mul_le_mul_right d htLe
    simpa [start, stop, Nat.succ_mul, hSizeMul] using hMul
  have hMin : Nat.min stop v.size = stop := Nat.min_eq_left hStop
  have hExtract :
      (v.extract start stop).size = Nat.min stop v.size - start := by
    simpa [start, stop] using
      (Array.size_extract (xs := v) (start := start) (stop := stop))
  calc
    (v.extract start stop).size = Nat.min stop v.size - start := hExtract
    _ = stop - start := by simp [hMin]
    _ = d := by simp [start, stop]

private theorem isDVec_extract_of_lt_div
  {v : Array F} {t : Nat}
  (hMod : v.size % d = 0)
  (ht : t < v.size / d) :
  IsDVec (v.extract (t * d) (t * d + d)) := by
  exact extract_size_eq_d_of_lt_div hMod ht

private theorem rowCtBarProduct_eq_dotVecBlocks_of_thm3CoreAssumption
  {bar : Array (Array F)} {row z : Array F}
  (hThm3 : thm3CoreAssumption bar)
  (hSize : row.size = z.size)
  (hMod : row.size % d = 0) :
  rowCtBarProduct bar row z = dotVecBlocks row z := by
  have hModZ : z.size % d = 0 := by
    simpa [hSize] using hMod
  unfold rowCtBarProduct dotVecBlocks
  simp [hSize, hModZ]
  apply sumBlocks_congr
  intro t ht
  let start := t * d
  let stop := start + d
  let aBlk := row.extract start stop
  let zBlk := z.extract start stop
  have htRow : t < row.size / d := by
    simpa [hSize] using ht
  have ha : IsDVec aBlk := by
    simpa [aBlk] using
      (isDVec_extract_of_lt_div (v := row) hMod htRow)
  have htZ : t < z.size / d := by
    simpa [hSize] using ht
  have hb : IsDVec zBlk := by
    simpa [zBlk] using
      (isDVec_extract_of_lt_div (v := z) hModZ htZ)
  simpa [p10CoreProp, aBlk, zBlk] using hThm3 ha hb

theorem matrixTransformEq_of_thm3CoreAssumption
  {bar : Array (Array F)} {m : Array (Array F)} {z : Array F}
  (hThm3 : thm3CoreAssumption bar)
  (hRows : MatrixRowsCompatible m z) :
  matrixVecDirect m z = matrixVecCtBar bar m z := by
  apply Array.ext
  · simp [matrixVecDirect, matrixVecCtBar]
  · intro i hiDirect hiCt
    have hi : i < m.size := by
      simpa [matrixVecDirect] using hiDirect
    have hSize : (m[i]'hi).size = z.size := matrixRowsCompatible_row_size hRows i hi
    have hMod : (m[i]'hi).size % d = 0 := matrixRowsCompatible_row_mod hRows i hi
    have hRowEq :
        dotVecBlocks (m[i]'hi) z = rowCtBarProduct bar (m[i]'hi) z := by
      exact (rowCtBarProduct_eq_dotVecBlocks_of_thm3CoreAssumption
        (bar := bar) (row := m[i]'hi) (z := z) hThm3 hSize hMod).symm
    simpa [matrixVecDirect, matrixVecCtBar] using hRowEq

theorem matrixTransformIdentity_sound
  {bar : Array (Array F)} {m : Array (Array F)} {z : Array F}
  (hOk : matrixTransformIdentity bar m z = true) :
  matrixVecDirect m z = matrixVecCtBar bar m z := by
  unfold matrixTransformIdentity at hOk
  simp at hOk
  exact hOk.2

theorem matrixTransformIdentity_sound_full
  {bar : Array (Array F)} {m : Array (Array F)} {z : Array F}
  (hOk : matrixTransformIdentity bar m z = true) :
  MatrixRowsCompatible m z ∧
    matrixVecDirect m z = matrixVecCtBar bar m z := by
  unfold matrixTransformIdentity at hOk
  simp at hOk
  exact hOk

theorem matrixTransformIdentity_sound_prop
  {bar : Array (Array F)} {m : Array (Array F)} {z : Array F}
  (hOk : matrixTransformIdentity bar m z = true) :
  matrixTransformIdentityProp bar m z := by
  exact matrixTransformIdentity_sound_full hOk

theorem matrixTransformIdentity_rows_guard
  {bar : Array (Array F)} {m : Array (Array F)} {z : Array F}
  (hOk : matrixTransformIdentity bar m z = true) :
  m.all (fun row => row.size = z.size ∧ row.size % d = 0) = true := by
  exact all_true_of_matrixRowsCompatible (matrixTransformIdentity_sound_full hOk).1

theorem matrixTransformIdentity_complete
  {bar : Array (Array F)} {m : Array (Array F)} {z : Array F}
  (hRows : m.all (fun row => row.size = z.size ∧ row.size % d = 0) = true)
  (hEq : matrixVecDirect m z = matrixVecCtBar bar m z) :
  matrixTransformIdentity bar m z = true := by
  unfold matrixTransformIdentity
  cases hAll : m.all (fun row => row.size = z.size ∧ row.size % d = 0)
  · have hContra : False := by
      rw [hAll] at hRows
      cases hRows
    exact False.elim hContra
  · simp [decide_eq_true hEq]

theorem matrixTransformIdentity_complete_of_rowsCompatible
  {bar : Array (Array F)} {m : Array (Array F)} {z : Array F}
  (hRows : MatrixRowsCompatible m z)
  (hEq : matrixVecDirect m z = matrixVecCtBar bar m z) :
  matrixTransformIdentity bar m z = true := by
  exact matrixTransformIdentity_complete (all_true_of_matrixRowsCompatible hRows) hEq

theorem p12MatrixTransformAssumption_of_thm3CoreAssumption
  {bar : Array (Array F)} {m : Array (Array F)}
  (hThm3 : thm3CoreAssumption bar) :
  p12MatrixTransformAssumption bar m := by
  intro z hRows
  exact matrixTransformEq_of_thm3CoreAssumption hThm3 hRows

theorem p12MatrixTransformAssumption_of_thm3_and_p11
  {bar : Array (Array F)} {m : Array (Array F)}
  (hThm3 : thm3CoreAssumption bar)
  (_hP11Add : p11AdditivityAssumption bar)
  (_hP11Scale : p11HomogeneityAssumption bar) :
  p12MatrixTransformAssumption bar m := by
  exact p12MatrixTransformAssumption_of_thm3CoreAssumption hThm3

theorem p12MatrixTransformCheckAssumption_of_thm3CoreAssumption
  {bar : Array (Array F)} {m : Array (Array F)}
  (hThm3 : thm3CoreAssumption bar) :
  p12MatrixTransformCheckAssumption bar m := by
  intro z hRows
  exact matrixTransformIdentity_complete_of_rowsCompatible hRows
    (matrixTransformEq_of_thm3CoreAssumption hThm3 hRows)

theorem p12MatrixTransformCheckAssumption_of_thm3_and_p11
  {bar : Array (Array F)} {m : Array (Array F)}
  (hThm3 : thm3CoreAssumption bar)
  (_hP11Add : p11AdditivityAssumption bar)
  (_hP11Scale : p11HomogeneityAssumption bar) :
  p12MatrixTransformCheckAssumption bar m := by
  exact p12MatrixTransformCheckAssumption_of_thm3CoreAssumption hThm3

theorem matrixTransformIdentity_complete_of_prop
  {bar : Array (Array F)} {m : Array (Array F)} {z : Array F}
  (hProp : matrixTransformIdentityProp bar m z) :
  matrixTransformIdentity bar m z = true := by
  exact matrixTransformIdentity_complete_of_rowsCompatible hProp.1 hProp.2

theorem matrixTransformIdentity_iff_prop
  {bar : Array (Array F)} {m : Array (Array F)} {z : Array F} :
  matrixTransformIdentity bar m z = true ↔ matrixTransformIdentityProp bar m z := by
  constructor
  · exact matrixTransformIdentity_sound_prop
  · exact matrixTransformIdentity_complete_of_prop

theorem matrixTransformEq_of_assumption
  {bar : Array (Array F)} {m : Array (Array F)} {z : Array F}
  (hAssm : p12MatrixTransformAssumption bar m)
  (hRows : MatrixRowsCompatible m z) :
  matrixVecDirect m z = matrixVecCtBar bar m z := by
  exact hAssm z hRows

theorem matrixTransformIdentity_true_of_assumption
  {bar : Array (Array F)} {m : Array (Array F)} {z : Array F}
  (hAssm : p12MatrixTransformAssumption bar m)
  (hRows : MatrixRowsCompatible m z) :
  matrixTransformIdentity bar m z = true := by
  exact matrixTransformIdentity_complete_of_rowsCompatible hRows (hAssm z hRows)

theorem p12MatrixTransformAssumption_of_checkAssumption
  {bar : Array (Array F)} {m : Array (Array F)}
  (hCheck : p12MatrixTransformCheckAssumption bar m) :
  p12MatrixTransformAssumption bar m := by
  intro z hRows
  exact (matrixTransformIdentity_sound_full (hCheck z hRows)).2

theorem p12MatrixTransformCheckAssumption_of_assumption
  {bar : Array (Array F)} {m : Array (Array F)}
  (hAssm : p12MatrixTransformAssumption bar m) :
  p12MatrixTransformCheckAssumption bar m := by
  intro z hRows
  exact matrixTransformIdentity_true_of_assumption hAssm hRows

end SuperNeo
