import SuperNeo.BarLift

namespace SuperNeo

open F

private def dotVec (a b : Array F) : F :=
  if a.size != b.size then
    0
  else
    Id.run do
      let mut acc : F := 0
      for i in [0:a.size] do
        acc := acc + a[i]! * b[i]!
      return acc

private def rowCtBarProduct (bar : Array (Array F)) (row z : Array F) : F :=
  if row.size != z.size then
    0
  else if row.size % d != 0 then
    0
  else
    let nBlocks := row.size / d
    Id.run do
      let mut acc : F := 0
      for t in [0:nBlocks] do
        let start := t * d
        let stop := start + d
        let aBlk := row.extract start stop
        let zBlk := z.extract start stop
        let term := ct (mulRq (superneoBarBlock bar aBlk) zBlk)
        acc := acc + term
      return acc

/-- Direct field matrix-vector product Mz over rows. -/
def matrixVecDirect (m : Array (Array F)) (z : Array F) : Array F :=
  m.map (fun row => dotVec row z)

/-- ct(bar(M)z) row values via blockwise ring products. -/
def matrixVecCtBar (bar : Array (Array F)) (m : Array (Array F)) (z : Array F) : Array F :=
  m.map (fun row => rowCtBarProduct bar row z)

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

end SuperNeo
