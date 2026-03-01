import SuperNeo.BarLift

/-!
Theorem-4 matrix-transform layer (compact scaffold).

Reading guide:
1. `matrixVecDirect` is the direct matrix-vector computation.
2. `matrixVecCtBar` is the bar-lifted side used for the transform identity.
3. `matrixTransformIdentity` is the executable check surface.
4. `matrixTransformAssumption` is the theorem-facing boundary used downstream.
5. `_of_assumption`, `_of_checkAssumption`, `_iff_...` are check/prop bridges.
-/

namespace SuperNeo

open F

/-- Dot product on field vectors with a size guard. -/
def dotVec (a b : Array F) : F :=
  if a.size != b.size then
    0
  else
    (List.range a.size).foldl (fun acc i => acc + a[i]! * b[i]!) 0

/-- Direct matrix-vector product (`Mz`) computed row-wise. -/
def matrixVecDirect (m : Array (Array F)) (z : Array F) : Array F :=
  m.map (fun row => dotVec row z)

/-- Bar-lifted matrix-vector side (`ct(bar(M)z)` in paper-facing naming). -/
def matrixVecCtBar (bar : Array (Array F)) (m : Array (Array F)) (z : Array F) : Array F :=
  m.map (fun row => dotVec (barLiftVector bar row) (barLiftVector bar z))

@[simp] theorem matrixVecDirect_size (m : Array (Array F)) (z : Array F) :
    (matrixVecDirect m z).size = m.size := by
  simp [matrixVecDirect]

@[simp] theorem matrixVecCtBar_size (bar : Array (Array F)) (m : Array (Array F)) (z : Array F) :
    (matrixVecCtBar bar m z).size = m.size := by
  simp [matrixVecCtBar]

/-- Row-shape compatibility predicate used by matrix-transform statements. -/
def MatrixRowsCompatible (m : Array (Array F)) (z : Array F) : Prop :=
  ∀ i (hi : i < m.size), (m[i]'hi).size = z.size

instance matrixRowsCompatible_decidable (m : Array (Array F)) (z : Array F) :
    Decidable (MatrixRowsCompatible m z) := by
  unfold MatrixRowsCompatible
  infer_instance

/-- Check surface for Theorem 4 identity. -/
def matrixTransformIdentity (bar : Array (Array F)) (m : Array (Array F)) (z : Array F) : Bool :=
  if !(m.all (fun row => row.size = z.size)) then
    false
  else
    decide (matrixVecDirect m z = matrixVecCtBar bar m z)

/-- Proposition-level counterpart of `matrixTransformIdentity`. -/
def matrixTransformIdentityProp (bar : Array (Array F)) (m : Array (Array F)) (z : Array F) : Prop :=
  MatrixRowsCompatible m z ∧ matrixVecDirect m z = matrixVecCtBar bar m z

/--
Theorem-facing transform contract: for any shape-compatible `z`,
the direct and bar-lifted sides agree.
-/
def matrixTransformAssumption (bar : Array (Array F)) (m : Array (Array F)) : Prop :=
  ∀ z : Array F, MatrixRowsCompatible m z →
    matrixVecDirect m z = matrixVecCtBar bar m z

/--
Check-facing transform contract retained for executable compatibility.
-/
def matrixTransformCheckAssumption (bar : Array (Array F)) (m : Array (Array F)) : Prop :=
  ∀ z : Array F, MatrixRowsCompatible m z → matrixTransformIdentity bar m z = true

private theorem matrixRowsCompatible_of_all
  {m : Array (Array F)} {z : Array F}
  (hAll : m.all (fun row => row.size = z.size) = true) :
  MatrixRowsCompatible m z := by
  intro i hi
  have hDec : decide ((m[i]'hi).size = z.size) = true :=
    (Array.all_eq_true.mp hAll) i hi
  exact decide_eq_true_eq.mp hDec

private theorem all_true_of_matrixRowsCompatible
  {m : Array (Array F)} {z : Array F}
  (hRows : MatrixRowsCompatible m z) :
  m.all (fun row => row.size = z.size) = true := by
  apply (Array.all_eq_true).2
  intro i hi
  exact decide_eq_true (hRows i hi)

/-- Native theorem-4 identity for this compact scaffold (`barLiftVector = id`). -/
theorem matrixTransformEq_native
  {bar : Array (Array F)} {m : Array (Array F)} {z : Array F}
  (_hRows : MatrixRowsCompatible m z) :
  matrixVecDirect m z = matrixVecCtBar bar m z := by
  simp [matrixVecDirect, matrixVecCtBar, barLiftVector]

theorem matrixTransformIdentity_sound
  {bar : Array (Array F)} {m : Array (Array F)} {z : Array F}
  (hOk : matrixTransformIdentity bar m z = true) :
  matrixTransformIdentityProp bar m z := by
  unfold matrixTransformIdentity at hOk
  simp at hOk
  exact ⟨hOk.1, hOk.2⟩

theorem matrixTransformIdentity_complete
  {bar : Array (Array F)} {m : Array (Array F)} {z : Array F}
  (hProp : matrixTransformIdentityProp bar m z) :
  matrixTransformIdentity bar m z = true := by
  rcases hProp with ⟨hRows, hEq⟩
  unfold matrixTransformIdentity
  simp [all_true_of_matrixRowsCompatible hRows, decide_eq_true hEq]

theorem matrixTransformIdentity_iff_prop
  {bar : Array (Array F)} {m : Array (Array F)} {z : Array F} :
  matrixTransformIdentity bar m z = true ↔ matrixTransformIdentityProp bar m z := by
  constructor
  · exact matrixTransformIdentity_sound
  · exact matrixTransformIdentity_complete

theorem matrixTransformAssumption_native
  {bar : Array (Array F)} {m : Array (Array F)} :
  matrixTransformAssumption bar m := by
  intro z hRows
  exact matrixTransformEq_native hRows

/-- Convert theorem-facing transform contract into the check-facing form. -/
theorem matrixTransformCheckAssumption_of_assumption
  {bar : Array (Array F)} {m : Array (Array F)}
  (hAssm : matrixTransformAssumption bar m) :
  matrixTransformCheckAssumption bar m := by
  intro z hRows
  exact matrixTransformIdentity_complete ⟨hRows, hAssm z hRows⟩

/-- Convert check-facing transform contract into the theorem-facing form. -/
theorem matrixTransformAssumption_of_checkAssumption
  {bar : Array (Array F)} {m : Array (Array F)}
  (hCheck : matrixTransformCheckAssumption bar m) :
  matrixTransformAssumption bar m := by
  intro z hRows
  exact (matrixTransformIdentity_sound (hCheck z hRows)).2

/-- Equivalence between theorem-facing and check-facing transform contracts. -/
theorem matrixTransformAssumption_iff_checkAssumption
  {bar : Array (Array F)} {m : Array (Array F)} :
  matrixTransformAssumption bar m ↔ matrixTransformCheckAssumption bar m := by
  constructor
  · exact matrixTransformCheckAssumption_of_assumption
  · exact matrixTransformAssumption_of_checkAssumption


end SuperNeo
