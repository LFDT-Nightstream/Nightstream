import SuperNeo.EvalLink
import SuperNeo.ModuleHom
import SuperNeo.MLE

/-!
Evaluation homomorphism layer (Theorem 5 / P14 core).

Reading guide:
1. `evalHom2` is the executable check surface.
2. `evalHom2Prop` is the proposition being checked.
3. `evalHomAssumption` is the theorem-facing boundary.
4. `_of_assumption`, `_of_checkAssumption`, and `_iff_...` bridge check/prop.
-/

namespace SuperNeo

open F
local instance : NeZero Goldilocks.q := ⟨Nat.ne_of_gt Goldilocks.q_pos⟩

/--
Evaluator at challenge point `r` over the `y`-space.

In this theorem-facing scaffold, `evalBarMzAt` is the multilinear evaluator on
its vector input; Theorem 5 then states this evaluation is linear under
`linComb2Vec`.
-/
def evalBarMzAt
  (_bar : Array (Array F))
  (_m : Array (Array F))
  (z r : Array F) : F :=
  mleInnerProductForm z r

private theorem f_add_zero (a : F) : a + 0 = a := by
  simpa using (Lean.Grind.Fin.add_zero (n := Goldilocks.q) a)

private theorem f_zero_add (a : F) : 0 + a = a := by
  simpa using (Lean.Grind.Fin.zero_add (n := Goldilocks.q) a)

private theorem f_zero_mul (a : F) : (0 : F) * a = 0 := by
  simpa using (Lean.Grind.Fin.zero_mul (n := Goldilocks.q) a)

private theorem foldl_eq_init_of_step_eq
    {α β : Type}
    (l : List α)
    (init : β)
    (step : β → α → β)
    (hStep : ∀ acc x, x ∈ l → step acc x = acc) :
    l.foldl step init = init := by
  induction l generalizing init with
  | nil =>
      rfl
  | cons x xs ih =>
      have hHead : step init x = init := hStep init x (by simp)
      calc
        (x :: xs).foldl step init
            = xs.foldl step (step init x) := by rfl
        _ = xs.foldl step init := by rw [hHead]
        _ = init := by
              apply ih
              intro acc y hy
              exact hStep acc y (by simp [hy])

private theorem linComb_zero_left_eq_vecScale
    (δ : F) (v : Array F) :
    linComb δ (Array.replicate v.size 0) v (by simp) = vecScale δ v := by
  apply Array.ext
  · simp [linComb, vecScale]
  · intro i hiL hiR
    have hi : i < v.size := by
      simpa [vecScale] using hiR
    simp [linComb, vecScale, hi]

private theorem mleInnerProductForm_zero
    (n : Nat) (r : Array F) :
    mleInnerProductForm (Array.replicate n (0 : F)) r = 0 := by
  unfold mleInnerProductForm
  apply foldl_eq_init_of_step_eq
  intro acc i hi
  have hiRep : i < (Array.replicate n (0 : F)).size := List.mem_range.mp hi
  have hiN : i < n := by simpa using hiRep
  have hZero : (Array.replicate n (0 : F))[i]! = 0 := by
    simp [hiN]
  calc
    acc + (Array.replicate n (0 : F))[i]! * eqPoly (bitsToFieldArray r.size i) r
        = acc + 0 * eqPoly (bitsToFieldArray r.size i) r := by simp [hZero]
    _ = acc + 0 := by simp [f_zero_mul]
    _ = acc := f_add_zero acc

private theorem mleInnerProductForm_vecScale
    (s : F) (v r : Array F) :
    mleInnerProductForm (vecScale s v) r = s * mleInnerProductForm v r := by
  let z : Array F := Array.replicate v.size 0
  have hSize : z.size = v.size := by simp [z]
  have hLin :
      mleInnerProductForm (linComb s z v hSize) r =
        mleInnerProductForm z r + s * mleInnerProductForm v r :=
    mleInnerProductLinearityAssumption_holds s z v r hSize
  have hZero : mleInnerProductForm z r = 0 := by
    simpa [z] using mleInnerProductForm_zero v.size r
  have hComb : linComb s z v hSize = vecScale s v := by
    simpa [z] using linComb_zero_left_eq_vecScale s v
  calc
    mleInnerProductForm (vecScale s v) r
        = mleInnerProductForm (linComb s z v hSize) r := by rw [← hComb]
    _ = mleInnerProductForm z r + s * mleInnerProductForm v r := hLin
    _ = 0 + s * mleInnerProductForm v r := by rw [hZero]
    _ = s * mleInnerProductForm v r := f_zero_add _

private theorem linComb2Vec_eq_linComb
    (ρ1 ρ2 : F) (u v : Array F) (huv : u.size = v.size) :
    linComb2Vec ρ1 ρ2 u v =
      linComb ρ2 (vecScale ρ1 u) v (by simpa [vecScale_size] using huv) := by
  apply Array.ext
  · simp [linComb2Vec, linComb, vecScale, vecScale_size, vecAdd_size_of_eq, huv]
  · intro i hiL hiR
    have hiU : i < u.size := by
      simpa [linComb2Vec, vecScale_size, vecAdd_size_of_eq, huv] using hiL
    have hiV : i < v.size := by simpa [huv] using hiU
    simp [linComb2Vec, linComb, vecAdd, vecScale, huv, hiU, hiV]

private theorem evalBarMzAt_linComb2Vec
    (bar m : Array (Array F))
    (ρ1 ρ2 : F)
    (u v r : Array F)
    (huv : u.size = v.size) :
    evalBarMzAt bar m (linComb2Vec ρ1 ρ2 u v) r =
      ρ1 * evalBarMzAt bar m u r + ρ2 * evalBarMzAt bar m v r := by
  have hSizeScaled : (vecScale ρ1 u).size = v.size := by
    simpa [vecScale_size] using huv
  have hComb :
      linComb2Vec ρ1 ρ2 u v =
        linComb ρ2 (vecScale ρ1 u) v hSizeScaled := by
    exact linComb2Vec_eq_linComb ρ1 ρ2 u v huv
  have hLin :
      mleInnerProductForm (linComb ρ2 (vecScale ρ1 u) v hSizeScaled) r =
        mleInnerProductForm (vecScale ρ1 u) r + ρ2 * mleInnerProductForm v r :=
    mleInnerProductLinearityAssumption_holds ρ2 (vecScale ρ1 u) v r hSizeScaled
  have hScale :
      mleInnerProductForm (vecScale ρ1 u) r = ρ1 * mleInnerProductForm u r :=
    mleInnerProductForm_vecScale ρ1 u r
  unfold evalBarMzAt
  calc
    mleInnerProductForm (linComb2Vec ρ1 ρ2 u v) r
        = mleInnerProductForm (linComb ρ2 (vecScale ρ1 u) v hSizeScaled) r := by
            rw [hComb]
    _ = mleInnerProductForm (vecScale ρ1 u) r + ρ2 * mleInnerProductForm v r := hLin
    _ = ρ1 * mleInnerProductForm u r + ρ2 * mleInnerProductForm v r := by
          rw [hScale]

/-- Proposition-level statement checked by `evalHom2`. -/
def evalHom2Prop
  (bar : Array (Array F))
  (m : Array (Array F))
  (z1 z2 r : Array F)
  (ρ1 ρ2 : F) : Prop :=
  z1.size = z2.size ∧
  MatrixRowsCompatible m z1 ∧
  evalBarMzAt bar m
    (linComb2Vec ρ1 ρ2 (matrixVecCtBar bar m z1) (matrixVecCtBar bar m z2))
    r
    =
  ρ1 * evalBarMzAt bar m (matrixVecCtBar bar m z1) r
    + ρ2 * evalBarMzAt bar m (matrixVecCtBar bar m z2) r

/-- Theorem 5 check surface. -/
def evalHom2
  (bar : Array (Array F))
  (m : Array (Array F))
  (z1 z2 r : Array F)
  (ρ1 ρ2 : F) : Bool :=
  if z1.size != z2.size then
    false
  else if MatrixRowsCompatible m z1 then
    decide
      (evalBarMzAt bar m
          (linComb2Vec ρ1 ρ2 (matrixVecCtBar bar m z1) (matrixVecCtBar bar m z2))
          r
        =
       ρ1 * evalBarMzAt bar m (matrixVecCtBar bar m z1) r
        + ρ2 * evalBarMzAt bar m (matrixVecCtBar bar m z2) r)
  else
    false

/--
Theorem-facing evaluation-hom contract.

Downstream proof files should depend on this Prop boundary directly.
-/
def evalHomAssumption
  (bar : Array (Array F))
  (m : Array (Array F))
  (r : Array F)
  (ρ1 ρ2 : F) : Prop :=
  ∀ z1 z2 : Array F, z1.size = z2.size → MatrixRowsCompatible m z1 →
    evalHom2Prop bar m z1 z2 r ρ1 ρ2

/-- Check-facing evaluation-hom contract retained for compatibility. -/
def evalHomCheckAssumption
  (bar : Array (Array F))
  (m : Array (Array F))
  (r : Array F)
  (ρ1 ρ2 : F) : Prop :=
  ∀ z1 z2 : Array F, z1.size = z2.size → MatrixRowsCompatible m z1 →
    evalHom2 bar m z1 z2 r ρ1 ρ2 = true

theorem evalHom2_sound
  {bar : Array (Array F)} {m : Array (Array F)}
  {z1 z2 r : Array F} {ρ1 ρ2 : F}
  (hOk : evalHom2 bar m z1 z2 r ρ1 ρ2 = true) :
  evalHom2Prop bar m z1 z2 r ρ1 ρ2 := by
  unfold evalHom2Prop
  unfold evalHom2 at hOk
  by_cases hSize : z1.size != z2.size
  · simp [hSize] at hOk
  · have hEq : z1.size = z2.size := by simpa using hSize
    by_cases hRows : MatrixRowsCompatible m z1
    · have hEval :
        evalBarMzAt bar m
            (linComb2Vec ρ1 ρ2 (matrixVecCtBar bar m z1) (matrixVecCtBar bar m z2))
            r
          =
        ρ1 * evalBarMzAt bar m (matrixVecCtBar bar m z1) r
          + ρ2 * evalBarMzAt bar m (matrixVecCtBar bar m z2) r := by
          exact decide_eq_true_eq.mp (by simpa [hSize, hRows] using hOk)
      exact ⟨hEq, hRows, hEval⟩
    · simp [hSize, hRows] at hOk

theorem evalHom2_complete
  {bar : Array (Array F)} {m : Array (Array F)}
  {z1 z2 r : Array F} {ρ1 ρ2 : F}
  (hProp : evalHom2Prop bar m z1 z2 r ρ1 ρ2) :
  evalHom2 bar m z1 z2 r ρ1 ρ2 = true := by
  rcases hProp with ⟨hSize, hRows, hEval⟩
  unfold evalHom2
  have hSizeNe : ¬ z1.size != z2.size := by
    simpa [hSize]
  have hEval' :
      evalBarMzAt bar m
          (vecAdd (vecScale ρ1 (matrixVecCtBar bar m z1)) (vecScale ρ2 (matrixVecCtBar bar m z2)))
          r
        =
      ρ1 * evalBarMzAt bar m (matrixVecCtBar bar m z1) r
        + ρ2 * evalBarMzAt bar m (matrixVecCtBar bar m z2) r := by
    simpa [linComb2Vec] using hEval
  have hDec :
      decide
        (evalBarMzAt bar m
            (vecAdd (vecScale ρ1 (matrixVecCtBar bar m z1)) (vecScale ρ2 (matrixVecCtBar bar m z2)))
            r
          =
         ρ1 * evalBarMzAt bar m (matrixVecCtBar bar m z1) r
          + ρ2 * evalBarMzAt bar m (matrixVecCtBar bar m z2) r) = true :=
    decide_eq_true hEval'
  simp [hSizeNe, hRows, hDec]

theorem evalHom2_iff_prop
  {bar : Array (Array F)} {m : Array (Array F)}
  {z1 z2 r : Array F} {ρ1 ρ2 : F} :
  evalHom2 bar m z1 z2 r ρ1 ρ2 = true ↔ evalHom2Prop bar m z1 z2 r ρ1 ρ2 := by
  constructor
  · exact evalHom2_sound
  · exact evalHom2_complete

/--
Constructive theorem-5 closure for the current evaluator surface.

This theorem is fully theorem-native: no check wrapper and no boundary assumption
is required to derive `evalHomAssumption`.
-/
theorem evalHomAssumption_constructive
  {bar : Array (Array F)} {m : Array (Array F)} {r : Array F} {ρ1 ρ2 : F} :
  evalHomAssumption bar m r ρ1 ρ2 := by
  intro z1 z2 hSize hRows
  have hYSize : (matrixVecCtBar bar m z1).size = (matrixVecCtBar bar m z2).size := by
    simp [matrixVecCtBar_size]
  have hEval :
      evalBarMzAt bar m
          (linComb2Vec ρ1 ρ2 (matrixVecCtBar bar m z1) (matrixVecCtBar bar m z2))
          r
        =
      ρ1 * evalBarMzAt bar m (matrixVecCtBar bar m z1) r
        + ρ2 * evalBarMzAt bar m (matrixVecCtBar bar m z2) r :=
    evalBarMzAt_linComb2Vec bar m ρ1 ρ2 (matrixVecCtBar bar m z1) (matrixVecCtBar bar m z2) r hYSize
  exact ⟨hSize, hRows, hEval⟩

/-- Backward-compatible alias for the constructive theorem-5 closure. -/
theorem evalHomAssumption_native
  {bar : Array (Array F)} {m : Array (Array F)} {r : Array F} {ρ1 ρ2 : F} :
  evalHomAssumption bar m r ρ1 ρ2 := by
  exact evalHomAssumption_constructive

/-- Convert check-facing eval-hom contract into theorem-facing form. -/
theorem evalHomAssumption_of_checkAssumption
  {bar : Array (Array F)} {m : Array (Array F)}
  {r : Array F} {ρ1 ρ2 : F}
  (hCheck : evalHomCheckAssumption bar m r ρ1 ρ2) :
  evalHomAssumption bar m r ρ1 ρ2 := by
  intro z1 z2 hSize hRows
  exact evalHom2_sound (hCheck z1 z2 hSize hRows)

/-- Convert theorem-facing eval-hom contract into check-facing form. -/
theorem evalHomCheckAssumption_of_assumption
  {bar : Array (Array F)} {m : Array (Array F)}
  {r : Array F} {ρ1 ρ2 : F}
  (hAssm : evalHomAssumption bar m r ρ1 ρ2) :
  evalHomCheckAssumption bar m r ρ1 ρ2 := by
  intro z1 z2 hSize hRows
  exact evalHom2_complete (hAssm z1 z2 hSize hRows)

/-- Equivalence between theorem-facing and check-facing eval-hom contracts. -/
theorem evalHomAssumption_iff_checkAssumption
  {bar : Array (Array F)} {m : Array (Array F)}
  {r : Array F} {ρ1 ρ2 : F} :
  evalHomAssumption bar m r ρ1 ρ2 ↔
    evalHomCheckAssumption bar m r ρ1 ρ2 := by
  constructor
  · exact evalHomCheckAssumption_of_assumption
  · exact evalHomAssumption_of_checkAssumption

/--
Derive the evaluation-hom assumption from eval-link and module-hom assumptions.

In the compact scaffold, `evalHom2Prop` is shape-centric, so this theorem threads
the intended dependency chain while keeping obligations explicit.
-/
theorem evalHomAssumption_of_evalLink_and_moduleAssumptions
  {bar : Array (Array F)} {m : Array (Array F)}
  {r : Array F} {ρ1 ρ2 : F}
  {hVec : VecModuleHom} {hScal : ScalarModuleHom}
  (hEvalLink : evalLinkAssumption bar m)
  (_hVec : vecModuleAssumption hVec)
  (_hScal : scalarModuleAssumption hScal) :
  evalHomAssumption bar m r ρ1 ρ2 := by
  have _hLink := hEvalLink
  exact evalHomAssumption_constructive

/--
Theorem-native `P14` constructor from Theorem-3 (`P10`) and module-hom boundaries.

This goes through `P12`/`P13` using theorem surfaces only.
-/
theorem evalHomAssumption_of_thm3_and_moduleAssumptions
  {bar : Array (Array F)} {m : Array (Array F)}
  {r : Array F} {ρ1 ρ2 : F}
  {hVec : VecModuleHom} {hScal : ScalarModuleHom}
  (hThm3 : thm3CoreAssumption bar)
  (hVecAssm : vecModuleAssumption hVec)
  (hScalAssm : scalarModuleAssumption hScal) :
  evalHomAssumption bar m r ρ1 ρ2 := by
  exact evalHomAssumption_of_evalLink_and_moduleAssumptions
    (hEvalLink := evalLinkAssumption_of_thm3CoreAssumption hThm3)
    hVecAssm hScalAssm

/-- Theorem-native `P14` constructor from `P10` and module-hom boundaries. -/
theorem evalHomAssumption_of_p10_and_moduleAssumptions
  {bar : Array (Array F)} {m : Array (Array F)}
  {r : Array F} {ρ1 ρ2 : F}
  {hVec : VecModuleHom} {hScal : ScalarModuleHom}
  (hThm3 : thm3CoreAssumption bar)
  (hVecAssm : vecModuleAssumption hVec)
  (hScalAssm : scalarModuleAssumption hScal) :
  evalHomAssumption bar m r ρ1 ρ2 := by
  exact evalHomAssumption_of_evalLink_and_moduleAssumptions
    (hEvalLink := evalLinkAssumption_of_p10 hThm3)
    hVecAssm hScalAssm

/--
Theorem-native `P14` constructor from `(P10 + P11)` and module-hom boundaries.

This is the preferred dependency-accounted path for protocol composition.
-/
theorem evalHomAssumption_of_p10_p11_and_moduleAssumptions
  {bar : Array (Array F)} {m : Array (Array F)}
  {r : Array F} {ρ1 ρ2 : F}
  {hVec : VecModuleHom} {hScal : ScalarModuleHom}
  (hThm3 : thm3CoreAssumption bar)
  (_hLift : barLiftLinearityAssumption bar)
  (hVecAssm : vecModuleAssumption hVec)
  (hScalAssm : scalarModuleAssumption hScal) :
  evalHomAssumption bar m r ρ1 ρ2 := by
  exact evalHomAssumption_of_p10_and_moduleAssumptions
    (hThm3 := hThm3)
    (hVecAssm := hVecAssm)
    (hScalAssm := hScalAssm)

/--
Native theorem-stack constructor for P14: use canonical native bar matrix and
avoid explicit `hThm3` threading.
-/
theorem evalHomAssumption_native_bar
  {m : Array (Array F)}
  {r : Array F} {ρ1 ρ2 : F}
  {hVec : VecModuleHom} {hScal : ScalarModuleHom}
  (hVecAssm : vecModuleAssumption hVec)
  (hScalAssm : scalarModuleAssumption hScal) :
  evalHomAssumption nativeBarMatrix m r ρ1 ρ2 := by
  exact evalHomAssumption_of_evalLink_and_moduleAssumptions
    (hEvalLink := evalLinkAssumption_native m)
    hVecAssm hScalAssm

/-- Rewrite helper for deriving native P14 under `bar = nativeBarMatrix`. -/
theorem evalHomAssumption_of_bar_eq_native
  {bar : Array (Array F)} {m : Array (Array F)}
  {r : Array F} {ρ1 ρ2 : F}
  {hVec : VecModuleHom} {hScal : ScalarModuleHom}
  (hBar : bar = nativeBarMatrix)
  (hVecAssm : vecModuleAssumption hVec)
  (hScalAssm : scalarModuleAssumption hScal) :
  evalHomAssumption bar m r ρ1 ρ2 := by
  subst hBar
  exact evalHomAssumption_native_bar
    (m := m) (r := r) (ρ1 := ρ1) (ρ2 := ρ2)
    hVecAssm hScalAssm

end SuperNeo
