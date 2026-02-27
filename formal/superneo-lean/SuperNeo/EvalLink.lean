import SuperNeo.MatrixTransform
import SuperNeo.MLE

/-! Links ring-vector evaluation with coefficient-row evaluation. -/


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

/-- `barMzRing` preserves the matrix row count. -/
theorem barMzRing_size (bar : Array (Array F)) (m : Array (Array F)) (z : Array F) :
  (barMzRing bar m z).size = m.size := by
  unfold barMzRing
  simp

/-- Coefficient-row view cf(ys)_ell of a ring vector ys. -/
def coeffRowsOfRingVec (ys : Array Coeffs) : Array (Array F) :=
  Array.ofFn (fun ell : Fin d => ys.map (fun yi => yi[ell.1]!))

/-- Coefficient-row view has exactly `d` rows. -/
theorem coeffRowsOfRingVec_size (ys : Array Coeffs) :
  (coeffRowsOfRingVec ys).size = d := by
  unfold coeffRowsOfRingVec
  simp

/-- Evaluate each coefficient row independently with the same weights. -/
def evalCoeffRows (rows : Array (Array F)) (weights : Array F) : Array F :=
  rows.map (fun row => dotF row weights)

/-- Evaluating rows preserves row count. -/
theorem evalCoeffRows_size (rows : Array (Array F)) (weights : Array F) :
  (evalCoeffRows rows weights).size = rows.size := by
  unfold evalCoeffRows
  simp

/-- Evaluate a ring-valued vector with scalar weights (inner product over rows). -/
def evalRingVec (ys : Array Coeffs) (weights : Array F) : Coeffs :=
  if ys.size != weights.size then
    #[]
  else
    evalCoeffRows (coeffRowsOfRingVec ys) weights

/-- Shape lemma for `evalRingVec` on size-compatible inputs. -/
theorem evalRingVec_size_of_size_eq
  {ys : Array Coeffs} {weights : Array F}
  (hSize : ys.size = weights.size) :
  (evalRingVec ys weights).size = d := by
  unfold evalRingVec
  simp [hSize, evalCoeffRows_size, coeffRowsOfRingVec_size]

/-- Shape lemma for `evalRingVec` on size-mismatched inputs. -/
theorem evalRingVec_size_of_size_ne
  {ys : Array Coeffs} {weights : Array F}
  (hSize : ys.size ≠ weights.size) :
  (evalRingVec ys weights).size = 0 := by
  unfold evalRingVec
  simp [hSize]

/-- Constant-term row projection ct(ys). -/
def ctRow (ys : Array Coeffs) : Array F := ys.map (fun yi => yi[0]!)

/-- Constant-term projection preserves row count. -/
theorem ctRow_size (ys : Array Coeffs) :
  (ctRow ys).size = ys.size := by
  unfold ctRow
  simp

/-- Remark 2 computational identity for an already-built ring vector ys. -/
def evalLinkIdentity (ys : Array Coeffs) (weights : Array F) : Bool :=
  if ys.size != weights.size then
    false
  else
    let y := evalRingVec ys weights
    let coeffSide := evalCoeffRows (coeffRowsOfRingVec ys) weights
    let ctSide := dotF (ctRow ys) weights
    decide (y = coeffSide ∧ ct y = ctSide)

/-- Proposition-level counterpart of `evalLinkIdentity`. -/
def evalLinkIdentityProp (ys : Array Coeffs) (weights : Array F) : Prop :=
  (ys.size != weights.size) = false ∧
    let y := evalRingVec ys weights
    let coeffSide := evalCoeffRows (coeffRowsOfRingVec ys) weights
    let ctSide := dotF (ctRow ys) weights
    y = coeffSide ∧ ct y = ctSide

/-- Remark 2 identity specialized to bar(Mz) with MLE-derived r_hat weights. -/
def evalLinkForMatrix (bar : Array (Array F)) (m : Array (Array F)) (z r : Array F) : Bool :=
  let ys := barMzRing bar m z
  let weights := rHat r ys.size
  evalLinkIdentity ys weights

/-- Proposition-level counterpart of `evalLinkForMatrix`. -/
def evalLinkForMatrixProp (bar : Array (Array F)) (m : Array (Array F)) (z r : Array F) : Prop :=
  evalLinkIdentityProp (barMzRing bar m z) (rHat r (barMzRing bar m z).size)

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

theorem evalLinkIdentity_sound_full
  {ys : Array Coeffs} {weights : Array F}
  (hOk : evalLinkIdentity ys weights = true) :
  evalLinkIdentityProp ys weights := by
  unfold evalLinkIdentity at hOk
  cases hsz : (ys.size != weights.size) with
  | true =>
      simp [hsz] at hOk
  | false =>
      simp [hsz] at hOk
      exact ⟨hsz, hOk⟩

theorem evalLinkIdentity_complete
  {ys : Array Coeffs} {weights : Array F}
  (hProp : evalLinkIdentityProp ys weights) :
  evalLinkIdentity ys weights = true := by
  rcases hProp with ⟨hsz, hEq⟩
  unfold evalLinkIdentity
  simp [hsz, decide_eq_true hEq]

theorem evalLinkIdentityProp_size_eq
  {ys : Array Coeffs} {weights : Array F}
  (hProp : evalLinkIdentityProp ys weights) :
  ys.size = weights.size := by
  have hSizeFalse : (ys.size != weights.size) = false := hProp.1
  by_cases hEq : ys.size = weights.size
  · exact hEq
  · have hNeTrue : (ys.size != weights.size) = true := by
      simp [hEq]
    rw [hNeTrue] at hSizeFalse
    cases hSizeFalse

theorem evalLinkIdentityProp_eval_eq
  {ys : Array Coeffs} {weights : Array F}
  (hProp : evalLinkIdentityProp ys weights) :
  let y := evalRingVec ys weights
  let coeffSide := evalCoeffRows (coeffRowsOfRingVec ys) weights
  y = coeffSide := by
  exact hProp.2.1

theorem evalLinkIdentityProp_ct_eq
  {ys : Array Coeffs} {weights : Array F}
  (hProp : evalLinkIdentityProp ys weights) :
  let y := evalRingVec ys weights
  let ctSide := dotF (ctRow ys) weights
  ct y = ctSide := by
  exact hProp.2.2

theorem evalLinkIdentityProp_of_size_eq
  {ys : Array Coeffs} {weights : Array F}
  (hSize : ys.size = weights.size) :
  evalLinkIdentityProp ys weights := by
  have hSizeBool : (ys.size != weights.size) = false := by simp [hSize]
  refine ⟨hSizeBool, ?_⟩
  unfold evalRingVec
  rw [hSizeBool]
  refine ⟨rfl, ?_⟩
  unfold ctRow evalCoeffRows coeffRowsOfRingVec ct
  simp [d]

theorem evalLinkIdentity_true_of_size_eq
  {ys : Array Coeffs} {weights : Array F}
  (hSize : ys.size = weights.size) :
  evalLinkIdentity ys weights = true := by
  exact evalLinkIdentity_complete (evalLinkIdentityProp_of_size_eq hSize)

theorem evalLinkIdentity_size_eq
  {ys : Array Coeffs} {weights : Array F}
  (hOk : evalLinkIdentity ys weights = true) :
  ys.size = weights.size := by
  exact evalLinkIdentityProp_size_eq (evalLinkIdentity_sound_full hOk)

theorem evalLinkIdentity_eval_eq
  {ys : Array Coeffs} {weights : Array F}
  (hOk : evalLinkIdentity ys weights = true) :
  let y := evalRingVec ys weights
  let coeffSide := evalCoeffRows (coeffRowsOfRingVec ys) weights
  y = coeffSide := by
  exact evalLinkIdentityProp_eval_eq (evalLinkIdentity_sound_full hOk)

theorem evalLinkIdentity_ct_eq
  {ys : Array Coeffs} {weights : Array F}
  (hOk : evalLinkIdentity ys weights = true) :
  let y := evalRingVec ys weights
  let ctSide := dotF (ctRow ys) weights
  ct y = ctSide := by
  exact evalLinkIdentityProp_ct_eq (evalLinkIdentity_sound_full hOk)

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

theorem evalLinkForMatrix_sound_full
  {bar : Array (Array F)} {m : Array (Array F)} {z r : Array F}
  (hOk : evalLinkForMatrix bar m z r = true) :
  evalLinkForMatrixProp bar m z r := by
  unfold evalLinkForMatrixProp
  unfold evalLinkForMatrix at hOk
  exact evalLinkIdentity_sound_full hOk

theorem evalLinkForMatrix_complete
  {bar : Array (Array F)} {m : Array (Array F)} {z r : Array F}
  (hProp : evalLinkForMatrixProp bar m z r) :
  evalLinkForMatrix bar m z r = true := by
  unfold evalLinkForMatrixProp at hProp
  unfold evalLinkForMatrix
  exact evalLinkIdentity_complete hProp

theorem evalLinkForMatrix_iff_prop
  {bar : Array (Array F)} {m : Array (Array F)} {z r : Array F} :
  evalLinkForMatrix bar m z r = true ↔ evalLinkForMatrixProp bar m z r := by
  constructor
  · exact evalLinkForMatrix_sound_full
  · exact evalLinkForMatrix_complete

theorem evalLinkForMatrixProp_from_defs
  {bar : Array (Array F)} {m : Array (Array F)} {z r : Array F} :
  evalLinkForMatrixProp bar m z r := by
  unfold evalLinkForMatrixProp
  exact evalLinkIdentityProp_of_size_eq (by simp [rHat_size])

theorem evalLinkForMatrix_true_from_defs
  {bar : Array (Array F)} {m : Array (Array F)} {z r : Array F} :
  evalLinkForMatrix bar m z r = true := by
  exact evalLinkForMatrix_complete (evalLinkForMatrixProp_from_defs (bar := bar) (m := m) (z := z) (r := r))

/--
Theorem-native assumption interface for P13 (Remark-2 evaluation/ct linkage).
-/
def p13EvalLinkAssumption (bar : Array (Array F)) : Prop :=
  ∀ (m : Array (Array F)) (z r : Array F),
    MatrixRowsCompatible m z →
    evalLinkForMatrixProp bar m z r

/--
Check-oriented universal assumption interface for P13, conditioned on row-shape
preconditions.
-/
def p13EvalLinkCheckAssumption (bar : Array (Array F)) : Prop :=
  ∀ (m : Array (Array F)) (z r : Array F),
    MatrixRowsCompatible m z →
    evalLinkForMatrix bar m z r = true

theorem p13EvalLinkAssumption_from_defs
  {bar : Array (Array F)} :
  p13EvalLinkAssumption bar := by
  intro m z r _hRows
  exact evalLinkForMatrixProp_from_defs (bar := bar) (m := m) (z := z) (r := r)

theorem p13EvalLinkCheckAssumption_from_defs
  {bar : Array (Array F)} :
  p13EvalLinkCheckAssumption bar := by
  intro m z r _hRows
  exact evalLinkForMatrix_true_from_defs (bar := bar) (m := m) (z := z) (r := r)

theorem p13EvalLinkAssumption_of_checkAssumption
  {bar : Array (Array F)}
  (hCheck : p13EvalLinkCheckAssumption bar) :
  p13EvalLinkAssumption bar := by
  intro m z r hRows
  exact evalLinkForMatrix_sound_full (hCheck m z r hRows)

theorem p13EvalLinkCheckAssumption_of_assumption
  {bar : Array (Array F)}
  (hAssm : p13EvalLinkAssumption bar) :
  p13EvalLinkCheckAssumption bar := by
  intro m z r hRows
  exact evalLinkForMatrix_complete (hAssm m z r hRows)

/--
P13 assumption specialized to a fixed matrix and randomness point.
-/
def p13EvalLinkAssumptionFor
  (bar : Array (Array F))
  (m : Array (Array F))
  (r : Array F) : Prop :=
  ∀ z : Array F, MatrixRowsCompatible m z -> evalLinkForMatrixProp bar m z r

/--
Check-oriented P13 assumption specialized to a fixed matrix/randomness point.
-/
def p13EvalLinkCheckAssumptionFor
  (bar : Array (Array F))
  (m : Array (Array F))
  (r : Array F) : Prop :=
  ∀ z : Array F, MatrixRowsCompatible m z -> evalLinkForMatrix bar m z r = true

theorem p13EvalLinkAssumptionFor_of_assumption
  {bar : Array (Array F)} {m : Array (Array F)} {r : Array F}
  (hAssm : p13EvalLinkAssumption bar) :
  p13EvalLinkAssumptionFor bar m r := by
  intro z hRows
  exact hAssm m z r hRows

theorem p13EvalLinkCheckAssumptionFor_of_checkAssumption
  {bar : Array (Array F)} {m : Array (Array F)} {r : Array F}
  (hCheck : p13EvalLinkCheckAssumption bar) :
  p13EvalLinkCheckAssumptionFor bar m r := by
  intro z hRows
  exact hCheck m z r hRows

theorem p13EvalLinkAssumptionFor_of_checkAssumptionFor
  {bar : Array (Array F)} {m : Array (Array F)} {r : Array F}
  (hCheck : p13EvalLinkCheckAssumptionFor bar m r) :
  p13EvalLinkAssumptionFor bar m r := by
  intro z hRows
  exact evalLinkForMatrix_sound_full (hCheck z hRows)

theorem p13EvalLinkCheckAssumptionFor_of_assumptionFor
  {bar : Array (Array F)} {m : Array (Array F)} {r : Array F}
  (hAssm : p13EvalLinkAssumptionFor bar m r) :
  p13EvalLinkCheckAssumptionFor bar m r := by
  intro z hRows
  exact evalLinkForMatrix_complete (hAssm z hRows)

theorem evalLinkForMatrixProp_of_assumption
  {bar : Array (Array F)} {m : Array (Array F)} {z r : Array F}
  (hAssm : p13EvalLinkAssumption bar)
  (hRows : MatrixRowsCompatible m z) :
  evalLinkForMatrixProp bar m z r := by
  exact hAssm m z r hRows

theorem evalLinkForMatrix_true_of_assumption
  {bar : Array (Array F)} {m : Array (Array F)} {z r : Array F}
  (hAssm : p13EvalLinkAssumption bar)
  (hRows : MatrixRowsCompatible m z) :
  evalLinkForMatrix bar m z r = true := by
  exact evalLinkForMatrix_complete (evalLinkForMatrixProp_of_assumption hAssm hRows)

end SuperNeo
