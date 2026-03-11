import SuperNeo.Thm3Core

/-!
Theorem-4 matrix-transform layer (paper-faithful).

Reading guide:
1. `matrixVecDirect` is the direct field-level matrix-vector computation.
2. `matrixVecCtBar` is the ring-level side: block-wise ct(mulRqPhi(bar(·), ·)).
3. `matrixTransformIdentity` is the executable check surface.
4. `matrixTransformAssumption` is the theorem-facing boundary used downstream.
5. `_of_assumption`, `_of_checkAssumption`, `_iff_...` are check/prop bridges.

Paper anchor: Theorem 4 (Matrix-Vector Product Transform), Appendix D.1.
-/

namespace SuperNeo

open F

/-- Dot product on field vectors with a size guard. -/
def dotVec (a b : Array F) : F :=
  if a.size != b.size then
    0
  else
    (List.range a.size).foldl (fun acc i => acc + a[i]! * b[i]!) 0

private theorem d_ne_zero : d ≠ 0 := Nat.ne_of_gt d_pos

/-! ### Ring-level block operations for paper-faithful Theorem 4 -/

/-- Constant term of ring product with bar transform on the left block (Theorem 3 kernel). -/
def ctBarDot (bar : Array (Array F)) (a b : Array F) : F :=
  ct (mulRqPhi (superneoBarBlock bar a) b)

/-- Extract the j-th d-sized block from a field vector. -/
def extractBlock (v : Array F) (j : Nat) : Array F :=
  v.extract (j * d) ((j + 1) * d)

/-- Ring-level dot product: Σ_j ct(mulRqPhi(bar(row_j), z_j)) over d-sized blocks. -/
def ringBlockDot (bar : Array (Array F)) (row z : Array F) : F :=
  let nR := min (row.size / d) (z.size / d)
  (List.range nR).foldl (fun acc j =>
    acc + ctBarDot bar (extractBlock row j) (extractBlock z j)) 0

/-- Direct field-side block dot product: Σ_j ⟨row_j, z_j⟩ over d-sized blocks. -/
def directBlockDot (row z : Array F) : F :=
  let nR := min (row.size / d) (z.size / d)
  (List.range nR).foldl (fun acc j =>
    acc + innerProduct (extractBlock row j) (extractBlock z j)) 0

/-- Direct matrix-vector product (`Mz`) computed row-wise in d-sized blocks. -/
def matrixVecDirect (m : Array (Array F)) (z : Array F) : Array F :=
  m.map (fun row => directBlockDot row z)

/-- Bar-lifted matrix-vector side (paper: ct(M̄z) via block-wise ring products). -/
def matrixVecCtBar (bar : Array (Array F)) (m : Array (Array F)) (z : Array F) : Array F :=
  m.map (fun row => ringBlockDot bar row z)

/-- `dotVec` is definitionally equivalent to the Theorem-3 `innerProduct` surface. -/
theorem dotVec_eq_innerProduct (a b : Array F) :
  dotVec a b = innerProduct a b := by
  by_cases h : a.size = b.size
  · simp [dotVec, innerProduct, h]
  · simp [dotVec, innerProduct, h]

@[simp] theorem matrixVecDirect_size (m : Array (Array F)) (z : Array F) :
    (matrixVecDirect m z).size = m.size := by
  simp [matrixVecDirect]

@[simp] theorem matrixVecCtBar_size (bar : Array (Array F)) (m : Array (Array F)) (z : Array F) :
    (matrixVecCtBar bar m z).size = m.size := by
  simp [matrixVecCtBar]

/-- Row-shape compatibility predicate used by matrix-transform statements. -/
def MatrixRowsCompatible (m : Array (Array F)) (z : Array F) : Prop :=
  z.size % d = 0 ∧
    ∀ i (hi : i < m.size), (m[i]'hi).size = z.size

instance matrixRowsCompatible_decidable (m : Array (Array F)) (z : Array F) :
    Decidable (MatrixRowsCompatible m z) := by
  unfold MatrixRowsCompatible
  infer_instance

/-- Check surface for Theorem 4 identity. -/
def matrixTransformIdentity (bar : Array (Array F)) (m : Array (Array F)) (z : Array F) : Bool :=
  if z.size % d != 0 then
    false
  else if !(m.all (fun row => row.size = z.size)) then
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
  (hMod : z.size % d = 0)
  (hAll : m.all (fun row => row.size = z.size) = true) :
  MatrixRowsCompatible m z := by
  refine ⟨hMod, ?_⟩
  intro i hi
  have hDec : decide ((m[i]'hi).size = z.size) = true :=
    (Array.all_eq_true.mp hAll) i hi
  exact decide_eq_true_eq.mp hDec

private theorem all_true_of_matrixRowsCompatible
  {m : Array (Array F)} {z : Array F}
  (hRows : MatrixRowsCompatible m z) :
  m.all (fun row => row.size = z.size) = true := by
  rcases hRows with ⟨_hMod, hRowsEq⟩
  apply (Array.all_eq_true).2
  intro i hi
  exact decide_eq_true (hRowsEq i hi)

private theorem extractBlock_size_of_lt_div
  {v : Array F} {j : Nat}
  (hj : j < v.size / d) :
  (extractBlock v j).size = d := by
  have hsucc : j + 1 ≤ v.size / d := Nat.succ_le_of_lt hj
  have hStopLeMul : (j + 1) * d ≤ (v.size / d) * d :=
    Nat.mul_le_mul_right d hsucc
  have hStopLe : (j + 1) * d ≤ v.size :=
    Nat.le_trans hStopLeMul (Nat.div_mul_le_self v.size d)
  have hStartLeStop : j * d ≤ (j + 1) * d := by
    simpa [Nat.succ_mul, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using
      (Nat.le_add_right (j * d) d)
  have hStartLe : j * d ≤ v.size := Nat.le_trans hStartLeStop hStopLe
  have hSub : (j + 1) * d - j * d = d := by
    simpa [Nat.succ_mul, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using
      (Nat.add_sub_cancel (j * d) d)
  calc
    (extractBlock v j).size
        = ((j + 1) * d) - (j * d) := by
            simp [extractBlock, Array.size_extract, Nat.min_eq_left hStopLe, Nat.min_eq_left hStartLe]
    _ = d := hSub

private theorem list_foldl_congr
  {α β : Type}
  (l : List α)
  (init : β)
  (f g : β → α → β)
  (hfg : ∀ acc x, x ∈ l → f acc x = g acc x) :
  l.foldl f init = l.foldl g init := by
  induction l generalizing init with
  | nil =>
      rfl
  | cons x xs ih =>
      have hHead : f init x = g init x := hfg init x (by simp)
      calc
        (x :: xs).foldl f init = xs.foldl f (f init x) := by rfl
        _ = xs.foldl f (g init x) := by rw [hHead]
        _ = xs.foldl g (g init x) := by
              apply ih
              intro acc y hy
              exact hfg acc y (by simp [hy])
        _ = (x :: xs).foldl g init := by rfl

private theorem directBlockDot_eq_ringBlockDot_of_thm3CoreAssumption
  {bar : Array (Array F)} {row z : Array F}
  (hThm3 : thm3CoreAssumption bar) :
  directBlockDot row z = ringBlockDot bar row z := by
  unfold directBlockDot ringBlockDot
  let nR := min (row.size / d) (z.size / d)
  apply list_foldl_congr
    (l := List.range nR)
    (init := (0 : F))
    (f := fun acc j => acc + innerProduct (extractBlock row j) (extractBlock z j))
    (g := fun acc j => acc + ctBarDot bar (extractBlock row j) (extractBlock z j))
  intro acc j hjMem
  have hj : j < nR := List.mem_range.mp hjMem
  have hjRow : j < row.size / d := Nat.lt_of_lt_of_le hj (Nat.min_le_left _ _)
  have hjZ : j < z.size / d := Nat.lt_of_lt_of_le hj (Nat.min_le_right _ _)
  have hRowSize : (extractBlock row j).size = d := extractBlock_size_of_lt_div hjRow
  have hZSize : (extractBlock z j).size = d := extractBlock_size_of_lt_div hjZ
  have hTerm :
      innerProduct (extractBlock row j) (extractBlock z j) =
        ctBarDot bar (extractBlock row j) (extractBlock z j) := by
    simpa [ctBarDot] using (hThm3 (extractBlock row j) (extractBlock z j) hRowSize hZSize).symm
  simpa [hTerm]

/--
Theorem-4 identity derived from the Theorem-3 assumption.

The proof is closed by reducing each row to block-wise terms and applying
Theorem-3 on each extracted `d`-sized block pair.
-/
theorem matrixTransformEq_of_thm3CoreAssumption
  {bar : Array (Array F)} {m : Array (Array F)} {z : Array F}
  (hThm3 : thm3CoreAssumption bar)
  (_hRows : MatrixRowsCompatible m z) :
  matrixVecDirect m z = matrixVecCtBar bar m z := by
  apply Array.ext
  · simp [matrixVecDirect, matrixVecCtBar]
  · intro i hiL hiR
    have hi : i < m.size := by simpa using hiL
    simpa [matrixVecDirect, matrixVecCtBar, hi] using
      (directBlockDot_eq_ringBlockDot_of_thm3CoreAssumption
        (bar := bar)
        (row := m[i]'hi)
        (z := z)
        hThm3)

theorem matrixTransformIdentity_sound
  {bar : Array (Array F)} {m : Array (Array F)} {z : Array F}
  (hOk : matrixTransformIdentity bar m z = true) :
  matrixTransformIdentityProp bar m z := by
  unfold matrixTransformIdentity at hOk
  by_cases hMod : z.size % d = 0
  · simp [hMod] at hOk
    exact ⟨⟨hMod, hOk.1⟩, hOk.2⟩
  · simp [hMod] at hOk

theorem matrixTransformIdentity_complete
  {bar : Array (Array F)} {m : Array (Array F)} {z : Array F}
  (hProp : matrixTransformIdentityProp bar m z) :
  matrixTransformIdentity bar m z = true := by
  rcases hProp with ⟨hRows, hEq⟩
  rcases hRows with ⟨hMod, hRowsEq⟩
  unfold matrixTransformIdentity
  simp [hMod, all_true_of_matrixRowsCompatible ⟨hMod, hRowsEq⟩, decide_eq_true hEq]

theorem matrixTransformIdentity_iff_prop
  {bar : Array (Array F)} {m : Array (Array F)} {z : Array F} :
  matrixTransformIdentity bar m z = true ↔ matrixTransformIdentityProp bar m z := by
  constructor
  · exact matrixTransformIdentity_sound
  · exact matrixTransformIdentity_complete

/-- Compatibility alias exposing both row-compatibility and transform equality from check success. -/
theorem matrixTransformIdentity_sound_full
  {bar : Array (Array F)} {m : Array (Array F)} {z : Array F}
  (hOk : matrixTransformIdentity bar m z = true) :
  MatrixRowsCompatible m z ∧ matrixVecDirect m z = matrixVecCtBar bar m z := by
  exact matrixTransformIdentity_sound hOk

/-- Compatibility constructor for the check surface from explicit row/equality proofs. -/
theorem matrixTransformIdentity_complete_of_rowsCompatible
  {bar : Array (Array F)} {m : Array (Array F)} {z : Array F}
  (hRows : MatrixRowsCompatible m z)
  (hEq : matrixVecDirect m z = matrixVecCtBar bar m z) :
  matrixTransformIdentity bar m z = true := by
  exact matrixTransformIdentity_complete ⟨hRows, hEq⟩

/-- Theorem-native `P12` constructor from Theorem-3 (`P10`). -/
theorem matrixTransformAssumption_of_thm3CoreAssumption
  {bar : Array (Array F)} {m : Array (Array F)}
  (hThm3 : thm3CoreAssumption bar) :
  matrixTransformAssumption bar m := by
  intro z hRows
  exact matrixTransformEq_of_thm3CoreAssumption hThm3 hRows

/-- Theorem-native `P12` constructor from `P10` only. -/
theorem matrixTransformAssumption_of_p10
  {bar : Array (Array F)} {m : Array (Array F)}
  (hThm3 : thm3CoreAssumption bar) :
  matrixTransformAssumption bar m := by
  exact matrixTransformAssumption_of_thm3CoreAssumption hThm3

/-- Theorem-native `P12` constructor from the finite basis-kernel Theorem-3 witness. -/
theorem matrixTransformAssumption_of_basisKernelAssumption
  {bar : Array (Array F)} {m : Array (Array F)}
  (hBasis : thm3BasisKernelAssumption bar) :
  matrixTransformAssumption bar m := by
  exact matrixTransformAssumption_of_thm3CoreAssumption
    (thm3CoreAssumption_of_basisKernelAssumption hBasis)

/-- Theorem-native `P12` constructor from the executable finite basis-kernel checker. -/
theorem matrixTransformAssumption_of_basisKernelCheck
  {bar : Array (Array F)} {m : Array (Array F)}
  (hCheck : thm3BasisKernelCheck bar = true) :
  matrixTransformAssumption bar m := by
  exact matrixTransformAssumption_of_basisKernelAssumption
    (thm3BasisKernelAssumption_of_check hCheck)

/--
Theorem-native `P12` constructor from `(P10 + P11)` boundaries.

`P11` is carried explicitly to keep dependency accounting symmetric at the theorem layer.
-/
theorem matrixTransformAssumption_of_p10_p11
  {bar : Array (Array F)} {m : Array (Array F)}
  (hThm3 : thm3CoreAssumption bar)
  (_hLift : barLiftLinearityAssumption bar) :
  matrixTransformAssumption bar m := by
  exact matrixTransformAssumption_of_p10 hThm3

/--
Native theorem-stack constructor: instantiate P12 directly from the canonical
Theorem-3 closure theorem (`thm3CoreAssumption_native`), without threading an
explicit `hThm3` argument.
-/
theorem matrixTransformAssumption_native
  (m : Array (Array F)) :
  matrixTransformAssumption nativeBarMatrix m := by
  exact matrixTransformAssumption_of_thm3CoreAssumption
    (bar := nativeBarMatrix)
    (m := m)
    thm3CoreAssumption_native

/--
Rewrite helper: if `bar = nativeBarMatrix`, obtain P12 without explicitly
passing `thm3CoreAssumption bar`.
-/
theorem matrixTransformAssumption_of_bar_eq_native
  {bar : Array (Array F)} {m : Array (Array F)}
  (hBar : bar = nativeBarMatrix) :
  matrixTransformAssumption bar m := by
  subst hBar
  exact matrixTransformAssumption_native m

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
