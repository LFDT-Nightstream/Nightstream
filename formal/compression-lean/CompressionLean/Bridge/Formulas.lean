import CompressionLean.Bridge.Field
import CompressionLean.Layer1.EvalK
import CompressionLean.Layer1.KeyStructure

/-!
Owns the denotation of the replay's public formulas (`Replay/Formulas.lean`):
each one denotes its model counterpart in `K` (or `QuadX K`). Points denote
lists (`pointK`), arrays denote tables that are zero past their end
(`tableK`). The model counterparts are the closed forms that `ClosedForms`,
`EvalK` and `Weights` prove equal to the tables' extensions.
-/

namespace CompressionLean.Bridge

open CompressionLean.Replay CompressionLean.Layer1

variable {K : Type} [Field K] [Algebra Fp K] [Fintype K] [ChallengeField K]

local notation "ι" => algebraMap Fp K

/-- A point of `Ext` coordinates in `K`. -/
def pointK (p : Array Ext) : List K := p.toList.map Ext.toK

/-- An `Ext` array as a table of `K`, zero past its end. -/
def tableK (a : Array Ext) (i : ℕ) : K := (a[i]!).toK

/-- A pair of `Ext` values in `K`. -/
def pairK (v : Ext × Ext) : K × K := (v.1.toK, v.2.toK)

@[simp] theorem pointK_length (p : Array Ext) : (pointK (K := K) p).length = p.size := by simp [pointK]

theorem toK_default : (default : Ext).toK (K := K) = 0 := Ext.toK_zero

theorem tableK_eq (a : Array Ext) (i : ℕ) : tableK (K := K) a i = (pointK a).getD i 0 := by
  unfold tableK pointK
  by_cases inside : i < a.size
  · rw [getElem!_pos a i inside, List.getD_eq_getElem?_getD, List.getElem?_map, Array.getElem?_toList,
      Array.getElem?_eq_getElem inside]
    rfl
  · rw [getElem!_neg a i inside, toK_default, List.getD_eq_getElem?_getD, List.getElem?_map,
      Array.getElem?_toList, Array.getElem?_eq_none (by omega)]
    rfl

theorem pointK_getD (p : Array Ext) (i : ℕ) : (p[i]!).toK (K := K) = (pointK p).getD i 0 := tableK_eq p i

theorem pointK_push (p : Array Ext) (x : Ext) : pointK (K := K) (p.push x) = pointK p ++ [x.toK] := by
  simp [pointK]

theorem pointK_append (p q : Array Ext) : pointK (K := K) (p ++ q) = pointK p ++ pointK q := by
  simp [pointK]

/-! ## `circuit/algebra.rs` -/

theorem toK_oneMinus (x : Ext) : (oneMinus x).toK (K := K) = 1 - x.toK := by
  rw [oneMinus, Ext.toK_sub, Ext.toK_one]

theorem toK_eq1 (x y : Ext) : (eq1 x y).toK (K := K) = x.toK * y.toK + (1 - x.toK) * (1 - y.toK) := by
  simp only [eq1, Ext.toK_sub, Ext.toK_add, Ext.toK_mul, Ext.toK_one]
  ring

theorem toK_eqEvalList : ∀ xs ys : List Ext,
    (eqEvalList xs ys).toK (K := K) = Layer1.eqEval (xs.map Ext.toK) (ys.map Ext.toK)
  | [], ys => by cases ys <;> simp [eqEvalList, Layer1.eqEval]
  | x :: xs, [] => by simp [eqEvalList, Layer1.eqEval]
  | x :: xs, y :: ys => by
      simp only [eqEvalList, Layer1.eqEval, List.map_cons, Ext.toK_mul, toK_eqEvalList xs ys, toK_eq1]

theorem toK_eqEval (x y : Array Ext) : (Replay.eqEval x y).toK (K := K) = Layer1.eqEval (pointK x) (pointK y) :=
  toK_eqEvalList _ _

theorem toK_eqAtList : ∀ (xs : List Ext) (i : ℕ), (eqAtList xs i).toK (K := K) = Layer1.eqAt (xs.map Ext.toK) i
  | [], i => by simp [eqAtList, Layer1.eqAt]
  | x :: xs, i => by
      simp only [eqAtList, Layer1.eqAt, List.map_cons, Ext.toK_mul, toK_eqAtList xs (i / 2)]
      split_ifs <;> simp [toK_oneMinus]

theorem toK_eqAt (p : Array Ext) (i : ℕ) : (Replay.eqAt p i).toK (K := K) = Layer1.eqAt (pointK p) i :=
  toK_eqAtList _ _

theorem eqTable_size (p : Array Ext) : (eqTable p).size = 2 ^ p.size := by simp [eqTable]

theorem eqTable_get (p : Array Ext) {i : ℕ} (inside : i < 2 ^ p.size) : (eqTable p)[i]! = Replay.eqAt p i := by
  rw [getElem!_pos (eqTable p) i (by rw [eqTable_size]; exact inside)]
  simp [eqTable]

/-- `Σ_u eq(point, u)·partials[u]` is the extension of the partials. -/
theorem toK_combine (partials point : Array Ext) (size : partials.size = 2 ^ point.size) :
    (combine partials point).toK (K := K) = mle (pointK point) (tableK partials) := by
  rw [combine, dot, toK_sumRange, mle_eq_cubeSum, cubeSum_eq_sum_range, eqTable_size, size, min_self,
    pointK_length]
  apply Finset.sum_congr rfl
  intro i inside
  rw [Ext.toK_mul, eqTable_get point (Finset.mem_range.mp inside), toK_eqAt]
  rfl

theorem getD_toList (cs : Array Ext) (i : ℕ) : cs.toList.getD i 0 = cs[i]! := by
  rw [List.getD_eq_getElem?_getD, ← Array.getElem!_toList, List.getElem!_eq_getElem?_getD]
  rfl

theorem toK_horner (cs : Array Ext) (x : Ext) :
    (horner cs x).toK (K := K) = ∑ i ∈ Finset.range cs.size, (cs[i]!).toK * x.toK ^ i := by
  have list : ∀ l : List Ext, (l.foldr (fun c total => total * x + c) 0).toK (K := K) =
      ∑ i ∈ Finset.range l.length, (l.getD i 0).toK * x.toK ^ i := by
    intro l
    induction l with
    | nil => simp
    | cons c rest ih =>
        rw [List.foldr_cons, Ext.toK_add, Ext.toK_mul, ih, List.length_cons, Finset.sum_range_succ',
          Finset.sum_mul]
        simp only [List.getD_cons_succ, List.getD_cons_zero, pow_zero, mul_one, pow_succ]
        congr 1
        apply Finset.sum_congr rfl
        intro i _
        ring
  rw [horner, ← Array.foldr_toList, list, Array.length_toList]
  simp only [getD_toList]

/-! ## `mle.rs` -/

theorem toK_identityValue (p : Array Ext) :
    (Replay.identityValue p).toK (K := K) = Layer1.identityValue (pointK p) := by
  rw [Replay.identityValue, ← Array.foldr_toList, pointK]
  induction p.toList with
  | nil => simp [Layer1.identityValue]
  | cons x xs ih =>
      simp only [List.foldr_cons, List.map_cons, Layer1.identityValue] at ih ⊢
      rw [Ext.toK_add, Ext.toK_add, ih]
      ring

theorem toK_lessThanScan (p : Array Ext) (bound : ℕ) : ∀ (t : ℕ) (total pre : Ext),
    ((Replay.lessThanScan p bound t (total, pre)).1.toK, (Replay.lessThanScan p bound t (total, pre)).2.toK) =
      Layer1.lessThanScan (pointK (K := K) p) bound t (total.toK, pre.toK)
  | 0, total, pre => rfl
  | t + 1, total, pre => by
      rw [Replay.lessThanScan, Layer1.lessThanScan]
      split_ifs
      · rw [toK_lessThanScan p bound t]
        simp [Ext.toK_add, Ext.toK_mul, toK_oneMinus, pointK_getD]
      · rw [toK_lessThanScan p bound t]
        simp [Ext.toK_mul, toK_oneMinus, pointK_getD]

theorem toK_lessThan (p : Array Ext) (bound : ℕ) :
    (Replay.lessThan p bound).toK (K := K) = Layer1.lessThan (pointK p) bound := by
  unfold Replay.lessThan Layer1.lessThan
  rw [pointK_length]
  split_ifs
  · exact Ext.toK_one
  · have := congrArg Prod.fst (toK_lessThanScan (K := K) p bound p.size 0 1)
    simpa using this

theorem nextStart_map (stop : ℕ) (segs : List (ℕ × Ext)) :
    Layer1.nextStart stop (segs.map fun s => (s.1, s.2.toK (K := K))) = Replay.nextStart stop segs := by
  cases segs <;> rfl

/-- The model's running total is the replay's sum of the remaining segments. -/
theorem stepLoop_toK (p : Array Ext) (stop : ℕ) : ∀ (segs : List (ℕ × Ext)) (low : Ext) (total : K),
    Layer1.stepLoop (pointK p) stop (segs.map fun s => (s.1, s.2.toK)) low.toK total =
      total + (stepSum p stop segs low).toK
  | [], low, total => by simp [Layer1.stepLoop, stepSum]
  | (a, v) :: rest, low, total => by
      rw [List.map_cons, Layer1.stepLoop, stepSum, nextStart_map, ← toK_lessThan, stepLoop_toK p stop rest,
        Ext.toK_add, Ext.toK_mul, Ext.toK_sub]
      ring

theorem toK_step (p : Array Ext) (segs : List (ℕ × Ext)) (stop : ℕ) :
    (Replay.step p segs stop).toK (K := K) = stepValue (pointK p) stop (segs.map fun s => (s.1, s.2.toK)) := by
  rw [Replay.step, stepValue, nextStart_map, ← toK_lessThan, stepLoop_toK, zero_add]

/-- `r` as the model's point in `K = F[u]/(u² - 7)`. -/
def quadPoint (r : List (Word × Word)) : List (Quad Fp) := r.map fun v => ⟨v.1.toF, v.2.toF⟩

theorem toQ_chiFactor (r : Word × Word) (x : Ext) :
    (chiFactor r (some x)).toQ (K := K) =
      (1 - algebraMap K (QuadX K) x.toK) + fromK (K := K) (⟨r.1.toF, r.2.toF⟩ : Quad Fp) *
        (2 * algebraMap K (QuadX K) x.toK - 1) := by
  ext
  · simp [chiFactor, Kx.toQ, Kx.scale, Kx.ofK, Ext.toK_add, Ext.toK_mul, Ext.toK_sub, Ext.toK_ofGl, toK_oneMinus,
      QuadraticAlgebra.algebraMap_eq, QuadraticAlgebra.re_one, QuadraticAlgebra.im_one, QuadraticAlgebra.re_ofNat,
      QuadraticAlgebra.im_ofNat]
    ring
  · simp only [chiFactor, Kx.toQ, Kx.scale, Kx.ofK, Ext.toK_add, Ext.toK_mul, Ext.toK_sub, Ext.toK_ofGl, Ext.toK_one,
      QuadraticAlgebra.algebraMap_eq, QuadraticAlgebra.im_one, QuadraticAlgebra.re_ofNat, QuadraticAlgebra.im_ofNat,
      QuadraticAlgebra.im_add, QuadraticAlgebra.im_sub, QuadraticAlgebra.im_mul, QuadraticAlgebra.re_one,
      QuadraticAlgebra.re_sub, QuadraticAlgebra.re_mul, fromK_re, fromK_im]
    ring

theorem toQ_chiList : ∀ (r : List (Word × Word)) (xs : List Ext),
    (chiList r xs).toQ (K := K) =
      chiValue ((quadPoint r).map (fromK (K := K))) ((xs.map Ext.toK).map (algebraMap K (QuadX K)))
  | [], xs => by simp [chiList, chiValue, quadPoint]
  | r :: rs, x :: xs => by
      simp only [chiList, quadPoint, List.map_cons, chiValue]
      rw [Kx.toQ_mul, toQ_chiFactor, toQ_chiList rs xs]
      rfl
  | r :: rs, [] => by
      simp only [chiList, chiFactor, quadPoint, List.map_cons, List.map_nil, chiValue]
      rw [Kx.toQ_mul, Kx.toQ_oneMinus, Kx.toQ_ofK, toQ_chiList rs []]
      rfl

theorem toQ_chi (r : Array (Word × Word)) (p : Array Ext) :
    (Replay.chi r p).toQ (K := K) =
      chiValue ((quadPoint r.toList).map (fromK (K := K))) ((pointK p).map (algebraMap K (QuadX K))) :=
  toQ_chiList _ _

/-! ## `ring.rs` -/

@[simp] theorem toK_commitment (m : Mixing) (row : ℕ) : (m.commitment row).toK (K := K) = m.lam.toK ^ row :=
  Ext.toK_pow _ _

theorem toK_evalK (m : Mixing) :
    pairK (K := K) m.evalK = (m.lam.toK ^ m.kappa, m.lam.toK ^ (m.kappa + 1)) := by
  simp [pairK, Mixing.evalK, Ext.toK_pow]

theorem toK_evalA (m : Mixing) (j : ℕ) :
    pairK (K := K) (m.evalA j) = (m.lam.toK ^ (m.kappa + 2 + 2 * j), m.lam.toK ^ (m.kappa + 3 + 2 * j)) := by
  simp [pairK, Mixing.evalA, Ext.toK_pow]

@[simp] theorem toK_public (m : Mixing) (block : ℕ) :
    (m.public block).toK (K := K) = m.lam.toK ^ (m.kappa + 2 + 2 * m.matrices + block) :=
  Ext.toK_pow _ _

theorem toK_project (value : Word × Word) (weights : Ext × Ext) :
    (project value weights).toK (K := K) = weights.1.toK * ι value.1.toF + weights.2.toK * ι value.2.toF := by
  simp [project, Ext.toK_add, Ext.toK_scale]

end CompressionLean.Bridge
