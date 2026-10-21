import CompressionLean.Bridge.Ring

/-!
Owns the denotation of the replay's Eval_K long division (`Replay.evalK`)
and of the linear stage's `Ω~` (`Replay.omegaComputed`). The replay's states
are the model's `evalKStates` digit by digit (`toQ_evalKStates`), so its value
is the model's `evalKValue` (`toQ_evalK`), whose parts `EvalK.evalK_split`
proves to be the extensions of the Eval_K tables. Then `Ω~` is the model's
`omegaComputed` (`toK_omegaComputed`).
-/

namespace CompressionLean.Bridge

open Polynomial
open CompressionLean.Replay CompressionLean.Layer1

variable {K : Type} [Field K] [Algebra Fp K] [Fintype K] [ChallengeField K]

local notation "ι" => algebraMap Fp K

/-- The point `r` in `K ⊗ Ext`. -/
def rQuad (r : Array (Gl × Gl)) : List (QuadX K) := (quadPoint r.toList).map fromK

/-- An `Ext` point in `K ⊗ Ext`. -/
def sQuad (s : Array Ext) : List (QuadX K) := (pointK s).map (algebraMap K (QuadX K))

theorem divStep_eq (blocks bb t rem below bit : ℕ) :
    Replay.divStep blocks bb t rem below bit = Layer1.divStep D blocks bb t rem below bit := rfl

theorem rQuad_getD (r : Array (Gl × Gl)) (t : ℕ) : (rQuad (K := K) r).getD t 0 = (Kx.ofK r[t]!).toQ := by
  rw [Kx.toQ_ofK]
  unfold rQuad quadPoint
  by_cases inside : t < r.size
  · rw [getElem!_pos r t inside, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_map,
      Array.getElem?_toList, Array.getElem?_eq_getElem inside]
    rfl
  · rw [getElem!_neg r t inside, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_map,
      Array.getElem?_toList, Array.getElem?_eq_none (by omega)]
    ext <;> simp [show (default : Gl × Gl) = (0, 0) from rfl]

theorem sQuad_getD (s : Array Ext) (t : ℕ) :
    (sQuad (K := K) s).getD t 0 = algebraMap K (QuadX K) (s[t]!).toK := by
  unfold sQuad
  rw [pointK_getD, List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_map]
  cases (pointK (K := K) s)[t]? <;> simp

theorem toQ_divFactor (r : Array (Gl × Gl)) (s : Array Ext) (t bit q : ℕ) :
    (Replay.divFactor r s t bit q).toQ (K := K) = Layer1.divFactor (rQuad r) (sQuad s) t bit q := by
  unfold Replay.divFactor Layer1.divFactor
  rw [rQuad_getD, sQuad_getD, show (sQuad (K := K) s).length = s.size by simp [sQuad]]
  by_cases inside : t < s.size
  · rw [if_pos inside, if_pos inside, Kx.toQ_scale]
    congr 1
    · split_ifs <;> simp [Kx.toQ_oneMinus]
    · split_ifs <;> simp [toK_oneMinus]
  · rw [if_neg inside, if_neg inside, mul_one]
    split_ifs <;> simp [Kx.toQ_oneMinus]

theorem toQ_contribution (step : Option (ℕ × ℕ × ℕ)) (rem' below' : ℕ) (factor : ℕ → Kx) (x : Kx) :
    (Replay.contribution step rem' below' factor x).toQ (K := K) =
      Layer1.contribution step rem' below' (fun q => (factor q).toQ) x.toQ := by
  unfold Replay.contribution Layer1.contribution
  rcases step with _ | ⟨r', b', q⟩
  · simp
  · simp only
    split_ifs <;> simp [Kx.toQ_mul]

theorem getElem!_map_range₂ {α : Type} [Inhabited α] (n m : ℕ) (f : ℕ → ℕ → α) {i j : ℕ} (hi : i < n) (hj : j < m) :
    (((Array.range n).map fun i => (Array.range m).map (f i))[i]!)[j]! = f i j := by
  rw [getElem!_map_range n _ hi, getElem!_map_range m _ hj]

/-- The replay's long-division states are the model's, digit by digit. -/
theorem toQ_evalKStates (r : Array (Gl × Gl)) (s : Array Ext) (blocks cb : ℕ) (k : ℕ) :
    ∀ rem below, rem < D → below < 2 →
      (((Replay.evalKStates r s blocks cb k)[rem]!)[below]!).toQ (K := K) =
        Layer1.evalKStates D blocks (rQuad r) (sQuad s) cb k rem below := by
  induction k with
  | zero =>
      intro rem below hr hb
      rw [Replay.evalKStates, getElem!_map_range₂ D 2 _ hr hb, Layer1.evalKStates]
      by_cases start : rem = 0 ∧ below = 0 <;> simp [start]
  | succ k ih =>
      intro rem' below' hr hb
      rw [Replay.evalKStates]
      rw [getElem!_map_range₂ D 2 _ hr hb, Layer1.evalKStates, toQ_sumKx]
      apply Finset.sum_congr rfl
      intro rem remIn
      rw [toQ_sumKx]
      apply Finset.sum_congr rfl
      intro below belowIn
      rw [toQ_sumKx]
      apply Finset.sum_congr rfl
      intro bit _
      rw [toQ_contribution, divStep_eq, ih rem below (Finset.mem_range.mp remIn) (Finset.mem_range.mp belowIn),
        show (sQuad (K := K) s).length = s.size by simp [sQuad]]
      congr 1
      funext q
      exact toQ_divFactor r s _ bit q

theorem toQ_tail (l : List (Gl × Gl)) (x : Kx) :
    (l.foldl (fun total value => total.mul (Kx.ofK value).oneMinus) x).toQ (K := K) =
      x.toQ * ((quadPoint l).map fromK |>.map fun v => 1 - v).prod := by
  induction l generalizing x with
  | nil => simp [quadPoint]
  | cons v rest ih =>
      rw [List.foldl_cons, ih, Kx.toQ_mul, Kx.toQ_oneMinus, Kx.toQ_ofK]
      simp only [quadPoint, List.map_cons, List.prod_cons]
      ring

/-- The replay's Eval_K is the model's long division. -/
theorem toQ_evalK (r : Array (Gl × Gl)) (s : Array Ext) (tau : Array Ext) (blocks : ℕ) :
    (Replay.evalK r s tau blocks).toQ (K := K) =
      evalKValue D blocks (rQuad r) (sQuad s) (fun l => algebraMap K (QuadX K) (tableK tau l))
        (Nat.log2 (D * 2 ^ s.size).nextPowerOfTwo) := by
  unfold Replay.evalK evalKValue
  simp only []
  rw [← Array.foldl_toList, toQ_tail, toQ_sumKx, Array.toList_extract, List.extract_eq_take_drop,
    List.take_of_length_le (by simp), rQuad, quadPoint, List.map_drop, List.map_drop]
  congr 1
  apply Finset.sum_congr rfl
  intro rem inside
  rw [Kx.toQ_scale, toQ_evalKStates r s blocks _ _ rem 1 (Finset.mem_range.mp inside) (by norm_num)]
  rfl

/-- The relation's Eval_K bits: they cover the block bits and the blocks
and fit the point `r` (`eval_k`'s coordinate bits). -/
structure EvalKBits (rel : Replay.Relation) : Prop where
  deep : rel.blockVars ≤ Nat.log2 (D * 2 ^ rel.blockVars).nextPowerOfTwo
  long : Nat.log2 (D * 2 ^ rel.blockVars).nextPowerOfTwo ≤ rel.pointVariables
  blocks_le : rel.key.blocks ≤ 2 ^ rel.blockVars
  fits : D * rel.key.blocks ≤ 2 ^ Nat.log2 (D * 2 ^ rel.blockVars).nextPowerOfTwo
  strict : rel.key.blocks < 2 ^ Nat.log2 (D * 2 ^ rel.blockVars).nextPowerOfTwo

/-- The replay's Eval_K at the block point is the pair of the extensions of
the model's Eval_K tables. -/
theorem evalK_parts (rel : Replay.Relation) (st : Replay.Statement) (pointSize : st.point.size = rel.pointVariables)
    (bits : EvalKBits rel) (zeta : Ext) (bp : Array Ext) (size : bp.size = rel.blockVars) :
    (Replay.evalK st.point bp (Replay.tau zeta) rel.key.blocks).toQ (K := K) =
      ⟨mle (pointK bp) (evalKTable (shapeOf rel) barOf (statementOf st) zeta.toK 0),
        mle (pointK bp) (evalKTable (shapeOf rel) barOf (statementOf st) zeta.toK 1)⟩ := by
  rw [toQ_evalK, size]
  have split := evalK_split (K := K) (shapeOf rel) barOf (statementOf st) zeta.toK (pointK bp)
    (Nat.log2 (D * 2 ^ rel.blockVars).nextPowerOfTwo) (show 0 < D by decide)
    (by rw [pointK_length, size]; exact bits.deep)
    (by rw [statement_r_length, pointSize]; exact bits.long)
    (by rw [pointK_length, size]; exact bits.blocks_le) bits.fits bits.strict
  rw [← split, statement_point]
  congr 1
  funext l
  rw [tableK_tau (shapeOf rel) rfl]

omit [Algebra Fp K] [Fintype K] [ChallengeField K] in
theorem mle_publicPart (s : Layer1.Shape) (lam : K) (p : List K) (small : s.publicBlocks ≤ 2 ^ p.length) :
    mle p (publicPart s lam) = ∑ b ∈ Finset.range s.publicBlocks, eqAt p b * lam ^ s.publicRow b := by
  rw [mle_eq_cubeSum, cubeSum_eq_sum_range]
  unfold publicPart
  simp only [Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro b inside
  simp only [mul_ite, mul_zero]
  rw [Finset.sum_ite_eq']
  rw [if_pos (Finset.mem_range.mpr (by have := Finset.mem_range.mp inside; omega))]

/-- The replay's `Ω~(s_b)` is the model's `omegaComputed`. -/
theorem toK_omegaComputed (rel : Replay.Relation) (st : Replay.Statement) (pointSize : st.point.size = rel.pointVariables)
    (bits : EvalKBits rel) (publicFits : rel.publicBlocks ≤ 2 ^ rel.blockVars)
    (lam zeta scale keyValue omegaValue : Ext) (bp : Array Ext) (size : bp.size = rel.blockVars)
    (hscale : scale.toK (K := K) = laneScale zeta.toK laneVars) :
    (Replay.omegaComputed rel st ⟨lam, rel.kappa, rel.matrices⟩ (Replay.tau zeta) bp scale keyValue omegaValue).toK
        (K := K) =
      Layer1.omegaComputed (shapeOf rel) barOf (statementOf st) lam.toK zeta.toK (pointK bp) keyValue.toK
        omegaValue.toK := by
  have parts := evalK_parts (K := K) rel st pointSize bits zeta bp size
  unfold Replay.omegaComputed Layer1.omegaComputed
  rw [Ext.toK_add, Ext.toK_add, Ext.toK_add, Ext.toK_add, Ext.toK_mul, Ext.toK_mul, Ext.toK_mul, hscale,
    show (Replay.evalK st.point bp (Replay.tau zeta) rel.key.blocks).re.toK (K := K) =
      mle (pointK bp) (evalKTable (shapeOf rel) barOf (statementOf st) zeta.toK 0) from congrArg QuadraticAlgebra.re parts,
    show (Replay.evalK st.point bp (Replay.tau zeta) rel.key.blocks).im.toK (K := K) =
      mle (pointK bp) (evalKTable (shapeOf rel) barOf (statementOf st) zeta.toK 1) from congrArg QuadraticAlgebra.im parts,
    mle_publicPart _ _ _ (by rw [pointK_length, size]; exact publicFits), toK_sumRange]
  simp only [Mixing.evalK, Ext.toK_pow, Ext.toK_mul, toK_public, toK_eqAt, Layer1.Shape.evalKRow, Layer1.Shape.publicRow,
    shapeOf]
  rfl

end CompressionLean.Bridge
