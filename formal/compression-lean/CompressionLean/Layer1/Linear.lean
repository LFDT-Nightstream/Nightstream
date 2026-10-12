import CompressionLean.Layer1.Weights
import Mathlib.Tactic.LinearCombination

/-!
Owns the linear stage of the layer-1 verifier (`verifier.rs`, from `λ` to
the linear claim): `λ`, the quotient `Q`, `ζ`, the linear sum-check (block
bits, then lane bits) on `Σ_b Ω_b·Σ_l Λ(l)·z(b, l)`, the finals and the check
`last = Ω~(s_b)·Λ~(s_l)·z~(s)` (`linear_world`).

The sum-check is replayed as one: the block rounds and the lane rounds are
two product sum-checks whose claims chain (`prove_linear` runs them so).
-/

namespace CompressionLean.Layer1

open Polynomial

variable {F K : Type} [Field F] [Field K] [Algebra F K]

local notation "ι" => algebraMap F K

/-- The linear stage's transcript. -/
structure LinearOut (K : Type) where
  lam : K
  quotient : ℕ → K
  zeta : K
  blockPoint : List K
  middle : K
  lanePart : List K
  last : K
  keyValue : K
  omegaValue : K
  zValue : K

/-- The prover's quotient from its `D - 1` coefficients. -/
noncomputable def quotientPoly (s : Shape) (q : ℕ → K) : K[X] :=
  ∑ i ∈ Finset.range (s.D - 1), C (q i) * X ^ i

/-- `Ȳ(ζ) + Q(ζ)·Φ(ζ)` (`ring::lifted_target`). -/
noncomputable def liftedTarget (s : Shape) (Φ : F[X]) (bar : ℕ → ℕ → F) (S : Setup F) (st : Statement F)
    (lam zeta : K) (Q : K[X]) : K :=
  ∑ ρ ∈ Finset.range s.rows, lam ^ ρ * (((relationRows s bar S st).target ρ).map (algebraMap F K)).eval zeta +
    Q.eval zeta * (Φ.map (algebraMap F K)).eval zeta

/-- The verifier's `Ω~(s_b)`: scaled key value, Eval_K, the opened `Ω^A~`, public. -/
noncomputable def omegaComputed (s : Shape) (bar : ℕ → ℕ → F) (st : Statement F) (lam zeta : K)
    (blockPoint : List K) (keyValue omegaValue : K) : K :=
  laneScale zeta s.laneVars * keyValue +
    lam ^ s.evalKRow 0 * mle blockPoint (evalKTable s bar st zeta 0) +
    lam ^ s.evalKRow 1 * mle blockPoint (evalKTable s bar st zeta 1) +
    omegaValue + mle blockPoint (publicPart s lam)

/-- The linear check (and no pole of the lane weights). -/
def LinearCheck (s : Shape) (bar : ℕ → ℕ → F) (st : Statement F) (o : LinearOut K) : Prop :=
  NoLanePole o.zeta s.laneVars ∧
    o.last = omegaComputed s bar st o.lam o.zeta o.blockPoint o.keyValue o.omegaValue *
      mle o.lanePart (laneTable s o.zeta) * o.zValue

/-- `X₀·X₁`. -/
noncomputable def productComb : MvPolynomial (Fin 2) K := MvPolynomial.X 0 * MvPolynomial.X 1

/-- Block rounds: `Ω` and `u(b) = Σ_l Λ(l)·z(2^laneVars·b + l)`. -/
noncomputable def blockTables (s : Shape) (bar : ℕ → ℕ → F) (S : Setup F) (st : Statement F)
    (lam zeta : K) (z : ℕ → F) : Fin 2 → ℕ → K := fun i =>
  if i = 0 then omegaTrue s bar S st lam zeta
  else fun b => ∑ l ∈ Finset.range (2 ^ s.laneVars), laneTable s zeta l * ι (z (2 ^ s.laneVars * b + l))

/-- Lane rounds: `Ω~(s_b)·Λ(l)` and `z(l, s_b)`. -/
noncomputable def laneTables (s : Shape) (bar : ℕ → ℕ → F) (S : Setup F) (st : Statement F)
    (lam zeta : K) (z : ℕ → F) (blockPoint : List K) : Fin 2 → ℕ → K := fun i =>
  if i = 0 then fun l => mle blockPoint (omegaTrue s bar S st lam zeta) * laneTable s zeta l
  else fun l => mle blockPoint fun b => ι (z (l + 2 ^ s.laneVars * b))

theorem product_truth_nil (n : ℕ) (T : Fin 2 → ℕ → K) :
    combTruth n T productComb [] = cubeSum n fun i => T 0 i * T 1 i := by
  rw [combTruth_nil]
  simp [productComb]

theorem product_truth_final (p : List K) (T : Fin 2 → ℕ → K) :
    combTruth p.length T productComb p = mle p (T 0) * mle p (T 1) := by
  rw [combTruth_final]
  simp [productComb]

theorem productComb_degree : (productComb : MvPolynomial (Fin 2) K).totalDegree ≤ 1 + 1 := by
  unfold productComb
  refine (MvPolynomial.totalDegree_mul _ _).trans ?_
  rw [MvPolynomial.totalDegree_X, MvPolynomial.totalDegree_X]

/-- The verifier's `Ω~(s_b)` is the true one when its key and `Ω^A` values are. -/
theorem omegaComputed_true (s : Shape) (bar : ℕ → ℕ → F) (S : Setup F) (st : Statement F)
    (fits : S.Fits s) (public_le : s.publicBlocks ≤ s.blocks) (D_le : s.D ≤ 2 ^ s.laneVars)
    (lam zeta : K) (blockPoint : List K) (keyValue omegaValue : K) (poles : NoLanePole zeta s.laneVars)
    (key : keyValue = mle (lanePoint zeta s.laneVars ++ blockPoint) (keyGroup s S lam))
    (omega : omegaValue = mle blockPoint (evalAPart s bar S st lam zeta)) :
    omegaComputed s bar st lam zeta blockPoint keyValue omegaValue =
      mle blockPoint (omegaTrue s bar S st lam zeta) := by
  have parts : omegaTrue s bar S st lam zeta = fun b =>
      keyPart s S lam zeta b + evalKPart s bar st lam zeta b + evalAPart s bar S st lam zeta b +
        publicPart s lam b := funext (omegaTrue_parts s bar S st fits public_le lam zeta)
  have keyTrue : laneScale zeta s.laneVars * keyValue = mle blockPoint (keyPart s S lam zeta) := by
    rw [key, mle_append_split, lanePoint_length, ← mle_mul_left]
    congr 1
    funext b
    rw [← scaled_eq _ _ poles, cubeSum_eq_sum_range]
    unfold keyPart
    rw [← Finset.sum_subset (Finset.range_mono D_le)]
    · apply Finset.sum_congr rfl
      intro l _
      rw [add_comm]
    · intro l inside outside
      unfold keyGroup
      rw [Finset.sum_eq_zero, mul_zero]
      intro i _
      rw [add_comm, fits.key_zero i b l (Or.inr (by simp at inside outside; omega)), map_zero, mul_zero]
  rw [parts]
  simp only [mle_add]
  unfold omegaComputed evalKPart
  rw [keyTrue, omega, mle_add, mle_mul_left, mle_mul_left]
  ring

/-- The block rounds' total is the batched row plus the target at `ζ`. -/
theorem blockTables_total (s : Shape) (bar : ℕ → ℕ → F) (S : Setup F) (st : Statement F)
    (blocks_le : s.blocks ≤ 2 ^ s.blockVars) (D_le : s.D ≤ 2 ^ s.laneVars) (lam zeta : K) (z : ℕ → F) :
    cubeSum s.blockVars (fun b => blockTables s bar S st lam zeta z 0 b * blockTables s bar S st lam zeta z 1 b) =
      ((relationRows s bar S st).batched (witnessBlock s z) lam).eval zeta +
        ∑ ρ ∈ Finset.range s.rows, lam ^ ρ * (((relationRows s bar S st).target ρ).map (algebraMap F K)).eval zeta := by
  have rows_eq : (relationRows s bar S st).rows = s.rows := rfl
  have blocks_eq : (relationRows s bar S st).blocks = s.blocks := rfl
  rw [RingRows.batched_eval, rows_eq, blocks_eq, sub_add_cancel, cubeSum_eq_sum_range]
  simp only [blockTables, if_true, show (1 : Fin 2) ≠ 0 by decide, if_false]
  rw [← Finset.sum_subset (Finset.range_mono blocks_le)]
  · apply Finset.sum_congr rfl
    intro b inside
    rw [omegaTrue, if_pos (Finset.mem_range.mp inside)]
    congr 1
    rw [witnessBlock, eval_coeffPoly, ← Finset.sum_subset (Finset.range_mono D_le)]
    · apply Finset.sum_congr rfl
      intro l member
      simp [laneTable, Finset.mem_range.mp member, mul_comm]
    · intro l inside' outside
      simp [laneTable, show ¬ l < s.D by simpa using outside]
  · intro b _ outside
    rw [omegaTrue, if_neg (by simpa using outside), zero_mul]

/-- The lane rounds start where the block rounds end. -/
theorem laneTables_nil (s : Shape) (bar : ℕ → ℕ → F) (S : Setup F) (st : Statement F) (lam zeta : K)
    (z : ℕ → F) (blockPoint : List K) :
    cubeSum s.laneVars (fun l => laneTables s bar S st lam zeta z blockPoint 0 l *
        laneTables s bar S st lam zeta z blockPoint 1 l) =
      mle blockPoint (blockTables s bar S st lam zeta z 0) * mle blockPoint (blockTables s bar S st lam zeta z 1) := by
  simp only [laneTables, blockTables, if_true, show (1 : Fin 2) ≠ 0 by decide, if_false]
  rw [cubeSum_eq_sum_range, mle_finset_sum, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro l _
  rw [mle_mul_left]
  have : (fun b => ι (z (2 ^ s.laneVars * b + l))) = fun b => ι (z (l + 2 ^ s.laneVars * b)) := by
    funext b; rw [add_comm]
  rw [this]
  ring

/-- The lane rounds end at `Ω~(s_b)·Λ~(s_l)·z~(s_l, s_b)`. -/
theorem laneTables_final (s : Shape) (bar : ℕ → ℕ → F) (S : Setup F) (st : Statement F) (lam zeta : K)
    (z : ℕ → F) (blockPoint lanePart : List K) (length : lanePart.length = s.laneVars) :
    combTruth lanePart.length (laneTables s bar S st lam zeta z blockPoint) productComb lanePart =
      mle blockPoint (omegaTrue s bar S st lam zeta) * mle lanePart (laneTable s zeta) *
        mle (lanePart ++ blockPoint) (fun x => ι (z x)) := by
  rw [product_truth_final]
  simp only [laneTables, if_true, show (1 : Fin 2) ≠ 0 by decide, if_false]
  rw [mle_mul_left, mle_swap, mle_append_split, length]

open Classical in
/-- The linear stage: `λ`, `Q`, `ζ`, the block and lane rounds, the finals and the check. -/
noncomputable def linearGame [Fintype K] [DecidableEq K] (s : Shape) (Φ : F[X]) (bar : ℕ → ℕ → F)
    (S : Setup F) (st : Statement F) (cands : List (P0Tables F)) : Game (Option (LinearOut K)) :=
  .draw K (Finset.univ.biUnion fun w : Fin cands.length =>
      (relationRows s bar S st).lambdaBad Φ (witnessBlock s (cands.get w).z)) fun lam =>
  .recv (ℕ → K) fun quotient =>
  .draw K (Finset.univ.biUnion fun w : Fin cands.length =>
      (relationRows s bar S st).zetaBad Φ (witnessBlock s (cands.get w).z) lam (quotientPoly s quotient))
    fun zeta =>
  Game.bind (sumcheck Finset.univ 1
      (fun w : Fin cands.length => combRound s.blockVars (blockTables s bar S st lam zeta (cands.get w).z) productComb)
      (fun _ => ∅) s.blockVars [] (liftedTarget s Φ bar S st lam zeta (quotientPoly s quotient))) fun first =>
  Game.bind (sumcheck Finset.univ 1
      (fun w : Fin cands.length =>
        combRound s.laneVars (laneTables s bar S st lam zeta (cands.get w).z first.1) productComb)
      (fun _ => ∅) s.laneVars [] first.2) fun second =>
  .recv (K × K × K) fun finals =>
    if LinearCheck s bar st ⟨lam, quotient, zeta, first.1, first.2, second.1, second.2,
        finals.1, finals.2.1, finals.2.2⟩ then
      .pure (some ⟨lam, quotient, zeta, first.1, first.2, second.1, second.2, finals.1, finals.2.1, finals.2.2⟩)
    else .pure none

/-- The linear stage forces every ring row of a candidate whose openings
agree, when the key and `Ω^A` values are true, off the bad sets. -/
theorem linear_world [Fintype K] [DecidableEq K] (s : Shape) (Φ : F[X]) (bar : ℕ → ℕ → F)
    (S : Setup F) (st : Statement F) (fits : S.Fits s) (public_le : s.publicBlocks ≤ s.blocks)
    (D_le : s.D ≤ 2 ^ s.laneVars) (blocks_le : s.blocks ≤ 2 ^ s.blockVars)
    (monic : Φ.Monic) (proper : Φ ≠ 1) (o : LinearOut K) (w : P0Tables F)
    (check : LinearCheck s bar st o)
    (hLam : o.lam ∉ (relationRows s bar S st).lambdaBad Φ (witnessBlock s w.z))
    (hζ : o.zeta ∉ (relationRows s bar S st).zetaBad Φ (witnessBlock s w.z) o.lam (quotientPoly s o.quotient))
    (blockLength : o.blockPoint.length = s.blockVars) (laneLength : o.lanePart.length = s.laneVars)
    (first : liftedTarget s Φ bar S st o.lam o.zeta (quotientPoly s o.quotient) ≠
        combTruth s.blockVars (blockTables s bar S st o.lam o.zeta w.z) productComb [] →
      o.middle ≠ combTruth s.blockVars (blockTables s bar S st o.lam o.zeta w.z) productComb o.blockPoint)
    (second : o.middle ≠ combTruth s.laneVars (laneTables s bar S st o.lam o.zeta w.z o.blockPoint) productComb [] →
      o.last ≠ combTruth s.laneVars (laneTables s bar S st o.lam o.zeta w.z o.blockPoint) productComb o.lanePart)
    (key : o.keyValue = mle (lanePoint o.zeta s.laneVars ++ o.blockPoint) (keyGroup s S o.lam))
    (omega : o.omegaValue = mle o.blockPoint (evalAPart s bar S st o.lam o.zeta))
    (zTrue : o.zValue = mle (o.lanePart ++ o.blockPoint) fun x => ι (w.z x)) :
    (relationRows s bar S st).Holds Φ (witnessBlock s w.z) := by
  obtain ⟨poles, last⟩ := check
  have omegaTrue_eq := omegaComputed_true s bar S st fits public_le D_le o.lam o.zeta o.blockPoint
    o.keyValue o.omegaValue poles key omega
  -- The lane rounds' end is true, so their start is.
  have laneEnd : o.last =
      combTruth s.laneVars (laneTables s bar S st o.lam o.zeta w.z o.blockPoint) productComb o.lanePart := by
    rw [← laneLength, laneTables_final _ _ _ _ _ _ _ _ _ laneLength, last, omegaTrue_eq, zTrue]
  have middle : o.middle =
      combTruth s.laneVars (laneTables s bar S st o.lam o.zeta w.z o.blockPoint) productComb [] := by
    by_contra wrong
    exact second wrong laneEnd
  -- The block rounds' end is the lane rounds' start, so their start is true.
  have blockEnd : o.middle =
      combTruth s.blockVars (blockTables s bar S st o.lam o.zeta w.z) productComb o.blockPoint := by
    rw [middle, product_truth_nil, laneTables_nil, ← blockLength, product_truth_final]
  have start : liftedTarget s Φ bar S st o.lam o.zeta (quotientPoly s o.quotient) =
      combTruth s.blockVars (blockTables s bar S st o.lam o.zeta w.z) productComb [] := by
    by_contra wrong
    exact first wrong blockEnd
  rw [product_truth_nil, blockTables_total s bar S st blocks_le D_le, liftedTarget] at start
  apply (relationRows s bar S st).ring_sound Φ (witnessBlock s w.z) monic proper o.lam o.zeta
    (quotientPoly s o.quotient) hLam hζ
  linear_combination -start

end CompressionLean.Layer1
