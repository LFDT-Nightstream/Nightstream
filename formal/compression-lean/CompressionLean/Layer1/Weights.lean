import CompressionLean.Layer1.Encoding

/-!
Owns the batched ring weights at `ζ` and the parts the verifier assembles
them from (`verifier.rs`, linear claim): `Ω_b = Σ_ρ λ^ρ·g_{ρ,b}(ζ)` is the key
part, the Eval_K part, the Eval_A part and the public part
(`omegaTrue_parts`). The key part is read through the scaled eq point:
`Σ_{l < 2^m} ζ^l·f(l) = κ_ζ·f~(x_ζ)` (`scaled_eq`).
-/

namespace CompressionLean.Layer1

open CompressionLean.Lib
open Polynomial

variable {F K : Type} [Field F] [Field K] [Algebra F K]

local notation "ι" => algebraMap F K

/-- `x_t = ζ^{2^t} / (1 + ζ^{2^t})`, the scaled eq point (`mle::scaled_eq_point`). -/
noncomputable def lanePoint (zeta : K) : ℕ → List K
  | 0 => []
  | m + 1 => zeta / (1 + zeta) :: lanePoint (zeta ^ 2) m

/-- `κ = Π_t (1 + ζ^{2^t})`. -/
noncomputable def laneScale (zeta : K) : ℕ → K
  | 0 => 1
  | m + 1 => (1 + zeta) * laneScale (zeta ^ 2) m

/-- No `1 + ζ^{2^t}` vanishes (else the verifier rejects). -/
def NoLanePole (zeta : K) : ℕ → Prop
  | 0 => True
  | m + 1 => 1 + zeta ≠ 0 ∧ NoLanePole (zeta ^ 2) m

theorem lanePoint_length (zeta : K) (m : ℕ) : (lanePoint zeta m).length = m := by
  induction m generalizing zeta with
  | zero => rfl
  | succ m ih => simp [lanePoint, ih]

/-- The power-weighted cube sum is the scaled extension at the scaled eq point. -/
theorem scaled_eq (m : ℕ) (zeta : K) (poles : NoLanePole zeta m) (f : ℕ → K) :
    cubeSum m (fun l => zeta ^ l * f l) = laneScale zeta m * mle (lanePoint zeta m) f := by
  induction m generalizing zeta f with
  | zero => simp [cubeSum, laneScale, lanePoint, mle, fold]
  | succ m ih =>
      obtain ⟨pole, rest⟩ := poles
      simp only [cubeSum, lanePoint, laneScale, mle_cons]
      have even : (fun i => zeta ^ (2 * i) * f (2 * i)) = fun i => (zeta ^ 2) ^ i * f (2 * i) := by
        funext i; rw [pow_mul]
      have odd : (fun i => zeta ^ (2 * i + 1) * f (2 * i + 1)) =
          fun i => zeta * ((zeta ^ 2) ^ i * f (2 * i + 1)) := by
        funext i; rw [pow_succ, pow_mul]; ring
      rw [even, odd, cubeSum_mul_left, ih _ rest, ih _ rest]
      unfold bindLow
      rw [mle_add, mle_mul_left, mle_mul_left]
      field_simp
      ring

/-- The lane weights `Λ(l) = ζ^l` on the used lanes. -/
def laneTable (s : Shape) (zeta : K) (l : ℕ) : K := if l < s.D then zeta ^ l else 0

/-- The key group `Σ_{i<κ} λ^i·a_i` on the cube. -/
noncomputable def keyGroup (s : Shape) (S : Setup F) (lam : K) (x : ℕ) : K :=
  ∑ i ∈ Finset.range s.kappa, lam ^ i * ι (S.key i x)

/-- The parts of `Ω_b`: key, Eval_K, Eval_A, public. -/
noncomputable def keyPart (s : Shape) (S : Setup F) (lam zeta : K) (b : ℕ) : K :=
  ∑ l ∈ Finset.range s.D, zeta ^ l * keyGroup s S lam (2 ^ s.laneVars * b + l)

noncomputable def evalKTable (s : Shape) (bar : ℕ → ℕ → F) (st : Statement F) (zeta : K) (part b : ℕ) : K :=
  if b < s.blocks then
    ∑ l ∈ Finset.range s.D, laneWeight s bar zeta l *
      ι (if part = 0 then (st.chi (s.D * b + l)).1 else (st.chi (s.D * b + l)).2)
  else 0

noncomputable def evalKPart (s : Shape) (bar : ℕ → ℕ → F) (st : Statement F) (lam zeta : K) (b : ℕ) : K :=
  lam ^ s.evalKRow 0 * evalKTable s bar st zeta 0 b + lam ^ s.evalKRow 1 * evalKTable s bar st zeta 1 b

noncomputable def evalAPart (s : Shape) (bar : ℕ → ℕ → F) (S : Setup F) (st : Statement F)
    (lam zeta : K) (b : ℕ) : K :=
  ∑ l ∈ Finset.range s.D, laneWeight s bar zeta l * evalAColumn s S st lam (s.D * b + l)

noncomputable def publicPart (s : Shape) (lam : K) (b : ℕ) : K :=
  ∑ j ∈ Finset.range s.publicBlocks, if b = j then lam ^ s.publicRow j else 0

/-- The true batched weight of block `b` at `ζ`; zero past the used blocks. -/
noncomputable def omegaTrue (s : Shape) (bar : ℕ → ℕ → F) (S : Setup F) (st : Statement F)
    (lam zeta : K) (b : ℕ) : K :=
  if b < s.blocks then
    ∑ ρ ∈ Finset.range s.rows,
      lam ^ ρ * (((relationRows s bar S st).weight ρ b).map (algebraMap F K)).eval zeta
  else 0

theorem eval_coeffPoly (D : ℕ) (v : ℕ → F) (zeta : K) :
    ((coeffPoly D v).map (algebraMap F K)).eval zeta = ∑ p ∈ Finset.range D, ι (v p) * zeta ^ p := by
  simp [coeffPoly, Polynomial.map_sum, eval_finsetSum]

theorem eval_barPoly (s : Shape) (bar : ℕ → ℕ → F) (w : ℕ → F) (zeta : K) :
    ((barPoly s.D bar w).map (algebraMap F K)).eval zeta =
      ∑ l ∈ Finset.range s.D, laneWeight s bar zeta l * ι (w l) := by
  rw [barPoly, eval_coeffPoly]
  simp only [map_sum, map_mul, Finset.sum_mul, laneWeight]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro l _
  apply Finset.sum_congr rfl
  intro p _
  ring

theorem sum_range_pairs {M : Type} [AddCommMonoid M] (n : ℕ) (g : ℕ → M) :
    ∑ r ∈ Finset.range (2 * n), g r = ∑ j ∈ Finset.range n, (g (2 * j) + g (2 * j + 1)) := by
  induction n with
  | zero => simp
  | succ n ih =>
      rw [show 2 * (n + 1) = 2 * n + 1 + 1 by ring, Finset.sum_range_succ, Finset.sum_range_succ, ih,
        Finset.sum_range_succ, add_assoc]

/-- `Ω_b` is the key, Eval_K, Eval_A and public parts. -/
theorem omegaTrue_parts (s : Shape) (bar : ℕ → ℕ → F) (S : Setup F) (st : Statement F) (fits : S.Fits s)
    (public_le : s.publicBlocks ≤ s.blocks) (D_le : s.D ≤ 2 ^ s.laneVars) (lam zeta : K) (b : ℕ) :
    omegaTrue s bar S st lam zeta b =
      keyPart s S lam zeta b + evalKPart s bar st lam zeta b + evalAPart s bar S st lam zeta b +
        publicPart s lam b := by
  unfold omegaTrue
  split
  · rename_i inside
    rw [show s.rows = s.kappa + (2 + (2 * s.matrices + s.publicBlocks)) by simp [Shape.rows]; ring,
      Finset.sum_range_add, Finset.sum_range_add, Finset.sum_range_add]
    have key : ∑ ρ ∈ Finset.range s.kappa,
        lam ^ ρ * (((relationRows s bar S st).weight ρ b).map (algebraMap F K)).eval zeta =
        keyPart s S lam zeta b := by
      unfold keyPart keyGroup
      simp only [Finset.mul_sum]
      rw [Finset.sum_comm]
      apply Finset.sum_congr rfl
      intro ρ member
      simp only [relationRows, if_pos (Finset.mem_range.mp member), eval_coeffPoly, Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro l _
      ring
    have evalK : ∑ r ∈ Finset.range 2,
        lam ^ (s.kappa + r) * (((relationRows s bar S st).weight (s.kappa + r) b).map
          (algebraMap F K)).eval zeta = evalKPart s bar st lam zeta b := by
      rw [Finset.sum_range_succ, Finset.sum_range_succ, Finset.sum_range_zero, zero_add]
      simp only [relationRows, add_zero]
      rw [if_neg (lt_irrefl _), if_pos (by omega : s.kappa < s.kappa + 2),
        if_neg (by omega : ¬ (s.kappa + 1 < s.kappa)), if_pos (by omega : s.kappa + 1 < s.kappa + 2)]
      simp only [eval_barPoly, evalKPart, evalKTable, if_pos inside, Shape.evalKRow, add_zero,
        if_true, show s.kappa + 1 ≠ s.kappa by omega, if_false, show (1 : ℕ) ≠ 0 by omega]
    have evalA : ∑ r ∈ Finset.range (2 * s.matrices),
        lam ^ (s.kappa + (2 + r)) * (((relationRows s bar S st).weight (s.kappa + (2 + r)) b).map
          (algebraMap F K)).eval zeta = evalAPart s bar S st lam zeta b := by
      rw [sum_range_pairs]
      unfold evalAPart evalAColumn
      simp only [Finset.mul_sum]
      rw [Finset.sum_comm]
      apply Finset.sum_congr rfl
      intro j member
      have hj := Finset.mem_range.mp member
      simp only [relationRows, Shape.evalARow]
      simp only [show ¬ (s.kappa + (2 + 2 * j) < s.kappa) by omega,
        show ¬ (s.kappa + (2 + (2 * j + 1)) < s.kappa) by omega,
        show ¬ (s.kappa + (2 + 2 * j) < s.kappa + 2) by omega,
        show ¬ (s.kappa + (2 + (2 * j + 1)) < s.kappa + 2) by omega,
        show s.kappa + (2 + 2 * j) < s.kappa + 2 + 2 * s.matrices by omega,
        show s.kappa + (2 + (2 * j + 1)) < s.kappa + 2 + 2 * s.matrices by omega,
        if_false, if_true, eval_barPoly]
      simp only [show s.kappa + (2 + 2 * j) - s.kappa - 2 = 2 * j by omega,
        show s.kappa + (2 + (2 * j + 1)) - s.kappa - 2 = 2 * j + 1 by omega,
        show 2 * j % 2 = 0 by omega, show (2 * j + 1) % 2 = 1 by omega,
        show 2 * j / 2 = j by omega, show (2 * j + 1) / 2 = j by omega,
        if_true, show (1 : ℕ) ≠ 0 by omega, if_false, Finset.mul_sum, ← Finset.sum_add_distrib]
      apply Finset.sum_congr rfl
      intro l _
      rw [show s.kappa + 2 + 2 * j + 0 = s.kappa + (2 + 2 * j) by ring,
        show s.kappa + 2 + 2 * j + 1 = s.kappa + (2 + (2 * j + 1)) by ring]
      ring
    have publicSum : ∑ r ∈ Finset.range s.publicBlocks,
        lam ^ (s.kappa + (2 + (2 * s.matrices + r))) *
          (((relationRows s bar S st).weight (s.kappa + (2 + (2 * s.matrices + r))) b).map
            (algebraMap F K)).eval zeta = publicPart s lam b := by
      unfold publicPart
      apply Finset.sum_congr rfl
      intro j member
      simp only [relationRows, Shape.publicRow]
      simp only [show ¬ (s.kappa + (2 + (2 * s.matrices + j)) < s.kappa) by omega,
        show ¬ (s.kappa + (2 + (2 * s.matrices + j)) < s.kappa + 2) by omega,
        show ¬ (s.kappa + (2 + (2 * s.matrices + j)) < s.kappa + 2 + 2 * s.matrices) by omega,
        if_false, show s.kappa + (2 + (2 * s.matrices + j)) - (s.kappa + 2 + 2 * s.matrices) = j by omega]
      split <;> simp [show s.kappa + 2 + 2 * s.matrices + j = s.kappa + (2 + (2 * s.matrices + j)) by ring]
    rw [key, evalK, evalA, publicSum]
    ring
  · rename_i outside
    have key : keyPart s S lam zeta b = 0 := by
      unfold keyPart keyGroup
      apply Finset.sum_eq_zero
      intro l member
      rw [Finset.sum_eq_zero, mul_zero]
      intro i _
      rw [fits.key_zero i b l (lt_of_lt_of_le (Finset.mem_range.mp member) D_le) (Or.inl (by omega)), map_zero,
        mul_zero]
    have evalK : evalKPart s bar st lam zeta b = 0 := by
      simp [evalKPart, evalKTable, outside]
    have evalA : evalAPart s bar S st lam zeta b = 0 := by
      unfold evalAPart evalAColumn evalAWeight
      apply Finset.sum_eq_zero
      intro l _
      rw [mul_eq_zero]
      right
      apply Finset.sum_eq_zero
      intro j _
      rw [Finset.filter_eq_empty_iff.mpr]
      · simp
      · intro k _ covers
        have := fits.inside (S.runSlot k)
        have : s.D * s.blocks ≤ s.D * b := Nat.mul_le_mul_left _ (by omega)
        omega
    have publicZero : publicPart s lam b = 0 := by
      unfold publicPart
      apply Finset.sum_eq_zero
      intro j member
      rw [if_neg (by have := Finset.mem_range.mp member; omega)]
    rw [key, evalK, evalA, publicZero]
    ring

end CompressionLean.Layer1
