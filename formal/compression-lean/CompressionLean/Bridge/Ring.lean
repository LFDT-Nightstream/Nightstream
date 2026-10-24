import CompressionLean.Bridge.Early

/-!
Owns the denotation of the linear stage's formulas: the scaled eq point
(`scaledEqPoint_ok`: the model's `laneScale`, `lanePoint`, `NoLanePole`),
the lane weights `τ` (`tableK_tau`: the model's `laneWeight`), the lane
value (`toK_laneValue`: the extension of `laneTable`) and the lifted target
(`toK_liftedTarget`: the model's `liftedTarget` at `Φ81`).
-/

namespace CompressionLean.Bridge

open CompressionLean.Spec
open CompressionLean.Lib
open Polynomial
open CompressionLean.Replay CompressionLean.Layer1

variable {K : Type} [Field K] [Algebra Gl K] [Fintype K] [ChallengeField K]

local notation "ι" => algebraMap Gl K

theorem scaledEqPoint_ok (bits : ℕ) : ∀ (c : Ext) {kappa : Ext} {point : Array Ext},
    scaledEqPoint c bits = .ok (kappa, point) →
      kappa.toK (K := K) = laneScale (c.toK (K := K)) bits ∧ pointK point = lanePoint (c.toK (K := K)) bits ∧
        NoLanePole (c.toK (K := K)) bits := by
  induction bits with
  | zero =>
      intro c kappa point ok
      rw [scaledEqPoint] at ok
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.pure_eq_ok ok)
      exact ⟨Ext.toK_one, rfl, trivial⟩
  | succ bits ih =>
      intro c kappa point ok
      rw [scaledEqPoint] at ok
      split_ifs at ok with pole
      · obtain ⟨_, thrown, _⟩ := Except.bind_ok ok
        cases thrown
      · obtain ⟨⟨k, rest⟩, recursive, ok⟩ := Except.bind_ok ok
        obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.pure_eq_ok ok)
        obtain ⟨scale, points, poles⟩ := ih (c * c) recursive
        have nonzero : (1 + c).toK (K := K) ≠ 0 := Ext.toK_ne_zero (by simpa using pole)
        rw [Ext.toK_add, Ext.toK_one] at nonzero
        rw [Ext.toK_mul, ← pow_two] at scale points poles
        refine ⟨?_, ?_, nonzero, poles⟩
        · rw [Ext.toK_mul, scale, Ext.toK_add, Ext.toK_one]
          rfl
        · rw [pointK_append, points]
          simp [pointK, lanePoint, Ext.toK_mul, Ext.toK_inv, Ext.toK_add, div_eq_mul_inv]

theorem bar_outside (p l : ℕ) (outside : D ≤ l) : barOf p l = 0 := by
  have : ¬ (p < NightstreamFPrime.Spec.ringDegree ∧ l < NightstreamFPrime.Spec.ringDegree) := fun h =>
    absurd h.2 (by rw [show NightstreamFPrime.Spec.ringDegree = D from rfl]; omega)
  simp [barOf, Replay.bar, this]

/-- The replay's `τ` is the model's lane weights. -/
theorem tableK_tau (s : Layer1.Shape) (hD : s.D = D) (zeta : Ext) (l : ℕ) :
    tableK (K := K) (Replay.tau zeta) l = laneWeight s barOf zeta.toK l := by
  unfold tableK laneWeight
  rw [hD]
  by_cases inside : l < D
  · rw [getElem!_pos (Replay.tau zeta) l (by simpa [Replay.tau] using inside)]
    simp only [Replay.tau, Array.getElem_map, Array.getElem_range]
    rw [toK_sumRange]
    apply Finset.sum_congr rfl
    intro p _
    rw [Ext.toK_scale, Ext.toK_pow, mul_comm]
    rfl
  · rw [getElem!_neg (Replay.tau zeta) l (by simpa [Replay.tau] using inside), toK_default]
    symm
    apply Finset.sum_eq_zero
    intro p _
    rw [bar_outside p l (by omega), map_zero, zero_mul]

/-- `Σ_{l<D} ζ^l·eq(x, l)` is the extension of the lane weights `ζ^l`. -/
theorem toK_laneValue (s : Layer1.Shape) (hD : s.D = D) (zeta : Ext) (point : Array Ext)
    (size : point.size = laneVars) :
    (laneValue zeta point).toK (K := K) = mle (pointK point) (laneTable s zeta.toK) := by
  rw [laneValue, toK_sumRange, mle_eq_cubeSum, cubeSum_eq_sum_range, pointK_length, size]
  have split : 2 ^ laneVars = D + 10 := rfl
  rw [split, Finset.sum_range_add, Finset.sum_eq_zero (s := Finset.range 10) (fun i _ => by
    rw [laneTable, if_neg (by rw [hD]; omega), mul_zero]), add_zero]
  apply Finset.sum_congr rfl
  intro l inside
  have lt := Finset.mem_range.mp inside
  rw [Ext.toK_mul, eqTable_get point (by rw [size, split]; omega), toK_eqAt, Ext.toK_pow, laneTable,
    if_pos (by rw [hD]; exact lt)]

/-! ## The lifted target -/

/-- Coefficient `l` of the target of row `ρ` (`relationRows`). -/
def targetCoeff (s : Layer1.Shape) (st : Layer1.Statement Gl) (ρ l : ℕ) : Gl :=
  if ρ < s.kappa then st.commitment ρ l
  else if ρ < s.kappa + 2 then (if ρ = s.evalKRow 0 then (st.evalK l).1 else (st.evalK l).2)
  else if ρ < s.kappa + 2 + 2 * s.matrices then
    (if (ρ - s.kappa - 2) % 2 = 0 then (st.evalA ((ρ - s.kappa - 2) / 2) l).1
    else (st.evalA ((ρ - s.kappa - 2) / 2) l).2)
  else st.publicInput (ρ - (s.kappa + 2 + 2 * s.matrices)) l

theorem target_eq (s : Layer1.Shape) (bar : ℕ → ℕ → Gl) (S : Setup Gl) (st : Layer1.Statement Gl) (ρ : ℕ) :
    (relationRows s bar S st).target ρ = coeffPoly s.D (targetCoeff s st ρ) := by
  simp only [relationRows]
  by_cases h1 : ρ < s.kappa
  · rw [if_pos h1]
    congr 1
    funext l
    simp [targetCoeff, h1]
  by_cases h2 : ρ < s.kappa + 2
  · rw [if_neg h1, if_pos h2]
    congr 1
    funext l
    simp [targetCoeff, h1, h2]
  by_cases h3 : ρ < s.kappa + 2 + 2 * s.matrices
  · rw [if_neg h1, if_neg h2, if_pos h3]
    congr 1
    funext l
    simp [targetCoeff, h1, h2, h3]
  · rw [if_neg h1, if_neg h2, if_neg h3]
    congr 1
    funext l
    simp [targetCoeff, h1, h2, h3]

omit [Fintype K] [ChallengeField K] in
/-- The batched target coefficient `Σ_ρ λ^ρ·y_ρ(l)`, row by row. -/
theorem rows_split (s : Layer1.Shape) (st : Layer1.Statement Gl) (lam : K) (l : ℕ) :
    ∑ ρ ∈ Finset.range s.rows, lam ^ ρ * ι (targetCoeff s st ρ l) =
      ∑ row ∈ Finset.range s.kappa, lam ^ row * ι (st.commitment row l) +
        (lam ^ s.kappa * ι (st.evalK l).1 + lam ^ (s.kappa + 1) * ι (st.evalK l).2) +
        ∑ j ∈ Finset.range s.matrices, (lam ^ (s.kappa + 2 + 2 * j) * ι (st.evalA j l).1 +
          lam ^ (s.kappa + 3 + 2 * j) * ι (st.evalA j l).2) +
        ∑ b ∈ Finset.range s.publicBlocks,
          lam ^ (s.kappa + 2 + 2 * s.matrices + b) * ι (st.publicInput b l) := by
  rw [show s.rows = s.kappa + (2 + (2 * s.matrices + s.publicBlocks)) by simp [Layer1.Shape.rows]; ring,
    Finset.sum_range_add, Finset.sum_range_add, Finset.sum_range_add, sum_range_pairs]
  have commitment : ∑ ρ ∈ Finset.range s.kappa, lam ^ ρ * ι (targetCoeff s st ρ l) =
      ∑ row ∈ Finset.range s.kappa, lam ^ row * ι (st.commitment row l) :=
    Finset.sum_congr rfl fun ρ inside => by rw [targetCoeff, if_pos (Finset.mem_range.mp inside)]
  have evalK : ∑ r ∈ Finset.range 2, lam ^ (s.kappa + r) * ι (targetCoeff s st (s.kappa + r) l) =
      lam ^ s.kappa * ι (st.evalK l).1 + lam ^ (s.kappa + 1) * ι (st.evalK l).2 := by
    rw [Finset.sum_range_succ, Finset.sum_range_succ, Finset.sum_range_zero, zero_add, targetCoeff, targetCoeff,
      if_neg (by omega), if_pos (by omega), if_pos (by simp [Layer1.Shape.evalKRow]), if_neg (by omega),
      if_pos (by omega), if_neg (by simp [Layer1.Shape.evalKRow]), add_zero]
  have evalA : ∑ j ∈ Finset.range s.matrices,
      (lam ^ (s.kappa + (2 + 2 * j)) * ι (targetCoeff s st (s.kappa + (2 + 2 * j)) l) +
        lam ^ (s.kappa + (2 + (2 * j + 1))) * ι (targetCoeff s st (s.kappa + (2 + (2 * j + 1))) l)) =
      ∑ j ∈ Finset.range s.matrices, (lam ^ (s.kappa + 2 + 2 * j) * ι (st.evalA j l).1 +
        lam ^ (s.kappa + 3 + 2 * j) * ι (st.evalA j l).2) := by
    apply Finset.sum_congr rfl
    intro j inside
    have hj := Finset.mem_range.mp inside
    rw [targetCoeff, targetCoeff, if_neg (by omega), if_neg (by omega), if_pos (by omega), if_pos (by omega),
      if_neg (by omega), if_neg (by omega), if_pos (by omega), if_neg (by omega),
      show (s.kappa + (2 + 2 * j) - s.kappa - 2) / 2 = j by omega,
      show (s.kappa + (2 + (2 * j + 1)) - s.kappa - 2) / 2 = j by omega,
      show s.kappa + (2 + 2 * j) = s.kappa + 2 + 2 * j by ring,
      show s.kappa + (2 + (2 * j + 1)) = s.kappa + 3 + 2 * j by ring]
  have pub : ∑ b ∈ Finset.range s.publicBlocks, lam ^ (s.kappa + (2 + (2 * s.matrices + b))) *
      ι (targetCoeff s st (s.kappa + (2 + (2 * s.matrices + b))) l) =
      ∑ b ∈ Finset.range s.publicBlocks, lam ^ (s.kappa + 2 + 2 * s.matrices + b) * ι (st.publicInput b l) := by
    apply Finset.sum_congr rfl
    intro b _
    rw [targetCoeff, if_neg (by omega), if_neg (by omega), if_neg (by omega),
      show s.kappa + (2 + (2 * s.matrices + b)) - (s.kappa + 2 + 2 * s.matrices) = b by omega,
      show s.kappa + (2 + (2 * s.matrices + b)) = s.kappa + 2 + 2 * s.matrices + b by ring]
  rw [commitment, evalK, evalA, pub]
  ring

/-- Lane `l` of the replay's `Ȳ` is the batched target coefficient. -/
theorem toK_targets (rel : Replay.Relation) (st : Replay.Statement) (fits : st.fits rel = true) (lam : Ext)
    {lane : ℕ} (inside : lane < D) :
    ((targets st ⟨lam, rel.kappa, rel.matrices⟩)[lane]!).toK (K := K) =
      ∑ ρ ∈ Finset.range (shapeOf rel).rows, lam.toK ^ ρ * ι (targetCoeff (shapeOf rel) (statementOf st) ρ lane) := by
  have sizes := fits
  simp only [Replay.Statement.fits, Bool.and_eq_true, beq_iff_eq] at sizes
  obtain ⟨⟨⟨⟨⟨⟨⟨commitmentSize, _⟩, publicSize⟩, _⟩, _⟩, _⟩, evalASize⟩, _⟩ := sizes
  rw [rows_split, getElem!_pos _ lane (by simpa [targets] using inside)]
  simp only [targets, Array.getElem_map, Array.getElem_range]
  rw [Ext.toK_add, Ext.toK_add, Ext.toK_add, toK_sumRange, toK_sumRange, toK_sumRange, toK_project,
    commitmentSize, publicSize, evalASize]
  congr 1
  congr 1
  congr 1
  · apply Finset.sum_congr rfl
    intro row _
    rw [Ext.toK_scale, toK_commitment]
    rfl
  · simp [Mixing.evalK, Ext.toK_pow, shapeOf, statementOf]
  · apply Finset.sum_congr rfl
    intro j _
    rw [toK_project]
    simp [Mixing.evalA, Ext.toK_pow, shapeOf, statementOf]
  · apply Finset.sum_congr rfl
    intro b _
    rw [Ext.toK_scale, toK_public]
    rfl

omit [Fintype K] [ChallengeField K] in
theorem phi_eval (zeta : K) : (phi.map ι).eval zeta = zeta ^ 54 + zeta ^ 27 + 1 := by
  simp [phi]

/-- The replay's lifted target is the model's at `Φ81`, with the quotient of the proof. -/
theorem toK_liftedTarget (rel : Replay.Relation) (st : Replay.Statement) (fits : st.fits rel = true)
    (lam zeta : Ext) (quotient : Array Ext) (size : quotient.size = D - 1) (S : Setup Gl) :
    (Replay.liftedTarget (targets st ⟨lam, rel.kappa, rel.matrices⟩) quotient zeta).toK (K := K) =
      Layer1.liftedTarget (shapeOf rel) phi barOf S (statementOf st) lam.toK zeta.toK
        (quotientPoly (shapeOf rel) (tableK quotient)) := by
  unfold Replay.liftedTarget Layer1.liftedTarget
  rw [Ext.toK_add, Ext.toK_mul, toK_horner, toK_horner, Ext.toK_add, Ext.toK_add, Ext.toK_mul, Ext.toK_pow,
    Ext.toK_one, phi_eval, size]
  have targetSize : (targets st ⟨lam, rel.kappa, rel.matrices⟩).size = D := by simp [targets]
  rw [targetSize]
  congr 1
  · simp only [target_eq, eval_coeffPoly]
    rw [show (shapeOf rel).D = D from rfl]
    simp only [Finset.mul_sum]
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro lane inside
    rw [toK_targets rel st fits lam (Finset.mem_range.mp inside), Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro ρ _
    ring
  · congr 1
    · simp [quotientPoly, eval_finsetSum, tableK, shapeOf]
    · ring

end CompressionLean.Bridge
