import CompressionLean.Layer1.Soundness

/-!
Owns the error of the layer-1 verifier game: the bad-challenge count of each
stage, the query term, and the error bound (`layer1Game_cost`): the bad sets
of `layer1Game` have total density at most `layer1Error` along every path.

The field part counts bad challenges over `|K|`: per candidate, the
polynomial degrees of the norm, lookup, scatter, sum-check, GKR, ring-row,
and fold identities. The query part is the chance that every query leaf of a
wrong P1 fold agrees with the honest fold.
-/

namespace CompressionLean.Layer1

open CompressionLean.Lib
open Polynomial

variable {F K : Type} [Field F] [Field K] [Algebra F K] [Fintype K] [DecidableEq K]

local notation "ι" => algebraMap F K

section Counting

theorem gkrCost_mono (s : GkrShape) {n n' : ℕ} (E : ℕ → ℕ) (hn : n ≤ n') :
    ∀ remaining k, gkrCost s n E remaining k ≤ gkrCost s n' E remaining k := by
  intro remaining
  induction remaining with
  | zero => intro k; simp [gkrCost]
  | succ r ih =>
      intro k
      simp only [gkrCost, gkrLayerCost]
      have := ih (k + 1)
      gcongr

omit [DecidableEq K] in
theorem density_pair (bad : Finset (K × K)) (n : ℕ) (card : bad.card ≤ n * Fintype.card K) :
    (bad.card : ℝ) / Fintype.card (K × K) ≤ n / Fintype.card K := by
  have positive : (0 : ℝ) < Fintype.card K := by exact_mod_cast Fintype.card_pos
  rw [Fintype.card_prod, Nat.cast_mul, div_le_div_iff₀ (by positivity) positive]
  have cast : (bad.card : ℝ) ≤ n * Fintype.card K := by exact_mod_cast card
  calc (bad.card : ℝ) * Fintype.card K ≤ (n * Fintype.card K) * Fintype.card K :=
        mul_le_mul_of_nonneg_right cast positive.le
    _ = n * (Fintype.card K * Fintype.card K) := by ring

end Counting
section Early

/-- The slot-block test rides on the draws of the run layer: one bad value
per candidate there, none elsewhere. -/
def earlyRide (s : Shape) (L0 k : ℕ) : ℕ := if k = s.runVars then L0 else 0

/-- The early stage's bad challenges: norm `β`, lookup, scatter, GKR (with the
slot-block test at the run layer). -/
def earlyCount (s : Shape) (L0 : ℕ) : ℕ :=
  L0 * (2 ^ s.cubeVars + (2 * s.normBound + 1)) +
    (L0 * (2 * 2 ^ s.rowVars + (2 ^ s.runVars + 2 ^ s.rowVars)) +
      (L0 * (1 + (2 ^ s.runVars + 2 ^ (s.slotVars + s.matrixVars))) +
        gkrCost (earlyShape s) L0 (earlyRide s L0) (earlyShape s).layers 1))

theorem blockTest_card (s : Shape) (S : Setup F) (w : P0Tables F) (point : List K) :
    (blockTest s S w point).card ≤ 1 := by
  unfold blockTest
  split
  · exact foldBad_card _ _ _
  · simp

theorem earlyGame_cost (s : Shape) (S : Setup F) (st : Statement F) (L0 : ℕ)
    (nodes : ∀ d, d ≤ 3 → NodesDistinct K d) :
    (earlyGame (K := K) s S st L0).CostLe ((earlyCount s L0 : ℝ) / Fintype.card K) := by
  intro cands hist
  have short := cands.2
  have count : (Finset.univ : Finset (Fin cands.1.length)).card ≤ L0 := by simpa using short
  unfold earlyCount
  apply Game.draw_step _ _ _ _ (Game.density_le _ _ ?_)
  swap
  · refine (card_biUnion_le_mul _ _ (2 ^ s.cubeVars + (2 * s.normBound + 1)) fun w _ =>
      zeros_card_le_of_natDegree_le (normPoly_natDegree _ _ _ _)).trans (Nat.mul_le_mul_right _ count)
  intro beta
  apply Game.draw_step _ _ _ _ (density_pair _ _ ?_)
  swap
  · refine (card_biUnion_le_mul _ _ ((2 * 2 ^ s.rowVars + (2 ^ s.runVars + 2 ^ s.rowVars)) * Fintype.card K)
      fun w _ => Lookup.bad_card _).trans ?_
    rw [← mul_assoc]
    exact Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ count)
  intro lookup
  apply Game.draw_step _ _ _ _ (density_pair _ _ ?_)
  swap
  · refine (card_biUnion_le_mul _ _ ((1 + (2 ^ s.runVars + 2 ^ (s.slotVars + s.matrixVars))) * Fintype.card K)
      fun w _ => Scatter.bad_card _).trans ?_
    rw [← mul_assoc]
    exact Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ count)
  intro scatter
  have small : ∀ t, (earlyShape s).factors t ≤ 2 := by
    intro t; simp only [earlyShape]; split_ifs <;> omega
  have extra : ∀ (k : ℕ) (point : List K), (earlyExtra s S cands.1 k point).card ≤ earlyRide s L0 k := by
    intro k point
    unfold earlyExtra earlyRide
    split
    · exact (card_biUnion_le_mul _ _ 1 fun w _ => blockTest_card s S _ point).trans (by simpa using count)
    · simp
  refine (Game.costLe_bind (gkr_cost Finset.univ _ _ (earlyRide s L0) extra (nodes_of_small _ small nodes))
    (δ := 0) ?_).mono ?_
  · rintro (_ | claims)
    · exact le_rfl
    · intro values
      dsimp only
      split_ifs <;> exact le_rfl
  · rw [add_zero]
    exact div_le_div_of_nonneg_right (by exact_mod_cast gkrCost_mono _ _ count _ _) (by positivity)

theorem earlyGame_ends (s : Shape) (S : Setup F) (st : Statement F) (L0 : ℕ) :
    (earlyGame (K := K) s S st L0).Ends fun r => ∀ o, r = some o → o.candidates.length ≤ L0 := by
  intro cands hist beta lookup scatter
  rw [Game.ends_bind]
  refine Game.Ends.of_forall ?_ _
  rintro (_ | claims)
  · intro o none; cases none
  · intro values
    dsimp only
    split_ifs
    · rintro o ⟨⟩; exact cands.2
    · intro o none; cases none

end Early
section Linear

/-- The linear stage's bad challenges: `λ`, `ζ`, and the two sum-checks. -/
def linearCount (s : Shape) (L0 : ℕ) : ℕ :=
  L0 * (s.rows - 1) + (L0 * (2 * s.D - 2) + (s.blockVars * (L0 * 2) + s.laneVars * (L0 * 2)))

omit [Field F] in
theorem coeffPoly_natDegree [Field F] (D : ℕ) (v : ℕ → F) : (coeffPoly D v).natDegree ≤ D - 1 :=
  RingRows.natDegree_powers_le D v

omit [Algebra F K] [Fintype K] [DecidableEq K] in
theorem quotientPoly_natDegree (s : Shape) (quotient : ℕ → K) :
    (quotientPoly s quotient).natDegree ≤ s.D - 2 := by
  have := RingRows.natDegree_powers_le (s.D - 1) quotient
  unfold quotientPoly
  omega

omit [Field K] [Algebra F K] [Fintype K] [DecidableEq K] in
theorem relationRows_weight_degree (s : Shape) (bar : ℕ → ℕ → F) (S : Setup F) (st : Statement F) :
    ∀ ρ b, ((relationRows s bar S st).weight ρ b).natDegree ≤ s.D - 1 := by
  intro ρ b
  simp only [relationRows]
  split_ifs <;> first | exact coeffPoly_natDegree _ _ | simp

omit [Field K] [Algebra F K] [Fintype K] [DecidableEq K] in
theorem relationRows_target_degree (s : Shape) (bar : ℕ → ℕ → F) (S : Setup F) (st : Statement F) :
    ∀ ρ, ((relationRows s bar S st).target ρ).natDegree ≤ s.D - 1 := by
  intro ρ
  simp only [relationRows]
  split_ifs <;> exact coeffPoly_natDegree _ _

theorem linearGame_cost (s : Shape) (Φ : F[X]) (bar : ℕ → ℕ → F) (S : Setup F) (st : Statement F)
    (cands : List (P0Tables F)) (L0 : ℕ) (short : cands.length ≤ L0) (nodes : NodesDistinct K 1)
    (degree : Φ.natDegree = s.D) (two : 2 ≤ s.D) :
    (linearGame (K := K) s Φ bar S st cands).CostLe ((linearCount s L0 : ℝ) / Fintype.card K) := by
  have count : (Finset.univ : Finset (Fin cands.length)).card ≤ L0 := by simpa using short
  unfold linearCount
  apply Game.draw_step _ _ _ _ (Game.density_le _ _ ?_)
  swap
  · exact (card_biUnion_le_mul _ _ (s.rows - 1) fun w _ => RingRows.lambdaBad_card _ _ _).trans
      (Nat.mul_le_mul_right _ count)
  intro lam quotient
  apply Game.draw_step _ _ _ _ (Game.density_le _ _ ?_)
  swap
  · exact (card_biUnion_le_mul _ _ (2 * s.D - 2) fun w _ => RingRows.zetaBad_card _ _ _ s.D lam _
      (relationRows_weight_degree s bar S st) (fun b => coeffPoly_natDegree _ _)
      (relationRows_target_degree s bar S st) (quotientPoly_natDegree s quotient) degree two).trans
      (Nat.mul_le_mul_right _ count)
  intro zeta
  have rounds : ∀ (n : ℕ) (T : Fin cands.length → Fin 2 → ℕ → K) (point : List K) (claim : K),
      point.length + n ≤ n →
      (Game.sumcheck Finset.univ 1 (fun w => combRound n (T w) productComb) (fun _ => ∅) n point claim).CostLe
        (n * (((Finset.univ : Finset (Fin cands.length)).card * (1 + 1) + 0 : ℕ) / (Fintype.card K : ℝ))) :=
    fun n T => Game.sumcheck_cost Finset.univ 1 nodes _ (fun _ => ∅) n 0
      (fun w _ => (combRound_truth n 1 (T w) productComb productComb_degree).degree) (by simp) n
  refine (Game.costLe_bind (rounds s.blockVars _ [] _ (by simp)) fun first =>
    Game.costLe_bind (δ := 0) (rounds s.laneVars _ [] _ (by simp)) fun second => ?_).mono ?_
  · intro finals
    dsimp only
    split_ifs <;> exact le_rfl
  · have c : (((Finset.univ : Finset (Fin cands.length)).card * (1 + 1) + 0 : ℕ) : ℝ) ≤ ((L0 * 2 : ℕ) : ℝ) := by
      exact_mod_cast (by omega)
    rw [show ((s.blockVars * (L0 * 2) + s.laneVars * (L0 * 2) : ℕ) : ℝ) / Fintype.card K =
        s.blockVars * (((L0 * 2 : ℕ) : ℝ) / Fintype.card K) +
          (s.laneVars * (((L0 * 2 : ℕ) : ℝ) / Fintype.card K) + 0) by push_cast; ring]
    gcongr

end Linear

section Honest

/-- The query term: a wrong P1 fold passes every query leaf, each leaf a
word below `P` mod `L`. -/
noncomputable def queryError (s : Shape) (L1 L P : ℕ) : ℝ :=
  L1 * ((((2 ^ (s.setupVars - 3) - 1) * (P / L + 1) : ℕ) : ℝ) / P) ^ s.queries

omit [Field F] [Field K] [Algebra F K] [Fintype K] [DecidableEq K] in
theorem queryError_nonneg (s : Shape) (L1 L P : ℕ) : 0 ≤ queryError s L1 L P := by
  unfold queryError; positivity

theorem honestGame_cost (s : Shape) (S : Setup F) (L1 L P : ℕ) [NeZero L] [NeZero P] (q : ℕ → K)
    (distinct : ∀ i j, i < L → j < L → q i = q j → i = j) (o : EarlyOut F K) (lin : LinearOut K) :
    (honestGame s S L1 L P q o lin).CostLe (((9 : ℕ) : ℝ) / Fintype.card K + queryError s L1 L P) := by
  refine ⟨((3 : ℕ) : ℝ) * (((2 : ℕ) : ℝ) / Fintype.card K) + queryError s L1 L P, ?_, fun mu partials => ?_⟩
  · have := Game.density_le (zeros (runGroupPoly s S o)) 3
      (zeros_card_le_of_natDegree_le (runGroupPoly_natDegree s S o))
    have nine : ((9 : ℕ) : ℝ) / Fintype.card K =
        ((3 : ℕ) : ℝ) / Fintype.card K + ((3 : ℕ) : ℝ) * (((2 : ℕ) : ℝ) / Fintype.card K) := by
      push_cast; ring
    rw [nine]
    linarith
  refine Game.costLe_bind (Game.drawPoint_cost _ 2 (fun point => ?_) 3 []) fun alpha => ?_
  · refine (Finset.card_union_le _ _).trans ?_
    have := foldBad_card (fun u => partials 0 u - truePartials (keyGroup s S lin.lam) (keyPoint s lin) u) 3 point
    have := foldBad_card (fun u => partials 1 u - truePartials (runGroup S mu) (padded s.setupVars o.runPoint) u)
      3 point
    omega
  split_ifs
  · intro p1
    refine ⟨0, ?_, fun _ => le_rfl⟩
    rw [add_zero]
    set B := fun v : Fin p1.1.length => foldQueryBad (s.setupVars - 3) s.queries L q
      (fold alpha (keyGroup s S lin.lam)) (fold alpha (runGroup S mu)) (p1.1.get v)
    have each := fun v : Fin p1.1.length => foldQueryBad_card (s.setupVars - 3) s.queries L P q
      (fold alpha (keyGroup s S lin.lam)) (fold alpha (runGroup S mu)) (p1.1.get v) distinct
    have cover : (Finset.univ.filter fun words : Fin s.queries → Fin P =>
          lowBits L words ∈ Finset.univ.biUnion B) ⊆
        Finset.univ.biUnion fun v => Finset.univ.filter fun words => lowBits L words ∈ B v := by
      intro words member
      simpa only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_biUnion] using member
    calc ((Finset.univ.filter fun words : Fin s.queries → Fin P =>
            lowBits L words ∈ Finset.univ.biUnion B).card : ℝ) / Fintype.card (Fin s.queries → Fin P)
        ≤ (∑ v : Fin p1.1.length, ((Finset.univ.filter fun words => lowBits L words ∈ B v).card : ℝ)) /
            Fintype.card (Fin s.queries → Fin P) :=
          div_le_div_of_nonneg_right (by exact_mod_cast (Finset.card_le_card cover).trans Finset.card_biUnion_le)
            (by positivity)
      _ = ∑ v : Fin p1.1.length, ((Finset.univ.filter fun words => lowBits L words ∈ B v).card : ℝ) /
            Fintype.card (Fin s.queries → Fin P) := Finset.sum_div _ _ _
      _ ≤ ∑ _v : Fin p1.1.length, ((((2 ^ (s.setupVars - 3) - 1) * (P / L + 1) : ℕ) : ℝ) / P) ^ s.queries :=
          Finset.sum_le_sum fun v _ => each v
      _ = p1.1.length * ((((2 ^ (s.setupVars - 3) - 1) * (P / L + 1) : ℕ) : ℝ) / P) ^ s.queries := by simp
      _ ≤ queryError s L1 L P :=
          mul_le_mul_of_nonneg_right (by exact_mod_cast p1.2) (by positivity)
  · exact queryError_nonneg s L1 L P

theorem honestGame_ends (s : Shape) (S : Setup F) (L1 L P : ℕ) [NeZero L] [NeZero P] (q : ℕ → K)
    (o : EarlyOut F K) (lin : LinearOut K) :
    (honestGame s S L1 L P q o lin).Ends fun r => ∀ h, r = some h → h.p1.length ≤ L1 := by
  intro mu partials
  rw [Game.ends_bind]
  refine Game.Ends.of_forall (fun alpha => ?_) _
  split_ifs
  · intro p1 queries
    rintro h ⟨⟩
    exact p1.2
  · intro h none; cases none

end Honest

section Late

/-- The late stage's bad challenges: the block `β` and the late GKR, per candidate pair. -/
def lateCount (s : Shape) (L0 L1 : ℕ) : ℕ :=
  L0 * L1 * (2 ^ (s.slotVars + 1) + 2 ^ s.blockVars) +
    gkrCost (lateShape s) (L0 * L1) (fun _ => 0) (lateShape s).layers 1

theorem lateGame_cost (s : Shape) (S : Setup F) (lam : K) (τ : ℕ → K) (cands : List (P0Tables F))
    (p1 : List (P1Tables K)) (L0 L1 : ℕ) (short0 : cands.length ≤ L0) (short1 : p1.length ≤ L1)
    (nodes : ∀ d, d ≤ 3 → NodesDistinct K d) :
    (lateGame s S lam τ cands p1).CostLe ((lateCount s L0 L1 : ℝ) / Fintype.card K) := by
  have count : (Finset.univ : Finset (Fin cands.length × Fin p1.length)).card ≤ L0 * L1 := by
    simpa using Nat.mul_le_mul short0 short1
  unfold lateCount
  apply Game.draw_step _ _ _ _ (Game.density_le _ _ ?_)
  swap
  · refine (card_biUnion_le_mul _ _ (2 ^ (s.slotVars + 1) + 2 ^ s.blockVars) fun w _ =>
      zeros_card_le_of_natDegree_le ?_).trans (Nat.mul_le_mul_right _ count)
    unfold blockPoly
    exact (twoSidedPoly_natDegree _ _ _ _ _ _).trans (by simp)
  intro beta
  have small : ∀ t, (lateShape s).factors t ≤ 2 := by
    intro t; simp only [lateShape]; split_ifs <;> omega
  refine (Game.costLe_bind (gkr_cost Finset.univ _ _ (fun _ => 0) (fun _ _ => by simp) (nodes_of_small _ small nodes))
    (δ := 0) ?_).mono ?_
  · rintro (_ | claims)
    · exact le_rfl
    · intro pairs
      dsimp only
      split_ifs <;> exact le_rfl
  · rw [add_zero]
    exact div_le_div_of_nonneg_right (by exact_mod_cast gkrCost_mono _ _ count _ _) (by positivity)

end Late
section Main

/-- The layer-1 bad-challenge count. -/
def layer1Count (s : Shape) (L0 L1 : ℕ) : ℕ :=
  earlyCount s L0 + linearCount s L0 + 9 + lateCount s L0 L1

/-- The layer-1 error over a challenge field of `size` elements, with list
sizes `L0` (P0) and `L1` (P1), `L` query leaves and query words below `P`. -/
noncomputable def layer1Error (s : Shape) (L0 L1 L P size : ℕ) : ℝ :=
  (layer1Count s L0 L1 : ℝ) / size + queryError s L1 L P

theorem layer1Game_cost (s : Shape) (Φ : F[X]) (bar : ℕ → ℕ → F) (S : Setup F)
    (L0 L1 L P : ℕ) [NeZero L] [NeZero P] (q : ℕ → K)
    (distinct : ∀ i j, i < L → j < L → q i = q j → i = j) (nodes : ∀ d, d ≤ 3 → NodesDistinct K d)
    (degree : Φ.natDegree = s.D) (two : 2 ≤ s.D) :
    (layer1Game s Φ bar S L0 L1 L P q).CostLe (layer1Error s L0 L1 L P (Fintype.card K)) := by
  have rest : 0 ≤ (linearCount s L0 : ℝ) / Fintype.card K +
      ((((9 : ℕ) : ℝ) / Fintype.card K + queryError s L1 L P) + ((lateCount s L0 L1 : ℝ) / Fintype.card K + 0)) := by
    have := queryError_nonneg s L1 L P
    positivity
  unfold layer1Game
  intro st
  refine (Game.costLe_bind_of (earlyGame_cost s S st L0 nodes) (earlyGame_ends s S st L0)
    (δ := (linearCount s L0 : ℝ) / Fintype.card K +
      ((((9 : ℕ) : ℝ) / Fintype.card K + queryError s L1 L P) + ((lateCount s L0 L1 : ℝ) / Fintype.card K + 0)))
    ?_).mono ?_
  · rintro (_ | o) short
    · exact rest
    refine Game.costLe_bind (linearGame_cost s Φ bar S st o.candidates L0 (short o rfl) (nodes 1 (by norm_num))
      degree two) fun linear => ?_
    rcases linear with _ | lin
    · exact (by have := queryError_nonneg s L1 L P; positivity :
        (0 : ℝ) ≤ (((9 : ℕ) : ℝ) / Fintype.card K + queryError s L1 L P) +
          ((lateCount s L0 L1 : ℝ) / Fintype.card K + 0))
    refine Game.costLe_bind_of (honestGame_cost s S L1 L P q distinct o lin) (honestGame_ends s S L1 L P q o lin)
      fun honest short1 => ?_
    rcases honest with _ | h
    · exact (by positivity : (0 : ℝ) ≤ (lateCount s L0 L1 : ℝ) / Fintype.card K + 0)
    refine Game.costLe_bind (lateGame_cost s S lin.lam (laneWeight s bar lin.zeta) o.candidates h.p1 L0 L1
      (short o rfl) (short1 h rfl) nodes) fun late => ?_
    rcases late with _ | e
    · exact le_rfl
    · dsimp only
      split_ifs <;> exact le_rfl
  · apply le_of_eq
    unfold layer1Error layer1Count
    push_cast
    ring

end Main

end CompressionLean.Layer1
