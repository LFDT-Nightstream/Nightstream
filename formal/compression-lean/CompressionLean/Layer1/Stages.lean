import CompressionLean.Layer1.HonestStage

namespace CompressionLean.Layer1

open CompressionLean.Lib
open Polynomial

variable {F K : Type} [Field F] [Field K] [Algebra F K] [Fintype K] [DecidableEq K]

local notation "ι" => algebraMap F K

theorem mem_get {α : Type} {l : List α} {a : α} (member : a ∈ l) : ∃ i : Fin l.length, l.get i = a :=
  List.mem_iff_get.mp member

omit [Fintype K] [DecidableEq K] in
theorem nodes_of_small (s : GkrShape) (small : ∀ t, s.factors t ≤ 2) (nodes : ∀ d, d ≤ 3 → NodesDistinct K d) :
    ∀ k, 1 ≤ k → k ≤ s.layers → NodesDistinct K (s.degree k - 1) := by
  intro k _ _
  apply nodes
  have : s.degree k ≤ 4 := s.degree_le k 4 (by norm_num) fun t _ => by
    unfold GkrShape.partFactors
    split
    · have := small t; omega
    · omega
  omega

section Early

def EarlyGood (s : Shape) (S : Setup F) (st : Statement F) (o : EarlyOut F K) : Prop :=
  EarlyChecks s st o ∧
    (∀ w ∈ o.candidates,
      o.challenges.beta ∉ zeros (normData s o.histogram w) ∧
      o.challenges.lookup ∉ (lookupData s S st w).bad ∧
      o.challenges.scatter ∉ (scatterData s S w).bad ∧
      ((∃ t, t < 5 ∧ o.gkr.roots t ≠ trueRoot (earlyShape s) (earlyBatch s S st o.challenges w) t) →
        LeafFalse (earlyShape s) (earlyBatch s S st o.challenges w) o.gkr) ∧
      Path (blockTest s S w) [] o.runPoint) ∧
    (∀ t, t < 5 → (o.gkr.leafPoint t).length = (earlyShape s).depth t) ∧
    o.gkr.leafPoint 1 = o.gkr.leafPoint 3

omit [Fintype K] [DecidableEq K] in
theorem earlyBatch_fits (s : Shape) (S : Setup F) (st : Statement F) (ch : EarlyChallenges K)
    (w : P0Tables F) : (earlyBatch s S st ch w).Fits (earlyShape s) := by
  intro t tree
  have cases : t = 0 ∨ t = 1 ∨ t = 2 ∨ t = 3 ∨ t = 4 := by simp only [earlyShape] at tree; omega
  obtain ⟨f0, f1, f2, f3, f4⟩ := early_factors s
  rcases cases with rfl | rfl | rfl | rfl | rfl
  · rw [earlyBatch_0, f0]; rfl
  · rw [earlyBatch_1, f1]; rfl
  · rw [earlyBatch_2, f2]; rfl
  · rw [earlyBatch_3, f3]; rfl
  · rw [earlyBatch_4, f4]; rfl

theorem earlyGame_avoids (s : Shape) (S : Setup F) (st : Statement F) (L0 : ℕ)
    (nodes : ∀ d, d ≤ 3 → NodesDistinct K d)
    (deep : 1 ≤ s.cubeVars ∧ 1 ≤ s.runVars ∧ 1 ≤ s.rowVars ∧ 1 ≤ s.slotVars + s.matrixVars) :
    (earlyGame (K := K) s S st L0).Avoids fun r => ∀ o, r = some o → EarlyGood s S st o := by
  intro cands hist beta hβ lookup hL scatter hS
  rw [Game.avoids_bind]
  have small : ∀ t, (earlyShape s).factors t ≤ 2 := by
    intro t; simp only [earlyShape]; split_ifs <;> omega
  have depths : ∀ t, t < (earlyShape s).trees → 1 ≤ (earlyShape s).depth t := by
    intro t _; simp only [earlyShape]; split_ifs <;> omega
  refine (gkr_avoids Finset.univ _ (earlyExtra s S cands.1) (fun w _ => earlyBatch_fits s S st _ _)
    depths (nodes_of_small _ small nodes)).mono ?_
  rintro r (rfl | ⟨fin, rfl, roots, structural, shared⟩)
  · intro o none; cases none
  · intro values
    dsimp only
    split_ifs with checks
    · rintro o ⟨⟩
      refine ⟨checks, ?_, fun t lt => (structural t (by simpa [earlyShape] using lt)).1,
        shared 1 3 (by simp [earlyShape]) (by simp [earlyShape])⟩
      intro w member
      obtain ⟨i, rfl⟩ := mem_get member
      refine ⟨fun hit => hβ (Finset.mem_biUnion.mpr ⟨i, Finset.mem_univ _, hit⟩),
        fun hit => hL (Finset.mem_biUnion.mpr ⟨i, Finset.mem_univ _, hit⟩),
        fun hit => hS (Finset.mem_biUnion.mpr ⟨i, Finset.mem_univ _, hit⟩),
        fun ⟨t, lt, ne⟩ => roots i (Finset.mem_univ _) ⟨t, by simpa [earlyShape] using lt, ne⟩, ?_⟩
      have path := (structural 3 (by simp [earlyShape])).2
      have depth3 : (earlyShape s).depth 3 = s.runVars := by simp [earlyShape]
      rw [depth3] at path
      refine Path.mono (fun q => ?_) _ _ path
      unfold earlyExtra
      rw [ite_eq_left rfl]
      exact Finset.subset_biUnion_of_mem (fun w : Fin cands.1.length => blockTest s S (cands.1.get w) q)
        (Finset.mem_univ i)
    · intro o none; cases none

end Early

section Linear

def LinearGood (s : Shape) (Φ : F[X]) (bar : ℕ → ℕ → F) (S : Setup F) (st : Statement F)
    (cands : List (P0Tables F)) (lin : LinearOut K) : Prop :=
  LinearCheck s bar st lin ∧ lin.blockPoint.length = s.blockVars ∧ lin.lanePart.length = s.laneVars ∧
    ∀ w ∈ cands,
      lin.lam ∉ (relationRows s bar S st).lambdaBad Φ (witnessBlock s w.z) ∧
      lin.zeta ∉ (relationRows s bar S st).zetaBad Φ (witnessBlock s w.z) lin.lam (quotientPoly s lin.quotient) ∧
      (liftedTarget s Φ bar S st lin.lam lin.zeta (quotientPoly s lin.quotient) ≠
          combTruth s.blockVars (blockTables s bar S st lin.lam lin.zeta w.z) productComb [] →
        lin.middle ≠ combTruth s.blockVars (blockTables s bar S st lin.lam lin.zeta w.z) productComb lin.blockPoint) ∧
      (lin.middle ≠ combTruth s.laneVars (laneTables s bar S st lin.lam lin.zeta w.z lin.blockPoint) productComb [] →
        lin.last ≠ combTruth s.laneVars (laneTables s bar S st lin.lam lin.zeta w.z lin.blockPoint) productComb
          lin.lanePart)

theorem linearGame_avoids (s : Shape) (Φ : F[X]) (bar : ℕ → ℕ → F) (S : Setup F) (st : Statement F)
    (cands : List (P0Tables F)) (nodes : NodesDistinct K 1) :
    (linearGame (K := K) s Φ bar S st cands).Avoids fun r =>
      ∀ lin, r = some lin → LinearGood s Φ bar S st cands lin := by
  intro lam hLam quotient zeta hζ
  rw [Game.avoids_bind]
  refine (Game.sumcheck_avoids Finset.univ 1 nodes
    (fun w : Fin cands.length => combTruth s.blockVars (blockTables s bar S st lam zeta (cands.get w).z) productComb)
    _ (fun _ => ∅) s.blockVars
    (fun w _ => combRound_truth s.blockVars 1 _ _ productComb_degree) s.blockVars [] _ (by simp)).mono ?_
  rintro ⟨blockPoint, middle⟩ ⟨⟨cs, shape, length, _⟩, first⟩
  rw [Game.avoids_bind]
  refine (Game.sumcheck_avoids Finset.univ 1 nodes
    (fun w : Fin cands.length =>
      combTruth s.laneVars (laneTables s bar S st lam zeta (cands.get w).z blockPoint) productComb)
    _ (fun _ => ∅) s.laneVars
    (fun w _ => combRound_truth s.laneVars 1 _ _ productComb_degree) s.laneVars [] _ (by simp)).mono ?_
  rintro ⟨lanePart, last⟩ ⟨⟨cs', shape', length', _⟩, second⟩
  intro finals
  simp only at shape shape' first second
  dsimp only
  split_ifs with check
  · rintro lin ⟨⟩
    refine ⟨check, by simp [shape, length], by simp [shape', length'], ?_⟩
    intro w member
    obtain ⟨i, rfl⟩ := mem_get member
    exact ⟨fun hit => hLam (Finset.mem_biUnion.mpr ⟨i, Finset.mem_univ _, hit⟩),
      fun hit => hζ (Finset.mem_biUnion.mpr ⟨i, Finset.mem_univ _, hit⟩),
      first i (Finset.mem_univ _), second i (Finset.mem_univ _)⟩
  · intro lin none; cases none

end Linear

section Honest

def HonestGood (s : Shape) (S : Setup F) (q : ℕ → K) (o : EarlyOut F K) (lin : LinearOut K) {L : ℕ}
    (h : HonestOut K s.queries L) : Prop :=
  h.mu ∉ zeros (runGroupPoly s S o) ∧ h.alpha.length = 3 ∧
    Path (alphaBad s S o lin h.mu h.partials) [] h.alpha ∧
    (mle ((keyPoint s lin).take 3) (h.partials 0) = lin.keyValue ∧
      mle ((padded s.setupVars o.runPoint).take 3) (h.partials 1) = runValue s S o h.mu) ∧
    ∀ v ∈ h.p1, h.queries ∉ foldQueryBad (s.setupVars - 3) s.queries L q
      (fold h.alpha (keyGroup s S lin.lam)) (fold h.alpha (runGroup S h.mu)) v

theorem honestGame_avoids (s : Shape) (S : Setup F) (L1 L P : ℕ) [NeZero L] [NeZero P] (q : ℕ → K)
    (o : EarlyOut F K) (lin : LinearOut K) :
    (honestGame s S L1 L P q o lin).Avoids fun r => ∀ h, r = some h → HonestGood s S q o lin h := by
  intro mu hμ partials
  rw [Game.avoids_bind]
  refine (Game.drawPoint_avoids _ 3 []).mono ?_
  rintro alpha ⟨cs, rfl, length, path⟩
  split_ifs with check
  · intro p1 words hq
    rintro h ⟨⟩
    refine ⟨hμ, by simpa using length, by simpa using path, check, ?_⟩
    intro v member
    obtain ⟨i, rfl⟩ := mem_get member
    exact fun hit =>
      hq (Finset.mem_filter.mpr ⟨Finset.mem_univ _, Finset.mem_biUnion.mpr ⟨i, Finset.mem_univ _, hit⟩⟩)
  · intro h none; cases none

end Honest

section Late

noncomputable def lateGame (s : Shape) (S : Setup F) (lam : K) (τ : ℕ → K)
    (cands : List (P0Tables F)) (p1 : List (P1Tables K)) : Game (Option (LateOut K)) :=
  .draw K (Finset.univ.biUnion fun w : Fin cands.length × Fin p1.length =>
      zeros (blockPoly s S lam τ (cands.get w.1) (p1.get w.2))) fun beta =>
  Game.bind (gkr (lateShape s) Finset.univ
      (fun w : Fin cands.length × Fin p1.length => lateBatch s S lam τ beta (cands.get w.1) (p1.get w.2))
      (fun _ _ => ∅))
    fun result =>
      match result with
      | none => .pure none
      | some claims =>
          .recv (ℕ → K × K) fun pairs =>
            have := Classical.propDecidable
            if LateChecks s S lam τ ⟨beta, claims, pairs⟩ then .pure (some ⟨beta, claims, pairs⟩)
            else .pure none

def LateGood (s : Shape) (S : Setup F) (lam : K) (τ : ℕ → K) (cands : List (P0Tables F))
    (p1 : List (P1Tables K)) (o : LateOut K) : Prop :=
  LateChecks s S lam τ o ∧ (o.gkr.leafPoint 0).length = s.slotVars + 1 ∧
    (o.gkr.leafPoint 1).length = s.blockVars ∧
    ∀ w ∈ cands, ∀ v ∈ p1,
      o.beta ∉ zeros (blockPoly s S lam τ w v) ∧
      ((∃ t, t < 2 ∧ o.gkr.roots t ≠ trueRoot (lateShape s) (lateBatch s S lam τ o.beta w v) t) →
        LeafFalse (lateShape s) (lateBatch s S lam τ o.beta w v) o.gkr)

omit [Fintype K] [DecidableEq K] in
theorem lateBatch_fits (s : Shape) (S : Setup F) (lam : K) (τ : ℕ → K) (beta : K)
    (w : P0Tables F) (v : P1Tables K) : (lateBatch s S lam τ beta w v).Fits (lateShape s) := by
  intro t tree
  have cases : t = 0 ∨ t = 1 := by simp only [lateShape] at tree; omega
  rcases cases with rfl | rfl <;> rfl

theorem lateGame_avoids (s : Shape) (S : Setup F) (lam : K) (τ : ℕ → K) (cands : List (P0Tables F))
    (p1 : List (P1Tables K)) (nodes : ∀ d, d ≤ 3 → NodesDistinct K d) (deep : 1 ≤ s.blockVars) :
    (lateGame s S lam τ cands p1).Avoids fun r => ∀ o, r = some o → LateGood s S lam τ cands p1 o := by
  intro beta hβ
  rw [Game.avoids_bind]
  have small : ∀ t, (lateShape s).factors t ≤ 2 := by
    intro t; simp only [lateShape]; split_ifs <;> omega
  have depths : ∀ t, t < (lateShape s).trees → 1 ≤ (lateShape s).depth t := by
    intro t _; simp only [lateShape]; split_ifs <;> omega
  refine (gkr_avoids Finset.univ _ (fun _ _ => ∅) (fun w _ => lateBatch_fits s S lam τ beta _ _)
    depths (nodes_of_small _ small nodes)).mono ?_
  rintro r (rfl | ⟨fin, rfl, roots, structural, _⟩)
  · intro o none; cases none
  · intro pairs
    dsimp only
    split_ifs with checks
    · rintro o ⟨⟩
      refine ⟨checks, by simpa [lateShape] using (structural 0 (by simp [lateShape])).1,
        by simpa [lateShape] using (structural 1 (by simp [lateShape])).1, ?_⟩
      intro w wm v vm
      obtain ⟨i, rfl⟩ := mem_get wm
      obtain ⟨j, rfl⟩ := mem_get vm
      exact ⟨fun hit => hβ (Finset.mem_biUnion.mpr ⟨(i, j), Finset.mem_univ _, hit⟩),
        fun ⟨t, lt, ne⟩ => roots (i, j) (Finset.mem_univ _) ⟨t, by simpa [lateShape] using lt, ne⟩⟩
    · intro o none; cases none

end Late

end CompressionLean.Layer1
