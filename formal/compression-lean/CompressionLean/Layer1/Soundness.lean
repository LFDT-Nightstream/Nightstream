import CompressionLean.Layer1.Stages

/-!
Owns the whole layer-1 verifier game and what its accepting paths give
(`layer1Game_avoids`): along every path that avoids the bad sets, an accepted
transcript has a P0 candidate that is a CE(B) witness.

The prover names the CE(B) statement first: Rust absorbs the statement
words before the first layer-1 challenge, so a prover may choose it.

The two WHIR openings are idealized (`Opens`): the verifier accepts them when
one P0 candidate and one P1 candidate agree with every opened value. The
lists are the candidates the prover fixed at its commitments. The WHIR error
that this idealization hides is a named term of the final bound, not proved
here.
-/

namespace CompressionLean.Layer1

open Polynomial

variable {F K : Type} [Field F] [Field K] [Algebra F K] [Fintype K] [DecidableEq K]

local notation "ι" => algebraMap F K

/-- One layer-1 transcript. -/
structure Transcript (F K : Type) (t L : ℕ) where
  statement : Statement F
  early : EarlyOut F K
  linear : LinearOut K
  honest : HonestOut K t L
  late : LateOut K

/-- The idealized WHIR openings: one P0 candidate agrees with every P0
opening and one P1 candidate agrees with every P1 opening. -/
def Opens (s : Shape) (S : Setup F) (q : ℕ → K) {L : ℕ} (tr : Transcript F K s.queries L) : Prop :=
  ∃ w ∈ tr.early.candidates, ∃ v ∈ tr.honest.p1,
    EarlyConsistent s tr.early w ∧ LateConsistent s tr.late w v ∧
    tr.linear.zValue = mle (tr.linear.lanePart ++ tr.linear.blockPoint) (fun x => ι (w.z x)) ∧
    HonestConsistent s S q tr.early tr.linear tr.honest v ∧
    tr.linear.omegaValue = mle tr.linear.blockPoint v.omegaA

instance : Nonempty (Statement F) := ⟨⟨fun _ _ => 0, fun _ _ => 0, [], fun _ => 0, fun _ _ => 0⟩⟩

/-- The layer-1 verifier: the statement, early, linear, honest fold, late, then the openings. -/
noncomputable def layer1Game (s : Shape) (Φ : F[X]) (bar : ℕ → ℕ → F) (S : Setup F)
    (L0 L1 L P : ℕ) [NeZero L] [NeZero P] (q : ℕ → K) : Game (Option (Transcript F K s.queries L)) :=
  .recv (Statement F) fun st =>
  Game.bind (earlyGame s S st L0) fun early =>
  match early with
  | none => .pure none
  | some o =>
  Game.bind (linearGame s Φ bar S st o.candidates) fun linear =>
  match linear with
  | none => .pure none
  | some lin =>
  Game.bind (honestGame s S L1 L P q o lin) fun honest =>
  match honest with
  | none => .pure none
  | some h =>
  Game.bind (lateGame s S lin.lam (laneWeight s bar lin.zeta) o.candidates h.p1) fun late =>
  match late with
  | none => .pure none
  | some e =>
    have := Classical.propDecidable
    if Opens s S q ⟨st, o, lin, h, e⟩ then .pure (some ⟨st, o, lin, h, e⟩) else .pure none

/-- The size conditions of the layer-1 argument. Integers below `N` stay
distinct in `K`. -/
structure Shape.Sizes (s : Shape) (N : ℕ) : Prop where
  big_nodes : 5 ≤ N
  big_cube : 2 ^ s.cubeVars < N
  big_runs : 2 ^ s.runVars < N
  big_rows : 2 ^ s.rowVars ≤ N
  big_pairs : 2 ^ (s.slotVars + s.matrixVars) ≤ N
  big_blocks : 2 ^ s.blockVars + 1 < N
  slot_le_run : s.slotVars ≤ s.runVars
  matrices_le : s.matrices ≤ 2 ^ s.matrixVars
  public_le : s.publicBlocks ≤ s.blocks
  D_le : s.D ≤ 2 ^ s.laneVars
  blocks_le : s.blocks ≤ 2 ^ s.blockVars
  deep_cube : 1 ≤ s.cubeVars
  deep_runs : 1 ≤ s.runVars
  deep_rows : 1 ≤ s.rowVars
  deep_pairs : 1 ≤ s.slotVars + s.matrixVars
  deep_blocks : 1 ≤ s.blockVars
  fold_three : 3 ≤ s.setupVars
  fold_key : s.laneVars + s.blockVars ≤ s.setupVars
  fold_run : s.runVars ≤ s.setupVars

omit [Fintype K] [DecidableEq K] in
/-- Interpolation nodes `0, ..., d + 1` are distinct when integers below `N` are. -/
theorem nodes_of_cast (N : ℕ) (cast : ∀ a b : ℕ, a < N → b < N → (a : K) = b → a = b) (big : 5 ≤ N) :
    ∀ d, d ≤ 3 → NodesDistinct K d := by
  intro d small i j equal
  exact Fin.ext (cast i j (by omega) (by omega) equal)

omit [Fintype K] [DecidableEq K] in
/-- The early facts make a candidate's slot sums the true ones under `Ū`. -/
theorem ubar_of_facts (s : Shape) (S : Setup F) (st : Statement F) (w : P0Tables F)
    (facts : EarlyFacts (K := K) s S st w) (lam : K) (σ : ℕ) (σ_lt : σ < 2 ^ s.slotVars) :
    ubar s lam w.U σ = ubar s lam (trueU s S st) σ := by
  unfold ubar
  apply Finset.sum_congr rfl
  intro j inside
  obtain ⟨re, im⟩ := facts.2.2.1 j σ (Finset.mem_range.mp inside) σ_lt
  have runs : ∀ part : F × F → F, (∀ a : F, ∀ x : F × F, part (a • x) = a * part x) →
      ∑ k ∈ (Finset.range (2 ^ s.runVars)).filter (fun k => S.runMatrix k = j ∧ S.runSlot k = σ),
        coefColumn S k * ι (part (w.e k)) =
      ∑ k ∈ (Finset.range (2 ^ s.runVars)).filter (fun k => S.runMatrix k = j ∧ S.runSlot k = σ),
        ι (part (S.runCoef k • st.chi (S.runRow k))) := by
    intro part smul
    apply Finset.sum_congr rfl
    intro k member
    rw [facts.2.1 k (Finset.mem_range.mp (Finset.mem_filter.mp member).1), smul, map_mul]
    rfl
  rw [re, im, runs Prod.fst (fun _ _ => rfl), runs Prod.snd (fun _ _ => rfl), ← map_sum, ← map_sum,
    ← Prod.fst_sum, ← Prod.snd_sum]
  rfl

omit [Fintype K] [DecidableEq K] in
/-- The early facts make the late slot weights the true ones. -/
theorem slotWeights_of_facts (s : Shape) (S : Setup F) (st : Statement F) (w : P0Tables F)
    (facts : EarlyFacts (K := K) s S st w) (lam : K) (τ : ℕ → K) (b : ℕ) :
    slotWeights s S lam τ w.U b = slotWeights s S lam τ (trueU s S st) b := by
  unfold slotWeights
  apply Finset.sum_congr rfl
  intro x _
  rw [ubar_of_facts s S st w facts lam _ (Nat.mod_lt _ (by positivity))]

omit [Fintype K] [DecidableEq K] in
/-- A cell that is a slot value in `K` is the slot value in `F`. -/
theorem slotValue_of_map (H t : ℕ) (x : F) (equal : ι x = slotValue (K := K) H t) :
    x = slotValue (K := F) H t := by
  apply (algebraMap F K).injective
  rw [equal]
  simp [slotValue]

/-- Along every path that avoids the bad sets, an accepted transcript has a
P0 candidate that is a CE(B) witness. -/
theorem layer1Game_avoids (s : Shape) (Φ : F[X]) (bar : ℕ → ℕ → F) (S : Setup F)
    (L0 L1 L P : ℕ) [NeZero L] [NeZero P] (q : ℕ → K) (N : ℕ) (sizes : s.Sizes N)
    (cast : ∀ a b : ℕ, a < N → b < N → (a : K) = b → a = b) (fits : S.Fits s) (monic : Φ.Monic)
    (proper : Φ ≠ 1) :
    (layer1Game s Φ bar S L0 L1 L P q).Avoids fun r =>
      ∀ tr, r = some tr → ∃ w ∈ tr.early.candidates, Relation s Φ bar S tr.statement w.z := by
  have nodes := nodes_of_cast N cast sizes.big_nodes
  unfold layer1Game
  intro st
  rw [Game.avoids_bind]
  refine (earlyGame_avoids (K := K) s S st L0 nodes
    ⟨sizes.deep_cube, sizes.deep_runs, sizes.deep_rows, sizes.deep_pairs⟩).mono ?_
  rintro (_ | o) early
  · intro tr none; cases none
  obtain ⟨earlyChecks, earlyBad, earlyLengths, shared⟩ := early o rfl
  rw [Game.avoids_bind]
  refine (linearGame_avoids s Φ bar S st o.candidates (nodes 1 (by norm_num))).mono ?_
  rintro (_ | lin) linear
  · intro tr none; cases none
  obtain ⟨linearCheck, blockLength, laneLength, linearBad⟩ := linear lin rfl
  rw [Game.avoids_bind]
  refine (honestGame_avoids s S L1 L P q o lin).mono ?_
  rintro (_ | h) honest
  · intro tr none; cases none
  obtain ⟨hμ, alphaLength, alphaPath, partialCheck, queryBad⟩ := honest h rfl
  rw [Game.avoids_bind]
  refine (lateGame_avoids s S lin.lam (laneWeight s bar lin.zeta) o.candidates h.p1 nodes
    sizes.deep_blocks).mono ?_
  rintro (_ | e) late
  · intro tr none; cases none
  obtain ⟨lateChecks, pairLength, _, lateBad⟩ := late e rfl
  dsimp only
  split_ifs with opens
  swap
  · intro tr none; cases none
  rintro tr ⟨⟩
  obtain ⟨w, wm, v, vm, earlyOpen, lateOpen, zOpen, honestOpen, omegaOpen⟩ := opens
  obtain ⟨hβ, hL, hS, earlyRoot, blockPath⟩ := earlyBad w wm
  obtain ⟨hLam, hζ, first, second⟩ := linearBad w wm
  obtain ⟨lateβ, lateRoot⟩ := lateBad w wm v vm
  obtain ⟨d0, d1, d2, d3, d4⟩ := early_depths s
  -- The honest fold makes the key value and the run-point setup claims true.
  have keyLength : (keyPoint s lin).length = s.setupVars := by
    have := sizes.fold_key
    simp [keyPoint, padded, lanePoint_length, blockLength]; omega
  have runLength : (padded s.setupVars o.runPoint).length = s.setupVars := by
    have := earlyLengths 3 (by norm_num)
    simp only [d3] at this
    simp [padded, EarlyOut.runPoint, this]; have := sizes.fold_run; omega
  obtain ⟨key, setup⟩ := honest_world s S q o lin h v (by rw [keyLength]; exact sizes.fold_three)
    (by rw [runLength]; exact sizes.fold_three) (by simp [keyLength]) (by simp [runLength])
    alphaLength hμ alphaPath (queryBad v vm) partialCheck honestOpen
  -- The early stage makes the candidate's norm, run, slot and block tables true.
  have facts := early_world s S st o w N cast sizes.big_cube sizes.big_runs sizes.big_rows sizes.big_pairs
    fits sizes.slot_le_run sizes.matrices_le earlyChecks setup earlyOpen hβ hL hS earlyRoot shared
    (by rw [← d4]; exact earlyLengths 4 (by norm_num)) (by rw [← d3]; exact earlyLengths 3 (by norm_num))
    blockPath
  -- The late stage makes the P1 block weights the Eval_A part of `Ω`.
  have weights := late_world s S st lin.lam (laneWeight s bar lin.zeta) e w v N cast
    (by have := sizes.big_blocks; omega) lateChecks lateOpen facts lateβ lateRoot pairLength
  have omega : lin.omegaValue = mle lin.blockPoint (evalAPart s bar S st lin.lam lin.zeta) := by
    rw [omegaOpen]
    apply mle_congr
    intro b lt
    rw [blockLength] at lt
    rw [weights b lt, slotWeights_of_facts s S st w facts, slotWeights_eq s S st fits lin.lam _ b N cast
      (by have := sizes.big_blocks; omega)]
    · rfl
    · intro σ real
      have inside := fits.inside σ
      have lane := fits.lane_lt σ
      have blocks : S.slotBlock σ ≤ s.blocks := by
        by_contra more
        have : s.D * (s.blocks + 1) ≤ s.D * S.slotBlock σ := Nat.mul_le_mul_left _ (by omega)
        rw [Nat.mul_succ] at this
        omega
      have := sizes.blocks_le
      have := sizes.big_blocks
      omega
  -- The linear stage then forces every ring row.
  refine ⟨w, wm, fun x lt => ?_, linear_world s Φ bar S st fits sizes.public_le sizes.D_le sizes.blocks_le
    monic proper lin w linearCheck hLam hζ blockLength laneLength first second key omega zOpen⟩
  obtain ⟨t, t_lt, equal⟩ := facts.1 x lt
  exact ⟨t, t_lt, slotValue_of_map _ _ _ equal⟩

end CompressionLean.Layer1
