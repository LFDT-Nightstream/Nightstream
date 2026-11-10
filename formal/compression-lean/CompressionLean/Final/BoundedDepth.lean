import CompressionLean.Spec.BoundedDepth
import CompressionLean.Game.Compose
import CompressionLean.Final.Budget
import CompressionLean.Shrink.Budget

/-! The bound of a final program with one layer-1 verifier per slot (CW-6.5). When every accepted
witness gives a terminal environment whose two slot statements are the leaves of the positions it
reads (`Encodes`), a break has probability at most `calls` times the largest density
(`restore_sound`), and at most `calls / 2^t` under Plonky3's budget checks for both shapes
(`restore_budget`). -/

namespace CompressionLean.Final.BoundedDepth

open CompressionLean.Spec
open CompressionLean.Spec.BoundedDepth
open CompressionLean.Game
open CompressionLean NightstreamFPrime Polynomial
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F K E : Type} [Field F] [Field K] [Algebra F K] [Fintype K] [DecidableEq K]
  {t0 L0 t1 L1 : ℕ}

/-- Every accepted witness gives an environment of the terminal whose slot statements are the
leaves of the positions that it reads. -/
def Encodes (T : Shrink.Template F) (s : Shrink.Counts) (prog : StateWords → Shrink.Run F)
    (Terminal : StateWords → E → Prop) (claim0 claim1 : E → Layer1.Statement F)
    (g0 : Game (Option (Layer1.Transcript F K t0 L0))) (g1 : Game (Option (Layer1.Transcript F K t1 L1)))
    (Reads0 : StateWords → (ℕ → F) → g0.Pos → Prop) (Reads1 : StateWords → (ℕ → F) → g1.Pos → Prop) :
    Prop :=
  ∀ x w, ((prog x).ccs T s).Relation w → ∃ env, Terminal x env ∧
    (∀ p, Reads0 x w p → ∃ tr, Pos.node g0 p = .pure (some tr) ∧ tr.statement = claim0 env) ∧
    ∀ p, Reads1 x w p → ∃ tr, Pos.node g1 p = .pure (some tr) ∧ tr.statement = claim1 env

omit [Field K] [Algebra F K] [Fintype K] [DecidableEq K] in
/-- A break of a weaker claim is a break of a stronger one. -/
theorem restoreBreak_mono {Claim Claim' : StateWords → Prop} (imp : ∀ x, Claim x → Claim' x)
    (T : Shrink.Template F) (s : Shrink.Counts) (prog : StateWords → Shrink.Run F)
    (g0 : Game (Option (Layer1.Transcript F K t0 L0))) (g1 : Game (Option (Layer1.Transcript F K t1 L1)))
    (Reads0 : StateWords → (ℕ → F) → g0.Pos → Prop) (Reads1 : StateWords → (ℕ → F) → g1.Pos → Prop)
    (r : Outcome F K g0 g1) (broken : RestoreBreak Claim' T s prog g0 g1 Reads0 Reads1 r) :
    RestoreBreak Claim T s prog g0 g1 Reads0 Reads1 r := by
  obtain ⟨tr, accepted, reads, failed⟩ := broken
  exact ⟨tr, accepted, reads, fun claim => failed (imp _ claim)⟩

/-- What a layer-1 game's leaves avoid: an accepted transcript extracts. -/
abbrev Extracts (s : Layer1.Shape) (Φ : F[X]) (bar : ℕ → ℕ → F) (S : Layer1.Setup F) {t L : ℕ}
    (r : Option (Layer1.Transcript F K t L)) : Prop :=
  ∀ tr, r = some tr → ∃ w ∈ tr.early.candidates, Layer1.Relation s Φ bar S tr.statement w.z

omit [Field K] [Algebra F K] [Fintype K] [DecidableEq K] in
theorem holds_of_walks {Terminal : StateWords → E → Prop} {claim0 claim1 : E → Layer1.Statement F}
    {s0 s1 : Layer1.Shape} {Φ : F[X]} {bar : ℕ → ℕ → F} {S0 S1 : Layer1.Setup F}
    {T : Shrink.Template F} {s : Shrink.Counts} {prog : StateWords → Shrink.Run F}
    {g0 : Game (Option (Layer1.Transcript F K t0 L0))} {g1 : Game (Option (Layer1.Transcript F K t1 L1))}
    {Reads0 : StateWords → (ℕ → F) → g0.Pos → Prop} {Reads1 : StateWords → (ℕ → F) → g1.Pos → Prop}
    (encodes : Encodes T s prog Terminal claim0 claim1 g0 g1 Reads0 Reads1) {r : Outcome F K g0 g1}
    (walks : ∀ p b, r.2 p = some b → Pos.node (choice g0 g1) p = .pure b ∧
      Sum.elim (Extracts s0 Φ bar S0) (Extracts s1 Φ bar S1) b)
    {x : StateWords} {w : ℕ → F} (holds : ((prog x).ccs T s).Relation w)
    (read0 : ∃ p, Reads0 x w p ∧ ∃ b, r.2 (Pos.left g1 p) = some b)
    (read1 : ∃ p, Reads1 x w p ∧ ∃ b, r.2 (Pos.right g0 p) = some b) :
    Holds s0 s1 Φ bar S0 S1 Terminal claim0 claim1 x := by
  obtain ⟨env, terminal, leaves0, leaves1⟩ := encodes x w holds
  obtain ⟨p0, reads0, b0, walk0⟩ := read0
  obtain ⟨p1, reads1, b1, walk1⟩ := read1
  obtain ⟨tr0, leaf0, statement0⟩ := leaves0 p0 reads0
  obtain ⟨tr1, leaf1, statement1⟩ := leaves1 p1 reads1
  obtain ⟨node0, extracts0⟩ := walks _ _ walk0
  obtain ⟨node1, extracts1⟩ := walks _ _ walk1
  rw [node_left g1 leaf0] at node0
  rw [node_right g0 leaf1] at node1
  cases Game.pure.inj node0
  cases Game.pure.inj node1
  obtain ⟨v0, -, relation0⟩ := extracts0 tr0 rfl
  obtain ⟨v1, -, relation1⟩ := extracts1 tr1 rfl
  exact ⟨env, terminal, ⟨v0.z, statement0 ▸ relation0⟩, v1.z, statement1 ▸ relation1⟩

theorem restore_sound (s0 s1 : Layer1.Shape) (Φ : F[X]) (bar : ℕ → ℕ → F) (S0 S1 : Layer1.Setup F)
    (L0 L1 La Pa Lb Pb : ℕ) [NeZero La] [NeZero Pa] [NeZero Lb] [NeZero Pb] (q0 q1 : ℕ → K)
    (N0 N1 : ℕ) (sizes0 : s0.Sizes N0) (sizes1 : s1.Sizes N1)
    (cast0 : ∀ a b : ℕ, a < N0 → b < N0 → (a : K) = b → a = b)
    (cast1 : ∀ a b : ℕ, a < N1 → b < N1 → (a : K) = b → a = b)
    (distinct0 : ∀ i j, i < La → j < La → q0 i = q0 j → i = j)
    (distinct1 : ∀ i j, i < Lb → j < Lb → q1 i = q1 j → i = j)
    (fits0 : S0.Fits s0) (fits1 : S1.Fits s1) (monic : Φ.Monic)
    (degree0 : Φ.natDegree = s0.D) (degree1 : Φ.natDegree = s1.D) (two : 2 ≤ s0.D)
    (T : Shrink.Template F) (s : Shrink.Counts) (prog : StateWords → Shrink.Run F) (Ls : ℕ)
    (wf : T.WellFormed) (valid : ∀ x, (prog x).Valid T)
    (cast9 : ∀ a b : ℕ, a < 9 → b < 9 → (a : K) = b → a = b)
    (Terminal : StateWords → E → Prop) (claim0 claim1 : E → Layer1.Statement F)
    (Reads0 : StateWords → (ℕ → F) → (Layer1.layer1Game s0 Φ bar S0 L0 L1 La Pa q0).Pos → Prop)
    (Reads1 : StateWords → (ℕ → F) → (Layer1.layer1Game s1 Φ bar S1 L0 L1 Lb Pb q1).Pos → Prop)
    (encodes : Encodes T s prog Terminal claim0 claim1 (Layer1.layer1Game s0 Φ bar S0 L0 L1 La Pa q0)
      (Layer1.layer1Game s1 Φ bar S1 L0 L1 Lb Pb q1) Reads0 Reads1)
    (calls : ℕ)
    (prover : Game.Strategy (Outcome F K (Layer1.layer1Game s0 Φ bar S0 L0 L1 La Pa q0)
      (Layer1.layer1Game s1 Φ bar S1 L0 L1 Lb Pb q1))) :
    Game.prob prover
        (restorePair (Shrink.shrinkGame T s prog Ls)
          (choice (Layer1.layer1Game s0 Φ bar S0 L0 L1 La Pa q0) (Layer1.layer1Game s1 Φ bar S1 L0 L1 Lb Pb q1))
          calls [] [])
        (RestoreBreak (Holds s0 s1 Φ bar S0 S1 Terminal claim0 claim1) T s prog
          (Layer1.layer1Game s0 Φ bar S0 L0 L1 La Pa q0) (Layer1.layer1Game s1 Φ bar S1 L0 L1 Lb Pb q1)
          Reads0 Reads1) ≤
      calls * max (Shrink.shrinkError (Shrink.Layout.rowVars T s) (Shrink.Layout.cellVars T s) Ls (Fintype.card K))
        (max (Layer1.layer1Error s0 L0 L1 La Pa (Fintype.card K)) (Layer1.layer1Error s1 L0 L1 Lb Pb (Fintype.card K))) := by
  have proper : Φ ≠ 1 := by
    rintro rfl
    rw [natDegree_one] at degree0
    omega
  have shrinkCost := Shrink.shrinkGame_cost T s prog Ls (Shrink.shrinkNodes cast9 7 le_rfl)
    (Shrink.shrinkNodes cast9 1 (by norm_num))
  have cost0 := Layer1.layer1Game_cost s0 Φ bar S0 L0 L1 La Pa q0 distinct0
    (Layer1.nodes_of_cast N0 cast0 sizes0.big_nodes) degree0 two
  have cost1 := Layer1.layer1Game_cost s1 Φ bar S1 L0 L1 Lb Pb q1 distinct1
    (Layer1.nodes_of_cast N1 cast1 sizes1.big_nodes) degree1 (degree1 ▸ degree0 ▸ two)
  refine (prob_mono prover _ ?_).trans (restorePair_sound _ _ prover _
    (Sum.elim (Extracts s0 Φ bar S0) (Extracts s1 Φ bar S1)) _
    (Shrink.shrinkGame_avoids T s prog Ls wf valid (Shrink.shrinkNodes cast9 7 le_rfl)
      (Shrink.shrinkNodes cast9 1 (by norm_num)))
    (choice_avoids (Layer1.layer1Game_avoids s0 Φ bar S0 L0 L1 La Pa q0 N0 sizes0 cast0 fits0 monic proper)
      (Layer1.layer1Game_avoids s1 Φ bar S1 L0 L1 Lb Pb q1 N1 sizes1 cast1 fits1 monic proper))
    (shrinkCost.densityLe.mono (le_max_left _ _))
    (choice_density (cost0.densityLe.mono ((le_max_left _ _).trans (le_max_right _ _)))
      (cost1.densityLe.mono ((le_max_right _ _).trans (le_max_right _ _))))
    (shrinkCost.nonneg.trans (le_max_left _ _)) calls)
  rintro r ⟨tr, accepted, reads, notClaim⟩ ⟨shrink, walks⟩
  obtain ⟨w, member, holds⟩ := shrink _ accepted tr rfl
  obtain ⟨read0, read1⟩ := reads w member holds
  exact notClaim (holds_of_walks encodes walks holds read0 read1)

theorem restore_budget (s0 s1 : Layer1.Shape) (Φ : F[X]) (bar : ℕ → ℕ → F) (S0 S1 : Layer1.Setup F)
    (L0 L1 : ℕ) (h0 : 1 ≤ L0) (h1 : 1 ≤ L1) (big : 2 ^ 191 ≤ Fintype.card K)
    (index0 : s0.setupVars ≤ 35) (index1 : s1.setupVars ≤ 35) (t : ℕ)
    (T : Shrink.Template F) (s : Shrink.Counts) (prog : StateWords → Shrink.Run F) (Ls : ℕ)
    (p00 : (L0 : ℝ) * L1 * Layer1.rustDegrees s0 / 2 ^ 191 ≤ 1 / 2 ^ (t + 1))
    (p01 : (L1 : ℝ) * (1 / 2) ^ (s0.queries - 1) ≤ 1 / 2 ^ (t + 1))
    (p10 : (L0 : ℝ) * L1 * Layer1.rustDegrees s1 / 2 ^ 191 ≤ 1 / 2 ^ (t + 1))
    (p11 : (L1 : ℝ) * (1 / 2) ^ (s1.queries - 1) ≤ 1 / 2 ^ (t + 1))
    (rust : (Ls : ℝ) * Shrink.rustDegrees (Shrink.Layout.rowVars T s) (Shrink.Layout.cellVars T s) / 2 ^ 191 ≤
      1 / 2 ^ t)
    (q0 q1 : ℕ → K) (N0 N1 : ℕ) (sizes0 : s0.Sizes N0) (sizes1 : s1.Sizes N1)
    (cast0 : ∀ a b : ℕ, a < N0 → b < N0 → (a : K) = b → a = b)
    (cast1 : ∀ a b : ℕ, a < N1 → b < N1 → (a : K) = b → a = b)
    (distinct0 : ∀ i j, i < 2 ^ (s0.setupVars - 2) → j < 2 ^ (s0.setupVars - 2) → q0 i = q0 j → i = j)
    (distinct1 : ∀ i j, i < 2 ^ (s1.setupVars - 2) → j < 2 ^ (s1.setupVars - 2) → q1 i = q1 j → i = j)
    (fits0 : S0.Fits s0) (fits1 : S1.Fits s1) (monic : Φ.Monic)
    (degree0 : Φ.natDegree = s0.D) (degree1 : Φ.natDegree = s1.D) (two : 2 ≤ s0.D)
    (wf : T.WellFormed) (valid : ∀ x, (prog x).Valid T)
    (cast9 : ∀ a b : ℕ, a < 9 → b < 9 → (a : K) = b → a = b)
    (Terminal : StateWords → E → Prop) (claim0 claim1 : E → Layer1.Statement F)
    (Reads0 : StateWords → (ℕ → F) →
      (Layer1.layer1Game s0 Φ bar S0 L0 L1 (2 ^ (s0.setupVars - 2)) goldilocksModulus q0).Pos → Prop)
    (Reads1 : StateWords → (ℕ → F) →
      (Layer1.layer1Game s1 Φ bar S1 L0 L1 (2 ^ (s1.setupVars - 2)) goldilocksModulus q1).Pos → Prop)
    (encodes : Encodes T s prog Terminal claim0 claim1
      (Layer1.layer1Game s0 Φ bar S0 L0 L1 (2 ^ (s0.setupVars - 2)) goldilocksModulus q0)
      (Layer1.layer1Game s1 Φ bar S1 L0 L1 (2 ^ (s1.setupVars - 2)) goldilocksModulus q1) Reads0 Reads1)
    (calls : ℕ)
    (prover : Game.Strategy (Outcome F K
      (Layer1.layer1Game s0 Φ bar S0 L0 L1 (2 ^ (s0.setupVars - 2)) goldilocksModulus q0)
      (Layer1.layer1Game s1 Φ bar S1 L0 L1 (2 ^ (s1.setupVars - 2)) goldilocksModulus q1))) :
    Game.prob prover
        (restorePair (Shrink.shrinkGame T s prog Ls)
          (choice (Layer1.layer1Game s0 Φ bar S0 L0 L1 (2 ^ (s0.setupVars - 2)) goldilocksModulus q0)
            (Layer1.layer1Game s1 Φ bar S1 L0 L1 (2 ^ (s1.setupVars - 2)) goldilocksModulus q1))
          calls [] [])
        (RestoreBreak (Holds s0 s1 Φ bar S0 S1 Terminal claim0 claim1) T s prog
          (Layer1.layer1Game s0 Φ bar S0 L0 L1 (2 ^ (s0.setupVars - 2)) goldilocksModulus q0)
          (Layer1.layer1Game s1 Φ bar S1 L0 L1 (2 ^ (s1.setupVars - 2)) goldilocksModulus q1)
          Reads0 Reads1) ≤
      calls / 2 ^ t := by
  refine (restore_sound s0 s1 Φ bar S0 S1 L0 L1 _ _ _ _ q0 q1 N0 N1 sizes0 sizes1 cast0 cast1 distinct0
    distinct1 fits0 fits1 monic degree0 degree1 two T s prog Ls wf valid cast9 Terminal claim0 claim1
    Reads0 Reads1 encodes calls prover).trans ?_
  have layer0 := Final.layer1_budget s0 L0 L1 (Fintype.card K) t h0 h1 big index0 0 0 (by simpa using p00)
    (by simpa using p01)
  have layer1 := Final.layer1_budget s1 L0 L1 (Fintype.card K) t h0 h1 big index1 0 0 (by simpa using p10)
    (by simpa using p11)
  have shrink := Shrink.shrink_budget _ _ Ls (Fintype.card K) t big 0 (by simpa using rust)
  rw [zero_add, zero_add] at layer0 layer1
  rw [zero_add] at shrink
  rw [div_eq_mul_one_div]
  exact mul_le_mul_of_nonneg_left (max_le shrink (max_le layer0 layer1)) (Nat.cast_nonneg _)

end CompressionLean.Final.BoundedDepth
