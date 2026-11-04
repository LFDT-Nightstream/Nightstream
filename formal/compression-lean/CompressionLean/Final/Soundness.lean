import CompressionLean.Game.Compose
import CompressionLean.Final.Claim
import CompressionLean.Final.Budget
import CompressionLean.Shrink.Budget
import CompressionLean.Spec.Bound

/-! Model: random oracle for the in-program hash; idealized WHIR openings. The read condition does not follow from this model: Fiat–Shamir inside the program is assumed (`TRUST.md` §3). -/

namespace CompressionLean.Final

open CompressionLean.Spec
open CompressionLean.Game
open CompressionLean NightstreamFPrime Polynomial
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F K : Type} [Field F] [Field K] [Algebra F K] [Fintype K] [DecidableEq K]

omit [Field K] [Algebra F K] [Fintype K] [DecidableEq K] in
theorem finalRestoreBreak_mono {t L : ℕ} {Claim Claim' : StateWords → Prop} (imp : ∀ x, Claim x → Claim' x)
    (T : Shrink.Template F) (s : Shrink.Counts) (prog : StateWords → Shrink.Run F)
    (g : Game (Option (Layer1.Transcript F K t L))) (Reads : StateWords → (ℕ → F) → g.Pos → Prop)
    (r : RestoreOutcome F K g) (break_ : FinalRestoreBreak Claim' T s prog g Reads r) :
    FinalRestoreBreak Claim T s prog g Reads r := by
  obtain ⟨tr, accepted, reads, notClaim⟩ := break_
  exact ⟨tr, accepted, reads, fun claim => notClaim (imp _ claim)⟩

theorem final_restore_sound (s1 : Layer1.Shape) (Φ : F[X]) (bar : ℕ → ℕ → F) (S : Layer1.Setup F)
    (L0 L1 L P : ℕ) [NeZero L] [NeZero P] (q : ℕ → K) (N : ℕ) (sizes : s1.Sizes N)
    (cast : ∀ a b : ℕ, a < N → b < N → (a : K) = b → a = b)
    (distinct : ∀ i j, i < L → j < L → q i = q j → i = j) (fits : S.Fits s1) (monic : Φ.Monic)
    (degree : Φ.natDegree = s1.D) (two : 2 ≤ s1.D)
    (T : Shrink.Template F) (s : Shrink.Counts) (prog : StateWords → Shrink.Run F) (Ls : ℕ)
    (wf : T.WellFormed) (valid : ∀ x, (prog x).Valid T)
    (cast9 : ∀ a b : ℕ, a < 9 → b < 9 → (a : K) = b → a = b)
    (authority : Lifecycle.VerifierContext.Authority)
    (Reads : StateWords → (ℕ → F) → (Layer1.layer1Game s1 Φ bar S L0 L1 L P q).Pos → Prop)
    (encodes : ProgramEncodes T s prog authority (Layer1.layer1Game s1 Φ bar S L0 L1 L P q) Reads)
    (calls : ℕ) (prover : Game.Strategy (RestoreOutcome F K (Layer1.layer1Game s1 Φ bar S L0 L1 L P q))) :
    Game.prob prover
        (restorePair (Shrink.shrinkGame T s prog Ls) (Layer1.layer1Game s1 Φ bar S L0 L1 L P q) calls [] [])
        (FinalRestoreBreak (TerminalClaim s1 Φ bar S authority) T s prog (Layer1.layer1Game s1 Φ bar S L0 L1 L P q)
          Reads) ≤
      calls * max (Shrink.shrinkError (Shrink.Layout.rowVars T s) (Shrink.Layout.cellVars T s) Ls (Fintype.card K))
        (Layer1.layer1Error s1 L0 L1 L P (Fintype.card K)) := by
  have proper : Φ ≠ 1 := by
    rintro rfl
    rw [natDegree_one] at degree
    omega
  have shrinkCost := Shrink.shrinkGame_cost T s prog Ls (Shrink.shrinkNodes cast9 7 le_rfl)
    (Shrink.shrinkNodes cast9 1 (by norm_num))
  have layer1Cost := Layer1.layer1Game_cost s1 Φ bar S L0 L1 L P q distinct
    (Layer1.nodes_of_cast N cast sizes.big_nodes) degree two
  refine (prob_mono prover _ ?_).trans (restorePair_sound _ _ prover _ _ _
    (Shrink.shrinkGame_avoids T s prog Ls wf valid (Shrink.shrinkNodes cast9 7 le_rfl)
      (Shrink.shrinkNodes cast9 1 (by norm_num)))
    (Layer1.layer1Game_avoids s1 Φ bar S L0 L1 L P q N sizes cast fits monic proper)
    (shrinkCost.densityLe.mono (le_max_left _ _)) (layer1Cost.densityLe.mono (le_max_right _ _))
    (shrinkCost.nonneg.trans (le_max_left _ _)) calls)
  rintro r ⟨tr, accepted, reads, notClaim⟩ ⟨shrink, layer1⟩
  obtain ⟨w, member, holds⟩ := shrink _ accepted tr rfl
  obtain ⟨p, read, b, walks⟩ := reads w member holds
  obtain ⟨leaf, extracts⟩ := layer1 p b walks
  exact notClaim (terminalClaim_of s1 Φ bar S encodes holds read fun r' node => by
    rw [leaf] at node
    rw [← Game.pure.inj node]
    exact extracts)

theorem final_restore_budget (s1 : Layer1.Shape) (Φ : F[X]) (bar : ℕ → ℕ → F) (S : Layer1.Setup F)
    (L0 L1 : ℕ) (h0 : 1 ≤ L0) (h1 : 1 ≤ L1) (big : 2 ^ 191 ≤ Fintype.card K)
    (index : s1.setupVars ≤ 35) (t : ℕ)
    (T : Shrink.Template F) (s : Shrink.Counts) (prog : StateWords → Shrink.Run F) (Ls : ℕ)
    (p0 : (L0 : ℝ) * L1 * Layer1.rustDegrees s1 / 2 ^ 191 ≤ 1 / 2 ^ (t + 1))
    (p1 : (L1 : ℝ) * (1 / 2) ^ (s1.queries - 1) ≤ 1 / 2 ^ (t + 1))
    (rust : (Ls : ℝ) * Shrink.rustDegrees (Shrink.Layout.rowVars T s) (Shrink.Layout.cellVars T s) / 2 ^ 191 ≤
      1 / 2 ^ t)
    (q : ℕ → K) (N : ℕ) (sizes : s1.Sizes N) (cast : ∀ a b : ℕ, a < N → b < N → (a : K) = b → a = b)
    (distinct : ∀ i j, i < 2 ^ (s1.setupVars - 2) → j < 2 ^ (s1.setupVars - 2) → q i = q j → i = j)
    (fits : S.Fits s1) (monic : Φ.Monic) (degree : Φ.natDegree = s1.D) (two : 2 ≤ s1.D)
    (wf : T.WellFormed) (valid : ∀ x, (prog x).Valid T)
    (cast9 : ∀ a b : ℕ, a < 9 → b < 9 → (a : K) = b → a = b)
    (authority : Lifecycle.VerifierContext.Authority)
    (Reads : StateWords → (ℕ → F) →
      (Layer1.layer1Game s1 Φ bar S L0 L1 (2 ^ (s1.setupVars - 2)) goldilocksModulus q).Pos → Prop)
    (encodes : ProgramEncodes T s prog authority
      (Layer1.layer1Game s1 Φ bar S L0 L1 (2 ^ (s1.setupVars - 2)) goldilocksModulus q) Reads)
    (calls : ℕ)
    (prover : Game.Strategy
      (RestoreOutcome F K (Layer1.layer1Game s1 Φ bar S L0 L1 (2 ^ (s1.setupVars - 2)) goldilocksModulus q))) :
    Game.prob prover
        (restorePair (Shrink.shrinkGame T s prog Ls)
          (Layer1.layer1Game s1 Φ bar S L0 L1 (2 ^ (s1.setupVars - 2)) goldilocksModulus q) calls [] [])
        (FinalRestoreBreak (TerminalClaim s1 Φ bar S authority) T s prog
          (Layer1.layer1Game s1 Φ bar S L0 L1 (2 ^ (s1.setupVars - 2)) goldilocksModulus q) Reads) ≤
      calls / 2 ^ t := by
  refine (final_restore_sound s1 Φ bar S L0 L1 _ _ q N sizes cast distinct fits monic degree two T s prog Ls
    wf valid cast9 authority Reads encodes calls prover).trans ?_
  have layer1 := layer1_budget s1 L0 L1 (Fintype.card K) t h0 h1 big index 0 0 (by simpa using p0)
    (by simpa using p1)
  have shrink := Shrink.shrink_budget _ _ Ls (Fintype.card K) t big 0 (by simpa using rust)
  rw [zero_add, zero_add] at layer1
  rw [zero_add] at shrink
  rw [div_eq_mul_one_div]
  exact mul_le_mul_of_nonneg_left (max_le shrink layer1) (Nat.cast_nonneg _)

end CompressionLean.Final
