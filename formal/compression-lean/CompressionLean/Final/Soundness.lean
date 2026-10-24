import CompressionLean.Game.Compose
import CompressionLean.Final.Claim
import CompressionLean.Layer1.FiatShamir
import CompressionLean.Shrink.Budget

/-!
Owns the end-to-end soundness of the final verifier (`CompressionKey::verify`,
`lifecycle/finish.rs:131-145`). The shrink layer, layer 1 and the terminal
package chain through the encoding premise (`Claim.lean`, `ProgramEncodes`):

- the shrink verifier accepts `x`, so a candidate `w` satisfies the CCS of
  program `x`, unless a shrink challenge is bad (`Shrink/Soundness.lean`);
- `w` holds a terminal assignment for `x` and accepting layer-1 paths for
  its parent claim (`ProgramEncodes`);
- the leaf of such a path has a P0 candidate that is a CE(B) witness of the
  parent claim, unless a layer-1 challenge is bad (`Layer1/Soundness.lean`);
- so the terminal claim for `x` holds (`TerminalClaim`).

Both layers' games run in one experiment (`Compose.lean`), so one bad-set
argument covers both, and layer 1's prover is not built from the shrink
prover. The shrink extraction may return any candidate that satisfies the
program's CCS, so the read condition covers every such candidate.

Read condition: each such candidate has a reading that the layer-1 verifier
ran. In sequence (`final_sound`), the reading ends at the leaf where layer 1
ended. Under Fiat–Shamir (`final_restore_sound`), the reading's walk under the
layer-1 log accepts, so each of its challenges is an oracle answer. This is
the random-oracle model for the hash inside the program, and the WHIR model
for the openings inside it.
-/

namespace CompressionLean.Final

open CompressionLean.Game
open CompressionLean NightstreamFPrime Polynomial
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F K : Type} [Field F] [Field K] [Algebra F K] [Fintype K] [DecidableEq K]

/-- The outcome in sequence: layer 1's result, then the shrink's. -/
abbrev Outcome (F K : Type) (t L : ℕ) :=
  Option (Layer1.Transcript F K t L) × Option (Shrink.Transcript StateWords F K)

/-- In sequence: the final verifier accepts the statement `x = tr.inst`, the
claim `Claim x` is false, and every program witness among the shrink
candidates has a reading that ends where layer 1 ended. -/
def FinalBreak {t L : ℕ} (Claim : StateWords → Prop)
    (T : Shrink.Template F) (s : Shrink.Counts) (prog : StateWords → Shrink.Run F)
    (g : Game (Option (Layer1.Transcript F K t L)))
    (Reads : StateWords → (ℕ → F) → g.Pos → Prop) (r : Outcome F K t L) : Prop :=
  ∃ tr, r.2 = some tr ∧
    (∀ w ∈ tr.cands, ((prog tr.inst).ccs T s).Relation w →
      ∃ p, Reads tr.inst w p ∧ Game.Pos.node g p = .pure r.1) ∧
    ¬ Claim tr.inst

omit [Field K] [Algebra F K] [Fintype K] [DecidableEq K] in
/-- A weaker claim breaks less often. -/
theorem finalBreak_mono {t L : ℕ} {Claim Claim' : StateWords → Prop} (imp : ∀ x, Claim x → Claim' x)
    (T : Shrink.Template F) (s : Shrink.Counts) (prog : StateWords → Shrink.Run F)
    (g : Game (Option (Layer1.Transcript F K t L))) (Reads : StateWords → (ℕ → F) → g.Pos → Prop)
    (r : Outcome F K t L) (break_ : FinalBreak Claim' T s prog g Reads r) : FinalBreak Claim T s prog g Reads r := by
  obtain ⟨tr, accepted, reads, notClaim⟩ := break_
  exact ⟨tr, accepted, reads, fun claim => notClaim (imp _ claim)⟩

/-- **End-to-end soundness, in sequence.** Under the encoding premise and
with idealized WHIR openings, a prover makes the final verifier accept a
statement whose terminal claim is false with probability at most the sum of
the layer errors. -/
theorem final_sound (s1 : Layer1.Shape) (Φ : F[X]) (bar : ℕ → ℕ → F) (S : Layer1.Setup F)
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
    (prover : Game.Strategy (Outcome F K s1.queries L)) :
    Game.prob prover
        (pair (Layer1.layer1Game s1 Φ bar S L0 L1 L P q) (Shrink.shrinkGame T s prog Ls))
        (FinalBreak (TerminalClaim s1 Φ bar S authority) T s prog (Layer1.layer1Game s1 Φ bar S L0 L1 L P q)
          Reads) ≤
      Layer1.layer1Error s1 L0 L1 L P (Fintype.card K) +
        Shrink.shrinkError (Shrink.Layout.rowVars T s) (Shrink.Layout.cellVars T s) Ls (Fintype.card K) := by
  have proper : Φ ≠ 1 := by
    rintro rfl
    rw [natDegree_one] at degree
    omega
  refine (prob_mono prover _ ?_).trans (pair_sound prover _ _ _ _ _ _
    (Layer1.layer1Game_avoids s1 Φ bar S L0 L1 L P q N sizes cast fits monic proper)
    (Shrink.shrinkGame_avoids T s prog Ls wf valid (Shrink.shrinkNodes cast9 7 le_rfl)
      (Shrink.shrinkNodes cast9 1 (by norm_num)))
    (Layer1.layer1Game_cost s1 Φ bar S L0 L1 L P q distinct (Layer1.nodes_of_cast N cast sizes.big_nodes)
      degree two)
    (Shrink.shrinkGame_cost T s prog Ls (Shrink.shrinkNodes cast9 7 le_rfl)
      (Shrink.shrinkNodes cast9 1 (by norm_num))))
  rintro r ⟨tr, accepted, reads, notClaim⟩ ⟨layer1, shrink⟩
  obtain ⟨w, member, holds⟩ := shrink tr accepted
  obtain ⟨p, read, leaf⟩ := reads w member holds
  exact notClaim (terminalClaim_of s1 Φ bar S encodes holds read fun r' node => by
    rw [leaf] at node
    rw [← Game.pure.inj node]
    exact layer1)

/-- The outcome under Fiat–Shamir: the shrink walk, and the walk of every
path of the layer-1 game `g` under the layer-1 log. -/
abbrev RestoreOutcome {t L : ℕ} (F K : Type) (g : Game (Option (Layer1.Transcript F K t L))) :=
  Option (Option (Shrink.Transcript StateWords F K)) × (g.Pos → Option (Option (Layer1.Transcript F K t L)))

/-- Under Fiat–Shamir: the final verifier accepts the statement
`x = tr.inst`, the claim `Claim x` is false, and every program witness among
the shrink candidates has a reading whose walk under the layer-1 log
accepts. -/
def FinalRestoreBreak {t L : ℕ} (Claim : StateWords → Prop)
    (T : Shrink.Template F) (s : Shrink.Counts) (prog : StateWords → Shrink.Run F)
    (g : Game (Option (Layer1.Transcript F K t L)))
    (Reads : StateWords → (ℕ → F) → g.Pos → Prop) (r : RestoreOutcome F K g) : Prop :=
  ∃ tr, r.1 = some (some tr) ∧
    (∀ w ∈ tr.cands, ((prog tr.inst).ccs T s).Relation w → ∃ p, Reads tr.inst w p ∧ ∃ b, r.2 p = some b) ∧
    ¬ Claim tr.inst

omit [Field K] [Algebra F K] [Fintype K] [DecidableEq K] in
/-- A weaker claim breaks less often. -/
theorem finalRestoreBreak_mono {t L : ℕ} {Claim Claim' : StateWords → Prop} (imp : ∀ x, Claim x → Claim' x)
    (T : Shrink.Template F) (s : Shrink.Counts) (prog : StateWords → Shrink.Run F)
    (g : Game (Option (Layer1.Transcript F K t L))) (Reads : StateWords → (ℕ → F) → g.Pos → Prop)
    (r : RestoreOutcome F K g) (break_ : FinalRestoreBreak Claim' T s prog g Reads r) :
    FinalRestoreBreak Claim T s prog g Reads r := by
  obtain ⟨tr, accepted, reads, notClaim⟩ := break_
  exact ⟨tr, accepted, reads, fun claim => notClaim (imp _ claim)⟩

/-- **End-to-end soundness under Fiat–Shamir.** In the state-restoration
experiment over both layers, a prover with `calls` queries in total makes
the final verifier accept a statement whose terminal claim is false with
probability at most `calls` times the larger layer error. -/
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

/-- **End-to-end budget at Rust's split, in sequence.** Rust gives each
layer the error `2^-t` for the layer bits `t` (`finish.rs:434-440`).
`Pcs::new` accepts a configuration only when each layer-1 commitment's
composed error is at most `2^-(t+1)` (`p0`, `p1`) and the shrink
commitment's is at most `2^-t` (`rust`). Then the WHIR errors plus the final
error are at most `2 / 2^t`, the error that the fold leaves. -/
theorem final_budget (s1 : Layer1.Shape) (Φ : F[X]) (bar : ℕ → ℕ → F) (S : Layer1.Setup F)
    (L0 L1 : ℕ) (h0 : 1 ≤ L0) (h1 : 1 ≤ L1) (big : 2 ^ 191 ≤ Fintype.card K)
    (index : s1.setupVars ≤ 35) (t : ℕ) (whirP0 whirP1 whirS : ℝ)
    (T : Shrink.Template F) (s : Shrink.Counts) (prog : StateWords → Shrink.Run F) (Ls : ℕ)
    (p0 : whirP0 + L0 * L1 * Layer1.rustDegrees s1 / 2 ^ 191 ≤ 1 / 2 ^ (t + 1))
    (p1 : whirP1 + L1 * (1 / 2) ^ (s1.queries - 1) ≤ 1 / 2 ^ (t + 1))
    (rust : whirS + Ls * Shrink.rustDegrees (Shrink.Layout.rowVars T s) (Shrink.Layout.cellVars T s) / 2 ^ 191 ≤
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
    (prover : Game.Strategy (Outcome F K s1.queries (2 ^ (s1.setupVars - 2)))) :
    whirP0 + whirP1 + whirS + Game.prob prover
        (pair (Layer1.layer1Game s1 Φ bar S L0 L1 (2 ^ (s1.setupVars - 2)) goldilocksModulus q)
          (Shrink.shrinkGame T s prog Ls))
        (FinalBreak (TerminalClaim s1 Φ bar S authority) T s prog
          (Layer1.layer1Game s1 Φ bar S L0 L1 (2 ^ (s1.setupVars - 2)) goldilocksModulus q) Reads) ≤
      2 / 2 ^ t := by
  have sound := final_sound s1 Φ bar S L0 L1 _ _ q N sizes cast distinct fits monic degree two T s prog Ls wf
    valid cast9 authority Reads encodes prover
  have layer1 := Layer1.layer1_budget s1 L0 L1 (Fintype.card K) t h0 h1 big index whirP0 whirP1 p0 p1
  have shrink := Shrink.shrink_budget (Shrink.Layout.rowVars T s) (Shrink.Layout.cellVars T s) Ls
    (Fintype.card K) t big whirS rust
  have split : (2 : ℝ) / 2 ^ t = 1 / 2 ^ t + 1 / 2 ^ t := by ring
  linarith

/-- **End-to-end budget at Rust's split under Fiat–Shamir.** With the
charged terms of each commitment within its share, as in
`layer1_restore_budget` and `shrink_restore_budget`, a prover with `calls`
queries breaks the final verifier with probability at most `calls / 2^t`. -/
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
  have layer1 := Layer1.layer1_budget s1 L0 L1 (Fintype.card K) t h0 h1 big index 0 0 (by simpa using p0)
    (by simpa using p1)
  have shrink := Shrink.shrink_budget _ _ Ls (Fintype.card K) t big 0 (by simpa using rust)
  rw [zero_add, zero_add] at layer1
  rw [zero_add] at shrink
  rw [div_eq_mul_one_div]
  exact mul_le_mul_of_nonneg_left (max_le shrink layer1) (Nat.cast_nonneg _)

end CompressionLean.Final
