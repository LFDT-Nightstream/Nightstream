import CompressionLean.Game.Choice
import CompressionLean.Spec.Bound

/-! What the bound of option B's final program states (CW-6.5). The final program runs one layer-1
verifier per slot, each on its own relation; the restoration game pairs the shrink game with the
choice of the two slots' layer-1 games (`Game.choice`). The prover breaks the program when the
shrink layer accepts a statement, both layer-1 positions that an accepted witness reads walk, and
the claim fails (`RestoreBreak`). The claim is a terminal environment whose two slot statements are
in their layer-1 relations (`Holds`). -/

namespace CompressionLean.Spec.OptionB

open CompressionLean.Game
open CompressionLean NightstreamFPrime Polynomial
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F K E : Type} [Field F] [Field K] [Algebra F K] [Fintype K] [DecidableEq K]
  {t0 L0 t1 L1 : ℕ}

/-- The result of the restoration game over the shrink game and the choice of the slot games. -/
abbrev Outcome (F K : Type) (g0 : Game (Option (Layer1.Transcript F K t0 L0)))
    (g1 : Game (Option (Layer1.Transcript F K t1 L1))) :=
  Option (Option (Shrink.Transcript StateWords F K)) ×
    (Pos (choice g0 g1) →
      Option (Option (Layer1.Transcript F K t0 L0) ⊕ Option (Layer1.Transcript F K t1 L1)))

/-- The shrink layer accepts a statement whose claim fails, and for each accepted witness both
positions that it reads walk. -/
def RestoreBreak (Claim : StateWords → Prop) (T : Shrink.Template F) (s : Shrink.Counts)
    (prog : StateWords → Shrink.Run F) (g0 : Game (Option (Layer1.Transcript F K t0 L0)))
    (g1 : Game (Option (Layer1.Transcript F K t1 L1)))
    (Reads0 : StateWords → (ℕ → F) → g0.Pos → Prop) (Reads1 : StateWords → (ℕ → F) → g1.Pos → Prop)
    (r : Outcome F K g0 g1) : Prop :=
  ∃ tr, r.1 = some (some tr) ∧
    (∀ w ∈ tr.cands, ((prog tr.inst).ccs T s).Relation w →
      (∃ p, Reads0 tr.inst w p ∧ ∃ b, r.2 (Pos.left g1 p) = some b) ∧
      ∃ p, Reads1 tr.inst w p ∧ ∃ b, r.2 (Pos.right g0 p) = some b) ∧
    ¬ Claim tr.inst

/-- The terminal holds in an environment whose slot statements are in their layer-1 relations. -/
def Holds (s0 s1 : Layer1.Shape) (Φ : F[X]) (bar : ℕ → ℕ → F) (S0 S1 : Layer1.Setup F)
    (Terminal : StateWords → E → Prop) (claim0 claim1 : E → Layer1.Statement F) (x : StateWords) :
    Prop :=
  ∃ env, Terminal x env ∧ (∃ z, Layer1.Relation s0 Φ bar S0 (claim0 env) z) ∧
    ∃ z, Layer1.Relation s1 Φ bar S1 (claim1 env) z

end CompressionLean.Spec.OptionB
