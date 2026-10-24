import CompressionLean.Spec.Program
import CompressionLean.Spec.Messages
import CompressionLean.Spec.Openings
import CompressionLean.Spec.Replay
import CompressionLean.Layer1.Soundness
import CompressionLean.Shrink.Ccs
import CompressionLean.Ports.Shrink.Game
import CompressionLean.Game.Path

/-!
Owns the event that the production bound bounds.

- `game`: the layer-1 game of a relation and a setup, with F′'s modulus.
- `Reads`: a path of that game reads, from a witness of the final program,
  the statement and the proof that the in-circuit verifier reads, the
  challenges that it computes from them, and candidate lists that WHIR's
  openings and the key's Merkle root bind.
- `RestoreOutcome`, `FinalRestoreBreak`: under Fiat–Shamir, the final
  verifier accepts a statement whose claim is false, while every program
  witness among the shrink candidates has a reading whose walk under the
  layer-1 log accepts.
-/

namespace CompressionLean.Spec

open CompressionLean.Circuit CompressionLean.Terminal.Package
open CompressionLean.Layer1
open NightstreamFPrime.Spec (goldilocksModulus)
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFPolynomial (modulus)

/-- The layer-1 game of the relation `rel` and the setup `S`, with F′'s
modulus, as the final theorems state it. -/
noncomputable abbrev game (rel : Replay.Relation) (S : Setup Gl) (L0 L1 : ℕ) :=
  layer1Game (shapeOf rel) modulus barOf S L0 L1 (2 ^ (rel.setupVars - 2)) goldilocksModulus (queryAt (K := Cubic) rel)

/-- **The reading.** The path `p` reads, from the witness `wit` of program
`x`, the statement and the proof that the in-circuit verifier reads, the
challenges that it computes from them (`verify` replayed up to WHIR), and
candidate lists that agree with the WHIR openings at its points and with a
setup that the key's Merkle root binds. -/
def Reads (key : Layer1Input) (loaded : Loaded) (context : Array Replay.Word) (S : Setup Gl) (L0 L1 : ℕ)
    (x : StateWords) (wit : ℕ → Gl) (p : (game key.relation S L0 L1).Pos) : Prop :=
  let w := wordsOf key loaded context x wit
  let st := statementWith loaded context w
  let seed := seedWith loaded context w
  let proof := proofWith key loaded context w
  ∃ (cands : List (P0Tables Gl)) (short0 : cands.length ≤ L0) (p1 : List (P1Tables Cubic))
    (short1 : p1.length ≤ L1),
    Game.Pos.msgs _ p = layer1Msgs key.relation st proof ⟨cands, short0⟩ ⟨p1, short1⟩ ∧
    Game.Pos.draws _ p =
      ((eval (verifyStages key.relation seed st proof) start).2.draws.toList).map
        (drawn (K := Cubic) key.relation.queries) ∧
    (∀ points t', Runs (Replay.verify key.relation seed st proof) start points t' →
      WhirOpens (shapeOf key.relation) proof points cands p1) ∧
    MerkleBinds key.relation S proof.leaves

variable {F K : Type} [Field F] [Field K] [Algebra F K] [Fintype K] [DecidableEq K]

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

end CompressionLean.Spec
