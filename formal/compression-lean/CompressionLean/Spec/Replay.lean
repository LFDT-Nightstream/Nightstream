import CompressionLean.Spec.Relation

/-!
Owns what the claims read from a run of the replay (`Ports/Replay/`): the
result and the final transcript of an action (`eval`), a successful run
(`Runs`), `verify` up to WHIR (`verifyStages`), and a replay draw as the
model's challenge (`drawn`).
-/

namespace CompressionLean.Spec

open NightstreamFPrime.Spec (goldilocksModulus)
open CompressionLean.Replay

/-- The result and the final transcript of `x` run from `t`. -/
def eval {α : Type} (x : ReplayM α) (t : Transcript) : Except String α × Transcript := (x.run.run t).run

/-- `x` run from `t` returns `a` and ends in `t'`. -/
def Runs {α : Type} (x : ReplayM α) (t : Transcript) (a : α) (t' : Transcript) : Prop :=
  eval x t = (.ok a, t')

/-- The replay up to the late stage: `verify` before WHIR's transcripts and the
opening checks. Its draws are the draws of the layer-1 game. -/
def verifyStages (rel : Replay.Relation) (seed : Array Word) (st : Replay.Statement) (proof : Replay.Proof) :
    ReplayM Unit := do
  ensure (st.fits rel) "claim words"
  liftExcept (readProof rel proof)
  observeAll seed
  observeAll rel.words
  observeAll st.words
  let early ← earlyStage rel st proof
  let lin ← linearStage rel st proof
  let _ ← honestStage rel proof early lin
  let _ ← lateStage rel proof lin
  pure ()

variable {K : Type} [Field K] [Algebra Gl K] [Fintype K] [ChallengeField K]

/-- A replay draw as the model's challenge: an `Ext` value in `K`, a pair in
`K × K`, the setup query words as `Fin queries → Fin goldilocksModulus`, a WHIR
query word in `Fin goldilocksModulus`. -/
def drawn (queries : ℕ) : Draw → (Σ C : Type, C)
  | .ext v => ⟨K, v.toK⟩
  | .pair a b => ⟨K × K, (a.toK, b.toK)⟩
  | .words ws => ⟨Fin queries → Fin goldilocksModulus, fun j => ⟨(ws[j.val]!).word.toNat, Word.canonical _⟩⟩
  | .sample w => ⟨Fin goldilocksModulus, ⟨w.word.toNat, w.canonical⟩⟩

end CompressionLean.Spec
