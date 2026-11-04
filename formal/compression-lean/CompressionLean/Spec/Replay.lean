import CompressionLean.Spec.Relation

namespace CompressionLean.Spec

open NightstreamFPrime.Spec (goldilocksModulus)
open CompressionLean.Replay

def eval {α : Type} (x : ReplayM α) (t : Transcript) : Except String α × Transcript := (x.run.run t).run

def Runs {α : Type} (x : ReplayM α) (t : Transcript) (a : α) (t' : Transcript) : Prop :=
  eval x t = (.ok a, t')

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

def drawn (queries : ℕ) : Draw → (Σ C : Type, C)
  | .ext v => ⟨K, v.toK⟩
  | .pair a b => ⟨K × K, (a.toK, b.toK)⟩
  | .words ws => ⟨Fin queries → Fin goldilocksModulus, fun j => ⟨(ws[j.val]!).word.toNat, Word.canonical _⟩⟩
  | .sample w => ⟨Fin goldilocksModulus, ⟨w.word.toNat, w.canonical⟩⟩

end CompressionLean.Spec
