import CompressionLean.Spec.Relation

/-!
Owns the model's messages of a replay proof (`layer1Msgs`): the statement,
then the early (`earlyMsgs`), linear (`linearMsgs`), honest-fold
(`honestMsgs`) and late (`lateMsgs`) messages, with the sum-check and GKR
messages that they hold (`sumcheckMsgs`, `gkrMsgs`).
-/

namespace CompressionLean.Spec

open CompressionLean.Replay CompressionLean.Layer1

variable {K : Type} [Field K] [Algebra Gl K] [Fintype K] [ChallengeField K]

/-- The model's messages of the replay's sum-check rounds: each round's
values in `K`. -/
def sumcheckMsgs (d : ℕ) (rounds : List (Array Ext)) : List (Σ M : Type, M) :=
  rounds.map fun m => ⟨Fin (d + 1) → K, fun j => (m[(j : ℕ)]!).toK⟩

/-- The model's children message: tree `t`'s part pairs from the replay. -/
def childrenOf (pairs : List (ℕ × Array Ext)) (t j : ℕ) : K × K :=
  match pairs.find? (·.1 == t) with
  | some pair => ((pair.2[2 * j]!).toK, (pair.2[2 * j + 1]!).toK)
  | none => (0, 0)

/-- The model's messages of the replay's GKR layers from layer `k`: each
layer's sum-check rounds, then its children. -/
def gkrLayersMsgs (shapes : Array TreeShape) : List GkrLayer → ℕ → List (Σ M : Type, M)
  | [], _ => []
  | layer :: rest, k =>
      sumcheckMsgs (K := K) (layerDegree (layerFactors shapes k) - 1) layer.rounds.toList ++
        ⟨ℕ → ℕ → K × K, childrenOf ((activeTrees shapes k).zip layer.children).toList⟩ ::
          gkrLayersMsgs shapes rest (k + 1)

/-- The model's messages of the replay's GKR proof: the roots, then every layer. -/
def gkrMsgs (shapes : Array TreeShape) (proof : GkrProof) : List (Σ M : Type, M) :=
  ⟨ℕ → K × K, fun t => pairK (proof.roots[t]!)⟩ :: gkrLayersMsgs (K := K) shapes proof.layers.toList 1

/-- The model's early values of the replay's. -/
def valuesOf (v : Replay.EarlyValues) : Layer1.EarlyValues K :=
  ⟨v.eRe.toK, v.eIm.toK, v.row.toK, fun j => pairK (v.pairs[j]!), v.block.toK⟩

/-- The model's histogram of the proof's counts. -/
def histOf (proof : Replay.Proof) (i : ℕ) : Gl := (proof.histogram[i]!).toF

/-- The model's messages of the replay's early stage: the P0 candidates, the
histogram, the GKR messages and the early values. -/
def earlyMsgs (rel : Replay.Relation) (proof : Replay.Proof) {L0 : ℕ}
    (cands : {l : List (P0Tables Gl) // l.length ≤ L0}) : List (Σ M : Type, M) :=
  ⟨{l : List (P0Tables Gl) // l.length ≤ L0}, cands⟩ :: ⟨ℕ → Gl, histOf proof⟩ ::
    (gkrMsgs (K := K) rel.earlyShapes proof.early ++ [⟨Layer1.EarlyValues K, valuesOf proof.earlyValues⟩])

/-- The model's messages of the replay's linear stage: the quotient, the
block and lane sum-check rounds, and the finals. -/
def linearMsgs (rel : Replay.Relation) (proof : Replay.Proof) : List (Σ M : Type, M) :=
  ⟨ℕ → K, tableK proof.quotient⟩ ::
    (sumcheckMsgs (K := K) 1 (proof.linear.extract 0 rel.blockVars).toList ++
      (sumcheckMsgs (K := K) 1 (proof.linear.extract rel.blockVars proof.linear.size).toList ++
        [⟨K × K × K, (proof.finals[0]!.toK, proof.finals[1]!.toK, proof.finals[2]!.toK)⟩]))

/-- The model's partials of the proof's. -/
def partialsOf (proof : Replay.Proof) (g : Fin 2) : ℕ → K := tableK (proof.partials[(g : ℕ)]!)

/-- The model's messages of the replay's honest fold: the partials and the
P1 candidates. -/
def honestMsgs (proof : Replay.Proof) {L1 : ℕ} (p1 : {l : List (P1Tables K) // l.length ≤ L1}) :
    List (Σ M : Type, M) :=
  [⟨Fin 2 → ℕ → K, partialsOf proof⟩, ⟨{l : List (P1Tables K) // l.length ≤ L1}, p1⟩]

/-- The model's messages of the replay's late stage: the GKR messages and
the late values. -/
def lateMsgs (rel : Replay.Relation) (proof : Replay.Proof) : List (Σ M : Type, M) :=
  gkrMsgs (K := K) rel.lateShapes proof.late ++ [⟨ℕ → K × K, fun j => pairK (proof.lateValues[j]!)⟩]

/-- The model's messages of the replay's layer-1 proof: the statement, then
the early, linear, honest-fold and late messages. -/
def layer1Msgs (rel : Replay.Relation) (st : Replay.Statement) (proof : Replay.Proof) {L0 L1 : ℕ}
    (cands : {l : List (P0Tables Gl) // l.length ≤ L0}) (p1 : {l : List (P1Tables K) // l.length ≤ L1}) :
    List (Σ M : Type, M) :=
  ⟨Layer1.Statement Gl, statementOf st⟩ :: (earlyMsgs (K := K) rel proof cands ++
    (linearMsgs (K := K) rel proof ++ (honestMsgs proof p1 ++ lateMsgs (K := K) rel proof)))

end CompressionLean.Spec
