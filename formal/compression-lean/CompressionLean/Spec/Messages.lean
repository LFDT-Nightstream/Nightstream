import CompressionLean.Spec.Relation

namespace CompressionLean.Spec

open CompressionLean.Replay CompressionLean.Layer1

variable {K : Type} [Field K] [Algebra Gl K] [Fintype K] [ChallengeField K]

def sumcheckMsgs (d : ℕ) (rounds : List (Array Ext)) : List (Σ M : Type, M) :=
  rounds.map fun m => ⟨Fin (d + 1) → K, fun j => (m[(j : ℕ)]!).toK⟩

def childrenOf (pairs : List (ℕ × Array Ext)) (t j : ℕ) : K × K :=
  match pairs.find? (·.1 == t) with
  | some pair => ((pair.2[2 * j]!).toK, (pair.2[2 * j + 1]!).toK)
  | none => (0, 0)

def gkrLayersMsgs (shapes : Array TreeShape) : List GkrLayer → ℕ → List (Σ M : Type, M)
  | [], _ => []
  | layer :: rest, k =>
      sumcheckMsgs (K := K) (layerDegree (layerFactors shapes k) - 1) layer.rounds.toList ++
        ⟨ℕ → ℕ → K × K, childrenOf ((activeTrees shapes k).zip layer.children).toList⟩ ::
          gkrLayersMsgs shapes rest (k + 1)

def gkrMsgs (shapes : Array TreeShape) (proof : GkrProof) : List (Σ M : Type, M) :=
  ⟨ℕ → K × K, fun t => pairK (proof.roots[t]!)⟩ :: gkrLayersMsgs (K := K) shapes proof.layers.toList 1

def valuesOf (v : Replay.EarlyValues) : Layer1.EarlyValues K :=
  ⟨v.eRe.toK, v.eIm.toK, v.row.toK, fun j => pairK (v.pairs[j]!), v.block.toK⟩

def histOf (proof : Replay.Proof) (i : ℕ) : Gl := (proof.histogram[i]!).toF

def earlyMsgs (rel : Replay.Relation) (proof : Replay.Proof) {L0 : ℕ}
    (cands : {l : List (P0Tables Gl) // l.length ≤ L0}) : List (Σ M : Type, M) :=
  ⟨{l : List (P0Tables Gl) // l.length ≤ L0}, cands⟩ :: ⟨ℕ → Gl, histOf proof⟩ ::
    (gkrMsgs (K := K) rel.earlyShapes proof.early ++ [⟨Layer1.EarlyValues K, valuesOf proof.earlyValues⟩])

def linearMsgs (rel : Replay.Relation) (proof : Replay.Proof) : List (Σ M : Type, M) :=
  ⟨ℕ → K, tableK proof.quotient⟩ ::
    (sumcheckMsgs (K := K) 1 (proof.linear.extract 0 rel.blockVars).toList ++
      (sumcheckMsgs (K := K) 1 (proof.linear.extract rel.blockVars proof.linear.size).toList ++
        [⟨K × K × K, (proof.finals[0]!.toK, proof.finals[1]!.toK, proof.finals[2]!.toK)⟩]))

def partialsOf (proof : Replay.Proof) (g : Fin 2) : ℕ → K := tableK (proof.partials[(g : ℕ)]!)

def honestMsgs (proof : Replay.Proof) {L1 : ℕ} (p1 : {l : List (P1Tables K) // l.length ≤ L1}) :
    List (Σ M : Type, M) :=
  [⟨Fin 2 → ℕ → K, partialsOf proof⟩, ⟨{l : List (P1Tables K) // l.length ≤ L1}, p1⟩]

def lateMsgs (rel : Replay.Relation) (proof : Replay.Proof) : List (Σ M : Type, M) :=
  gkrMsgs (K := K) rel.lateShapes proof.late ++ [⟨ℕ → K × K, fun j => pairK (proof.lateValues[j]!)⟩]

def layer1Msgs (rel : Replay.Relation) (st : Replay.Statement) (proof : Replay.Proof) {L0 L1 : ℕ}
    (cands : {l : List (P0Tables Gl) // l.length ≤ L0}) (p1 : {l : List (P1Tables K) // l.length ≤ L1}) :
    List (Σ M : Type, M) :=
  ⟨Layer1.Statement Gl, statementOf st⟩ :: (earlyMsgs (K := K) rel proof cands ++
    (linearMsgs (K := K) rel proof ++ (honestMsgs proof p1 ++ lateMsgs (K := K) rel proof)))

end CompressionLean.Spec
