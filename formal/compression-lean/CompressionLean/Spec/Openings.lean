import CompressionLean.Spec.Relation
import CompressionLean.Lib.Univariate

namespace CompressionLean.Spec

open CompressionLean.Lib
open Polynomial
open CompressionLean.Replay CompressionLean.Layer1

/-- Assumption: Merkle binding (hash collision resistance), not proved here. The index bound matters: the path reads only the low `n - 2` bits. -/
def MerkleBinds (rel : Replay.Relation) (S : Setup Gl) (leaves : Array Leaf) : Prop :=
  ∀ leaf ∈ leaves.toList, ∀ i < 2 ^ (rel.setupVars - 2), pathOk rel.key.root i leaf = true → ∀ o v,
    o < rel.kappa + runOracles →
    v < 2 ^ foldVars → (leaf.values[o * 2 ^ foldVars + v]!).toF =
      (univ rel.setupVars (setupOracle rel S o)).eval (leafY rel.setupVars i * twoAdic 3 ^ v)

variable {K : Type} [Field K] [Algebra Gl K] [Fintype K] [ChallengeField K]

local notation "ι" => algebraMap Gl K

noncomputable def p0Opened (s : Shape) (pts : OpeningPoints) (w : P0Tables Gl) : List (List K) :=
  [[mle (pointK pts.norm) fun x => ι (w.z x)],
    [mle (pointK pts.linear) fun x => ι (w.z x)],
    [mle (pointK pts.run) fun k => ι (w.e k).1, mle (pointK pts.run) fun k => ι (w.e k).2],
    (List.range s.matrices).flatMap (fun j => [mle (pointK pts.pairs) fun σ => ι (w.U j σ).1,
      mle (pointK pts.pairs) fun σ => ι (w.U j σ).2]),
    [mle (pointK pts.slots) fun σ => ι (w.blockCopy σ)],
    (List.range s.matrices).flatMap (fun j => [mle (pointK pts.latePairs) fun σ => ι (w.U j σ).1,
      mle (pointK pts.latePairs) fun σ => ι (w.U j σ).2]) ++
      [mle (pointK pts.latePairs) fun σ => ι (w.blockCopy σ)],
    [mle (pointK pts.rows) fun i => ι (w.mult i)]]

noncomputable def p1Opened (pts : OpeningPoints) (v : P1Tables K) : List K :=
  [mle (pointK pts.keyTail) v.foldKey, mle (pointK pts.runTail) v.foldRun] ++
    pts.queries.toList.flatMap (fun pt => [mle (pointK pt) v.foldKey, mle (pointK pt) v.foldRun]) ++
    [mle (pointK pts.block) v.omegaA, mle (pointK pts.lateBlock) v.omegaA]

/-- Assumption: WHIR opening soundness, not proved here. -/
def WhirOpens (s : Shape) (proof : Replay.Proof) (pts : OpeningPoints) (cands : List (P0Tables Gl))
    (p1 : List (P1Tables K)) : Prop :=
  ∃ w ∈ cands, ∃ v ∈ p1,
    proof.p0Opening.evals.toList.map (fun batch => batch.toList.map (Ext.toK (K := K))) = p0Opened s pts w ∧
      (Replay.p1Values proof.p1Opening.evals).toList.map (Ext.toK (K := K)) = p1Opened pts v

end CompressionLean.Spec
