import CompressionLean.Spec.Relation
import CompressionLean.Lib.Univariate

/-!
Owns the two assumptions that a reading of the claims takes on a replay
proof (`TRUST.md` §3): WHIR opens the candidates at the points that the
verifier gives it (`WhirOpens`, with the opened values `p0Opened` and
`p1Opened`), and the proof's setup leaves are bound by the key's Merkle root
(`MerkleBinds`).
-/

namespace CompressionLean.Spec

open CompressionLean.Lib
open Polynomial
open CompressionLean.Replay CompressionLean.Layer1

/-- **Merkle binding**, an assumption on the proof's leaves: a leaf whose path
reaches the key root at index `i` holds, for each setup oracle, its codeword
`Ô_o` on the coset `y_i·ω_8^v` of the honest leaf `i`. A leaf that breaks it
and still reaches the root gives a hash collision. -/
def MerkleBinds (rel : Replay.Relation) (S : Setup Gl) (leaves : Array Leaf) : Prop :=
  ∀ leaf ∈ leaves.toList, ∀ i, pathOk rel.key.root i leaf = true → ∀ o v, o < rel.kappa + runOracles →
    v < 2 ^ foldVars → (leaf.values[o * 2 ^ foldVars + v]!).toF =
      (univ rel.setupVars (setupOracle rel S o)).eval (leafY rel.setupVars i * twoAdic 3 ^ v)

variable {K : Type} [Field K] [Algebra Gl K] [Fintype K] [ChallengeField K]

local notation "ι" => algebraMap Gl K

/-- P0 candidate `w`'s values at the points, one batch per point: `z` at the
norm leaf and at the linear point, `e`, the early pairs, the early
slot-block copy, the late pairs and copy, the row counts. -/
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

/-- P1 candidate `v`'s values at the points: the key fold and the run fold at
their tails, both folds at each query point, `Ω^A` at the two block points. -/
noncomputable def p1Opened (pts : OpeningPoints) (v : P1Tables K) : List K :=
  [mle (pointK pts.keyTail) v.foldKey, mle (pointK pts.runTail) v.foldRun] ++
    pts.queries.toList.flatMap (fun pt => [mle (pointK pt) v.foldKey, mle (pointK pt) v.foldRun]) ++
    [mle (pointK pts.block) v.omegaA, mle (pointK pts.lateBlock) v.omegaA]

/-- **WHIR opens**, an assumption: one listed P0 candidate and one listed P1
candidate have the opened values at the points that the verifier gives WHIR.
A P1 value is read from its three coordinate openings (`from_coordinates`). -/
def WhirOpens (s : Shape) (proof : Replay.Proof) (pts : OpeningPoints) (cands : List (P0Tables Gl))
    (p1 : List (P1Tables K)) : Prop :=
  ∃ w ∈ cands, ∃ v ∈ p1,
    proof.p0Opening.evals.toList.map (fun batch => batch.toList.map (Ext.toK (K := K))) = p0Opened s pts w ∧
      (Replay.p1Values proof.p1Opening.evals).toList.map (Ext.toK (K := K)) = p1Opened pts v

end CompressionLean.Spec
