import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.Embedding

/-!
Focused surface regression for the executable Phi81 coefficient embedding.

| Stage path | Regression |
|---|---|
| `nifs.pi_rlc.verify.evaluation_hom.embedding.raw_convolution` | every raw coefficient is mapped through `K.embed` |
| `nifs.pi_rlc.verify.evaluation_hom.embedding.phi81_reduction` | the complete reduced product commutes with coefficientwise embedding |
-/

namespace tests.Phi81Embedding

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism

#check Embedding.rawMulCoeffK_embedChallenge
#check Embedding.embedChallenge_ringFMul

/-- The public theorem has the exact executable operations on both sides; no
abstract ring multiplication can satisfy this regression by substitution. -/
example (left right : RingF) :
    RingKAction.embedChallenge (ringFMul left right) =
      ringKMul
        (RingKAction.embedChallenge left)
        (RingKAction.embedChallenge right) := by
  exact Embedding.embedChallenge_ringFMul left right

end tests.Phi81Embedding
