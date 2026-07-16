import Nightstream.SuperNeo.Concrete.Phi81StrongSet

/-!
Challenge-validity predicate for the typed Phi81 `PiRLC.Algebra`.

Protocol: SuperNeo Definition 17 and `Pi_RLC`.
Phase: verifier challenge membership before witness combination.
Constraint family: semantic predicate only; this file emits no rows.

Owns: the exact unary algebra-field predicate and its connection to the
independently defined 54-coordinate five-symbol production set.

Does not own: Poseidon2 transcript derivation, rejection sampling, decoded
Rust/R1CS challenge membership, the external low-norm invertibility theorem,
norm growth of folded witnesses, row removal, or constraint counts.

Emits constraints: no.

Authority boundary: valid challenges are exactly images of complete sampled
coefficient vectors. Coefficient range checks, list length, or a digest alone
do not establish this predicate. Pairwise security remains conditional on the
explicit `LowNormInvertibility` mathematical boundary.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_rlc.verify.challenge.membership` | one challenge is the exact embedded 54-coordinate production scalar | checked predicate | `challengeValid` |
| `nifs.pi_rlc.verify.challenge.honest` | every semantic sampler scalar embeds to a valid challenge | derived | `embedScalar_valid` |
| `nifs.pi_rlc.verify.challenge.pairwise_security` | distinct valid challenges have invertible difference | security boundary | `pairwiseSecure_of_lowNormInvertibility` |
| `nifs.pi_rlc.verify.challenge.transcript_refinement` | decoded Poseidon2/rejection-sampler output satisfies membership | missing implementation bridge | not owned here |
-/

namespace Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Challenge

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81StrongSet
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet

/-- Exact predicate supplied to `PiRLC.Algebra.challengeValid`. -/
def challengeValid (value : RingF) : Prop := ProductionMember value

theorem challengeValid_iff (value : RingF) :
    challengeValid value ↔ ProductionMember value := by
  rfl

/-- Every complete semantic production scalar yields a valid challenge. -/
theorem embedScalar_valid (scalar : Scalar) :
    challengeValid (embedScalar scalar) :=
  embedScalar_member scalar

/-- Pairwise security of the chosen unary predicate, conditional only on the
isolated analytic theorem boundary. -/
theorem pairwiseSecure_of_lowNormInvertibility
    (theorem8 : LowNormInvertibility) :
    forall {left right : RingF},
      challengeValid left -> challengeValid right -> left ≠ right ->
        RingFInvertible (ringFSub left right) := by
  exact productionSet_strong theorem8

end Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Challenge
