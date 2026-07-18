import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.SemanticMinimality.Baseline

/-!
Removal witnesses for selected-NIFS challenge and computed tail outputs.

Assurance tier: model-level.

Owns: inclusion-necessity witnesses for production strong-set membership, the
canonical `Pi_RLC` parent, and the canonical `Pi_DEC` children.

Does not own: transcript replay, sampler security, child-opening extraction,
physical rows, costs, Rust/R1CS refinement, or row removal.

Emits constraints: no.

Authority boundary: each witness mutates the accepted semantic candidate and
recomputes dependent outputs where required. Strong-set membership is a
semantic obligation; this file does not assert that separate membership rows
are necessary when a verifier-driven sampler replay already derives it.

| Phase | Stage path | Counterexample | Lean owner |
|---|---|---|---|
| `Pi_RLC` | `fprime.active.nifs.pi_rlc.challenge.strong_set.necessity` | replace every challenge by constant three and recompute outputs | `challengeStrongSet_necessary` |
| `Pi_RLC` | `fprime.active.nifs.pi_rlc.parent.exact.necessity` | change only the parent stage | `parentExact_necessary` |
| `Pi_DEC` | `fprime.active.nifs.pi_dec.children.exact.necessity` | change only child zero's stage | `childrenExact_necessary` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.SemanticMinimality

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81

/-- Closed inclusion-necessity of unary production-set membership for the
exact semantic plan. This is not a claim that separate physical membership
rows are necessary. -/
theorem challengeStrongSet_necessary :
    CheckPlan.NecessaryForSoundness
      baselineSemantics baselineTarget SemanticFold.ObligationPlan.checks
      .challengeStrongSet := by
  rcases baseline with ⟨realization⟩
  exact realization.challengeNecessary

/-- Closed inclusion-necessity of binding the public parent to the canonical
`Pi_RLC` computation. -/
theorem parentExact_necessary :
    CheckPlan.NecessaryForSoundness
      baselineSemantics baselineTarget SemanticFold.ObligationPlan.checks
      .parentExact := by
  rcases baseline with ⟨realization⟩
  exact realization.parentNecessary

/-- Closed inclusion-necessity of binding the public child family to the
canonical `Pi_DEC` split. -/
theorem childrenExact_necessary :
    CheckPlan.NecessaryForSoundness
      baselineSemantics baselineTarget SemanticFold.ObligationPlan.checks
      .childrenExact := by
  rcases baseline with ⟨realization⟩
  exact realization.childrenNecessary

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.SemanticMinimality
