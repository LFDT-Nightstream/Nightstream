import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc
import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.CombinedNc.Step

/-!
Protocol-owned adjacent-step surface for the delayed packed-`yZcol` deviation.

Owns: the narrow protocol export surface for the raw packed-parent projection
and adjacent-step combined-NC closure theorems.

Does not own: the underlying polynomial, acceptance proof, transcript
sampling, state continuity, terminal closure, Rust/R1CS refinement, costs, or
rows.

Emits constraints: no.

Authority boundary: the positive theorem consumes the actual combined-NC
`FixedPhase.Accepted` predicate over authoritative raw running assignments. It
does not accept a source-projection equality, child `yZcol` sidecar, digest, or
generic refinement predicate.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.delayed.combined_nc.parent` | expose the raw recomposed parent and its verifier-weighted projection | direct dataflow | `parentProjection`, `rawPackedParent` |
| `fprime.delayed.combined_nc.step` | reduce accepted successor combined-NC to predecessor packed-`yZcol` authority or a named event | derived/security partition | `accepted_next_implies_rawProjection_or_badEvent`, `accepted_next_of_rawRecomposition_implies_previous_packedYZcolBound_or_badEvent`, `accepted_next_of_parentOpening_implies_previous_packedYZcolBound_or_bindingEvent` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.CombinedNc

export Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.CombinedNc.Step
  (parentProjection rawPackedParent ProducerBetaBadRoot
    accepted_next_implies_rawProjection_or_badEvent
    accepted_next_of_rawRecomposition_implies_previous_packedYZcolBound_or_badEvent
    accepted_next_of_parentOpening_implies_previous_packedYZcolBound_or_bindingEvent)

end Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.CombinedNc
