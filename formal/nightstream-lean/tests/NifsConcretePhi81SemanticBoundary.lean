import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Evaluator.SemanticBoundary

/-!
Focused compile-time regression for the fixed-active NIFS semantic boundary.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.fixed_active.soundness.closure` | semantic closure remains an explicit three-premise package | hiding source/output/security premises in a Boolean checker |
| `nifs.fixed_active.soundness.closed` | execution refines semantics only after the named premises are supplied | unconditional exactness claim despite open bad-event/output authority |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Evaluator

#check SoundnessClosure
#check run_sound_of_closure
#check run_sound_of_outputBound_noBadEvent

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Evaluator
