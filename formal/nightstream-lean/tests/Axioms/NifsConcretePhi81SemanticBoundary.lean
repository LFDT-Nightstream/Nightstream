import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Evaluator.SemanticBoundary
import tests.Axioms.Support

/-! Fail-closed dependency gate for conditional fixed-active soundness. -/

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Evaluator.run_sound_of_closure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Evaluator.run_sound_of_closure

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Evaluator.run_sound_of_outputBound_noBadEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Evaluator.run_sound_of_outputBound_noBadEvent
