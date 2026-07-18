import Nightstream.Protocol.FPrime.ConcretePhi81.Semantics
import tests.Axioms.Support

/-! Fail-closed dependency gate for the branch-complete Construction-2 bridge. -/

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.BaseSemantics.complete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.BaseSemantics.complete

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.BaseSemantics.sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.BaseSemantics.sound

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Construction2.sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Construction2.sound

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Semantics.base_recursive_disjoint' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.Semantics.base_recursive_disjoint

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Semantics.sound' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.Semantics.sound

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.SelectedNifsSemantics.transition_of_result' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.SelectedNifsSemantics.transition_of_result

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Construction2.selectedNifsRefinement' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Construction2.selectedNifsRefinement

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Construction2.sound_selectedNifs' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Construction2.sound_selectedNifs

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Semantics.sound_selectedNifs' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.Semantics.sound_selectedNifs
