import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
import tests.Axioms.Support

/-! Fail-closed dependency gate for the concrete Phi81 Π_RLC sampler boundary. -/

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Bound.challengeValid' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Bound.challengeValid

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Bound.excludesShortfall' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Bound.excludesShortfall

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.exists_bound_or_exists_shortfall' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.exists_bound_or_exists_shortfall

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.certificateBound_challengesValid' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.certificateBound_challengesValid

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.certificateAccepted_challengesValid' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.certificateAccepted_challengesValid
