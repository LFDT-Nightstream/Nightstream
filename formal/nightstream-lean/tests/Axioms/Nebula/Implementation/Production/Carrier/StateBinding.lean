import Nightstream.Implementation.Nebula.Production.Carrier.StateBinding
import tests.Axioms.Support

/-! Dependency audit for the field-native full-claim state binding. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionFullClaimStateBinding.authoritativeFrame_lengthFor' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionFullClaimStateBinding.authoritativeFrame_lengthFor

/-- info: 'Nightstream.Implementation.Nebula.ProductionFullClaimStateBinding.bindingState_replays_authoritativeFrame' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionFullClaimStateBinding.bindingState_replays_authoritativeFrame

/-- info: 'Nightstream.Implementation.Nebula.ProductionFullClaimStateBinding.equal_bindingState_recovers_claim_or_named_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionFullClaimStateBinding.equal_bindingState_recovers_claim_or_named_failure

/-- info: 'Nightstream.Implementation.Nebula.ProductionFullClaimStateBinding.authoritativeFrames_ne_of_candidate_ne' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionFullClaimStateBinding.authoritativeFrames_ne_of_candidate_ne
