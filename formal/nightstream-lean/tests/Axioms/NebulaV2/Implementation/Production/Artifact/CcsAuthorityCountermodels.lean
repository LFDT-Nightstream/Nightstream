import Nightstream.Implementation.NebulaV2.Production.Artifact.CcsAuthorityCountermodels
import tests.Axioms.Support

/-! Dependency audit for incomplete production CCS authority countermodels. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionCcsAuthorityCountermodels.badAffine_not_fullMatches' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionCcsAuthorityCountermodels.badAffine_not_fullMatches

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionCcsAuthorityCountermodels.wrongState_not_fullMatches' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionCcsAuthorityCountermodels.wrongState_not_fullMatches

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionCcsAuthorityCountermodels.badPadding_not_fullMatches' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionCcsAuthorityCountermodels.badPadding_not_fullMatches
