import Nightstream.Implementation.Nebula.Production.Artifact.CcsAuthorityCountermodels
import tests.Axioms.Support

/-! Dependency audit for incomplete production CCS authority countermodels. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionCcsAuthorityCountermodels.badAffine_not_fullMatches' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionCcsAuthorityCountermodels.badAffine_not_fullMatches

/-- info: 'Nightstream.Implementation.Nebula.ProductionCcsAuthorityCountermodels.wrongState_not_fullMatches' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionCcsAuthorityCountermodels.wrongState_not_fullMatches

/-- info: 'Nightstream.Implementation.Nebula.ProductionCcsAuthorityCountermodels.badPadding_not_fullMatches' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionCcsAuthorityCountermodels.badPadding_not_fullMatches
