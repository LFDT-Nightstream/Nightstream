import Nightstream.Implementation.Nebula.Production.FPrime.Fresh.ClaimProducerFor
import tests.Axioms.Support

/-! Fail-closed dependency audit for the common-source fresh-claim boundary. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionFreshClaimProducerFor.freshStatement_holds' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionFreshClaimProducerFor.freshStatement_holds

/-- info: 'Nightstream.Implementation.Nebula.ProductionFreshClaimProducerFor.EncodedFreshRelationWitnessForRows.toFreshRelationWitness' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionFreshClaimProducerFor.EncodedFreshRelationWitnessForRows.toFreshRelationWitness

/-- info: 'Nightstream.Implementation.Nebula.ProductionFreshClaimProducerFor.FreshRelationWitnessForRows.authorityRows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionFreshClaimProducerFor.FreshRelationWitnessForRows.authorityRows

/-- info: 'Nightstream.Implementation.Nebula.ProductionFreshClaimProducerFor.FreshRelationWitnessForRows.selectedBranch' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionFreshClaimProducerFor.FreshRelationWitnessForRows.selectedBranch

/-- info: 'Nightstream.Implementation.Nebula.ProductionFreshClaimProducerFor.FreshRelationWitnessForRows.exists_of_ccsHolds' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionFreshClaimProducerFor.FreshRelationWitnessForRows.exists_of_ccsHolds

/-- info: 'Nightstream.Implementation.Nebula.ProductionFreshClaimProducerFor.freshStatement_holds_from_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionFreshClaimProducerFor.freshStatement_holds_from_rows

/-- info: 'Nightstream.Implementation.Nebula.ProductionFreshClaimProducerFor.freshStatement_holds_iff_exists_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionFreshClaimProducerFor.freshStatement_holds_iff_exists_rows

/-- info: 'Nightstream.Implementation.Nebula.ProductionFreshClaimProducerFor.RelationAuthority.selectedBranchOfCcsPublic' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionFreshClaimProducerFor.RelationAuthority.selectedBranchOfCcsPublic
