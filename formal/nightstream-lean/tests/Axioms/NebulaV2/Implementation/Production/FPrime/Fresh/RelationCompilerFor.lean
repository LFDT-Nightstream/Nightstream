import Nightstream.Implementation.NebulaV2.Production.FPrime.Fresh.ClaimProducerFor
import tests.Axioms.Support

/-! The source-to-CCS compiler uses only Lean's standard logical axioms. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionFreshRelationCompilerFor.SourceProgram.encoded_ccsSatisfied_iff_sourceRows' depends on axioms: [propext,
 Classical.choice.{u},
 Quot.sound.{u}] -/
#guard_msgs in
set_option pp.universes true in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionFreshRelationCompilerFor.SourceProgram.encoded_ccsSatisfied_iff_sourceRows

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionFreshRelationCompilerFor.SourceProgram.ccsSatisfied_iff_decodedSourceRows' depends on axioms: [propext,
 Classical.choice.{u},
 Quot.sound.{u}] -/
#guard_msgs in
set_option pp.universes true in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionFreshRelationCompilerFor.SourceProgram.ccsSatisfied_iff_decodedSourceRows

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionFreshClaimProducerFor.FreshRelationWitness.relation' depends on axioms: [propext,
 Classical.choice.{u},
 Quot.sound.{u}] -/
#guard_msgs in
set_option pp.universes true in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionFreshClaimProducerFor.FreshRelationWitness.relation
