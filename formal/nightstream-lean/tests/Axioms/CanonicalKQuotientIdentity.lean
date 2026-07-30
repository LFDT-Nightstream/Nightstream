import Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity
import Nightstream.Implementation.R1CS.Canonical.KQuotientIdentityHonest
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKQuotientIdentity

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.mulRows_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.mulRows_sound

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.productRows_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.productRows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.productRows_length_production' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.productRows_length_production

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.productRows_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.productRows_sound

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.projectionWidth_derived' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.projectionWidth_derived

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.quotientWidth_derived' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.quotientWidth_derived

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.modulusWidth_derived' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.modulusWidth_derived

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.pairsRows_length' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.pairsRows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.identityRows_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.identityRows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.identityRows_length_production' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.identityRows_length_production

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.carriedValue_concat' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.carriedValue_concat

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.pairsRows_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.pairsRows_sound

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.identityRows_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.identityRows_sound

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.atomFrames_eq' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KQuotientIdentity.atomFrames_eq

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.atomWidth_eq' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KQuotientIdentity.atomWidth_eq

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.identityFrames_eq' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.identityFrames_eq

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.identityColumns_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.identityColumns_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.identityColumns_nodup' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.identityColumns_nodup

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.frameColumns_disjoint' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.frameColumns_disjoint

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.frameColumns_subset' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.frameColumns_subset

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.atoms_disjoint' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.atoms_disjoint

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.atom_blocks_separated' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.atom_blocks_separated

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.atom_inside' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.atom_inside

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.tail_blocks_separated' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.tail_blocks_separated

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.tail_inside' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.tail_inside

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.allocated_iff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.allocated_iff

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.frameOfRun_interval' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.frameOfRun_interval

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.hornerBlock_conservation' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.hornerBlock_conservation

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.hornerCarried_conservation' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.hornerCarried_conservation

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.productRows_conservation' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.productRows_conservation

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.pairsRows_conservation' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.pairsRows_conservation

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.productCarried_mentions' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.productCarried_mentions

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.pairsCarried_mentions' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.pairsCarried_mentions

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.layout_bases' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KQuotientIdentity.layout_bases

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.identityRows_conservation' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.identityRows_conservation

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.productRows_honest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.productRows_honest

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.atomWitness_off_block' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.atomWitness_off_block

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.pairsWitness_off_block' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.pairsWitness_off_block

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.pairsRows_honest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.pairsRows_honest

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.identityCost_rows' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.identityCost_rows

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.identityCost_columns' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.identityCost_columns

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.identityCost_gap' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.identityCost_gap

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.identityWitness_off_block' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.identityWitness_off_block

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.projected_preserved' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.projected_preserved

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity.identityRows_honest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotientIdentity.identityRows_honest

end NightstreamTests.Axioms.CanonicalKQuotientIdentity
