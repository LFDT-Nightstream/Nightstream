import Nightstream.Implementation.R1CS.Canonical.KQuotient
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKQuotient

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.reductionQuotient_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KQuotient.reductionQuotient_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.coeffAt_reductionQuotient' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotient.coeffAt_reductionQuotient

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.reductionQuotient_canonical' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotient.reductionQuotient_canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.coeffAt_modulus' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotient.coeffAt_modulus

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.modulus_nonzero_positions' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotient.modulus_nonzero_positions

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.modulus_zero_elsewhere' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotient.modulus_zero_elsewhere

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.modulus_term_vanishes' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotient.modulus_term_vanishes

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.offset_zero_iff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotient.offset_zero_iff

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.survives_self' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KQuotient.survives_self

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.survivorList_succ' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotient.survivorList_succ

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.filter_map_succ' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KQuotient.filter_map_succ

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.filter_survivors' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotient.filter_survivors

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.convolution_modulus_eq_survivor_sum' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotient.convolution_modulus_eq_survivor_sum

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.mem_survivorList' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotient.mem_survivorList

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.term_at_survivor' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotient.term_at_survivor

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.convolution_modulus_eq_quotient_sum' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotient.convolution_modulus_eq_quotient_sum

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.quotientSum_low' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotient.quotientSum_low

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.quotientSum_middle' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotient.quotientSum_middle

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.quotientSum_high' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotient.quotientSum_high

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.quotientCoeff_shift27' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotient.quotientCoeff_shift27

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.quotientCoeff_shift54' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotient.quotientCoeff_shift54

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.quotient_identity_high' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotient.quotient_identity_high

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.quotientCoeff_mid_top' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotient.quotientCoeff_mid_top

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.quotientCoeff_mid_shift' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotient.quotientCoeff_mid_shift

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.quotient_identity_middle' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotient.quotient_identity_middle

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.quotient_identity_low' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotient.quotient_identity_low

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.reducedList_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KQuotient.reducedList_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.coeffAt_reducedList' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KQuotient.coeffAt_reducedList

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.coeffAt_reducedList_beyond' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KQuotient.coeffAt_reducedList_beyond

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.coefficient_identity' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotient.coefficient_identity

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.reducedList_canonical' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotient.reducedList_canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.quotientMultiple_length' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotient.quotientMultiple_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.raw_eq_reduced_add_quotient_multiple' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotient.raw_eq_reduced_add_quotient_multiple

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.polyEval_raw_eq_quotientForm' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotient.polyEval_raw_eq_quotientForm

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.toList_ringKMul' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotient.toList_ringKMul

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KQuotient.polyEval_ringKMul_quotientForm' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KQuotient.polyEval_ringKMul_quotientForm

end NightstreamTests.Axioms.CanonicalKQuotient
