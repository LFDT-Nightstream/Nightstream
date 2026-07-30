import Nightstream.Implementation.R1CS.Canonical.KFoldSum
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKFoldSum

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KFoldSum.foldl_step_accumulator' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KFoldSum.foldl_step_accumulator

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KFoldSum.foldl_step_from_zero' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KFoldSum.foldl_step_from_zero

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KFoldSum.guard_collapses' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KFoldSum.guard_collapses

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KFoldSum.term_is_polynomial' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KFoldSum.term_is_polynomial

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KFoldSum.sumOver_append' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KFoldSum.sumOver_append

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KFoldSum.sumOver_of_no_guard' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KFoldSum.sumOver_of_no_guard

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KFoldSum.sumOver_map' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KFoldSum.sumOver_map

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KFoldSum.sumOver_of_zero_terms' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KFoldSum.sumOver_of_zero_terms

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KFoldSum.convolution_eq_sumOver' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KFoldSum.convolution_eq_sumOver

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KFoldSum.ofConcrete_foldl' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KFoldSum.ofConcrete_foldl

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KFoldSum.sumOver_congr_guard' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KFoldSum.sumOver_congr_guard

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KFoldSum.sumOver_range_truncate' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KFoldSum.sumOver_range_truncate

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KFoldSum.rawMulCoeffK_eq_coeffAt_polyMul' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KFoldSum.rawMulCoeffK_eq_coeffAt_polyMul

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KFoldSum.sumOver_filter' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KFoldSum.sumOver_filter

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KFoldSum.sumOver_congr_term' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KFoldSum.sumOver_congr_term

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KFoldSum.sumOver_of_guard_zero' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KFoldSum.sumOver_of_guard_zero

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KFoldSum.sumOver_range_truncate_terms' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KFoldSum.sumOver_range_truncate_terms

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KFoldSum.rawMulCoeffK_eq_coeffAt_polyMul_high' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KFoldSum.rawMulCoeffK_eq_coeffAt_polyMul_high

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KFoldSum.rawMulCoeffK_eq_coeffAt_polyMul_all' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KFoldSum.rawMulCoeffK_eq_coeffAt_polyMul_all

end NightstreamTests.Axioms.CanonicalKFoldSum
