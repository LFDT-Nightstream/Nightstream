import Nightstream.Implementation.R1CS.Canonical.KLowNorm
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKLowNorm

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNorm.lowNormRows_length' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KLowNorm.lowNormRows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNorm.lowNormColumns_length' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KLowNorm.lowNormColumns_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNorm.lowNormColumns_nodup' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KLowNorm.lowNormColumns_nodup

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNorm.lowNormRows_sound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KLowNorm.lowNormRows_sound

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNorm.lowNormWitness_off_column' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KLowNorm.lowNormWitness_off_column

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNorm.lcEval_lowNormWitness' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KLowNorm.lcEval_lowNormWitness

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNorm.lowNormRows_honest' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KLowNorm.lowNormRows_honest

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNorm.cubic_expansion' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KLowNorm.cubic_expansion

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNorm.cubic_expansion_multiple' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KLowNorm.cubic_expansion_multiple

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNorm.cubicResidual_eq_zero_of_cube' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KLowNorm.cubicResidual_eq_zero_of_cube

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNorm.lowNormRows_conservation' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KLowNorm.lowNormRows_conservation

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNorm.lowNormCost_rows' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KLowNorm.lowNormCost_rows

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNorm.lowNormCost_columns' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KLowNorm.lowNormCost_columns


/-- info: 'Nightstream.Implementation.R1CS.Canonical.KLowNorm.lowNormRows_use_squareColumn' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KLowNorm.lowNormRows_use_squareColumn

end NightstreamTests.Axioms.CanonicalKLowNorm
