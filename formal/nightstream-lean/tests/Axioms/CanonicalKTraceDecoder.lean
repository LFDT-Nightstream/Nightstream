import Nightstream.Implementation.R1CS.Canonical.KTraceDecoder
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKTraceDecoder

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceDecoder.decodeVector_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KTraceDecoder.decodeVector_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceDecoder.decodeModulus_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KTraceDecoder.decodeModulus_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceDecoder.carriedValue_decodeBase' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KTraceDecoder.carriedValue_decodeBase

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceDecoder.carriedValue_decodeVector' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KTraceDecoder.carriedValue_decodeVector

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceDecoder.decodeBase_mentions' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KTraceDecoder.decodeBase_mentions

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceDecoder.decodeBase_high_empty' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KTraceDecoder.decodeBase_high_empty

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceDecoder.decodeVector_belowBase' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KTraceDecoder.decodeVector_belowBase

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceDecoder.decodeModulus_belowBase' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KTraceDecoder.decodeModulus_belowBase

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceDecoder.decoded_output_sized' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KTraceDecoder.decoded_output_sized

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceDecoder.decoded_quotient_sized' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KTraceDecoder.decoded_quotient_sized

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceDecoder.projected_decodeVector' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KTraceDecoder.projected_decodeVector

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceDecoder.carriedValue_decodeModulus' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KTraceDecoder.carriedValue_decodeModulus

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceDecoder.projected_decodeModulus' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KTraceDecoder.projected_decodeModulus

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceDecoder.pairSum_toPair' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KTraceDecoder.pairSum_toPair

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceDecoder.eval_identity_lhs' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KTraceDecoder.eval_identity_lhs

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceDecoder.eval_identity_rhs' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KTraceDecoder.eval_identity_rhs

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceDecoder.equation_reaches_frozen_eval' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KTraceDecoder.equation_reaches_frozen_eval

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceDecoder.equation_of_exact' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KTraceDecoder.equation_of_exact

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceDecoder.accepted_of_equation' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KTraceDecoder.accepted_of_equation

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KTraceDecoder.exact_or_badRoot_of_equation' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KTraceDecoder.exact_or_badRoot_of_equation
end NightstreamTests.Axioms.CanonicalKTraceDecoder
