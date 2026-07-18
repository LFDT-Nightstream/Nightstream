import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.Accumulator.OrderedCommitmentTypedDecoder
import tests.Axioms.Support

/-! Fail-closed dependency gate for the artifact-to-typed decoder. -/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentTypedDecoder.childFields_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentTypedDecoder.childFields_length

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentTypedDecoder.serialize_decodedPayload' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentTypedDecoder.serialize_decodedPayload
