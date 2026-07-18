import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.Accumulator.OrderedCommitmentMessage
import tests.Axioms.Support

/-! Fail-closed dependency gate for the ordered-commitment hash message. -/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentMessage.domainNats_eq' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentMessage.domainNats_eq

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentMessage.serialize_injective' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentMessage.serialize_injective

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentMessage.digest_eq_or_fieldHashCollision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentMessage.digest_eq_or_fieldHashCollision

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentMessage.claimDigest_eq_payloadDigest' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentMessage.claimDigest_eq_payloadDigest

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentMessage.fixed_serialize_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentMessage.fixed_serialize_length
