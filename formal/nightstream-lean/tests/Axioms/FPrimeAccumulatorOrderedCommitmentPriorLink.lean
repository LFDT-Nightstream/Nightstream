import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.Accumulator.OrderedCommitmentPriorLink
import tests.Axioms.Support

/-! Fail-closed dependency gate for the exact ordered-message prior link. -/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentPriorLink.slot_eq_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentPriorLink.slot_eq_or_failure

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentPriorLink.slot_eq_or_failure_of_selectedNifs' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentPriorLink.slot_eq_or_failure_of_selectedNifs
