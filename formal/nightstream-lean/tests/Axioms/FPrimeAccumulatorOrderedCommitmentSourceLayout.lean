import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.Accumulator.OrderedCommitmentSourceLayout
import tests.Axioms.Support

/-! Fail-closed dependency gate for the prospective source-column layout. -/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentSourceLayout.expectedSourceColumns_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentSourceLayout.expectedSourceColumns_length

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentSourceLayout.expectedSourceColumns_values' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentSourceLayout.expectedSourceColumns_values

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentSourceLayout.domainFields_eq_residues' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentSourceLayout.domainFields_eq_residues

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentSourceLayout.expectedSourceColumns_fields' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentSourceLayout.expectedSourceColumns_fields
