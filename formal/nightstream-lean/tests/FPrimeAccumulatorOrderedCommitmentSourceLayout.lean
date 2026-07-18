import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.Accumulator.OrderedCommitmentSourceLayout

/-! Focused interface gate for the exact prospective source-column layout. -/

open Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentSourceLayout

#check pointColumns_length
#check childCommitmentColumns_length
#check expectedSourceColumns_length
#check expectedSourceColumns_values
#check domainFields_eq_residues
#check expectedSourceColumns_fields

#guard domainConstantValues.length = 10
#guard (pointColumns []).length = 2
#guard (childCommitmentColumns []).length = 13608
