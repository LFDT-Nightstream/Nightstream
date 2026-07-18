import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.YZcolProjection
import tests.Axioms.Support

/-!
Fail-closed kernel dependency expectations for the fixed-profile parent
`y_zcol` output-evaluation ownership tree.
-/

open Nightstream.Implementation.R1CS.FPrimeRecursiveYZcolProjection

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursiveYZcolProjection.data_check' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms data_check

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursiveYZcolProjection.owned_row_count' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms owned_row_count

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursiveYZcolProjection.source_rows_match' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms source_rows_match
