import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.YZcolIdentities
import tests.Axioms.Support

/-! Fail-closed dependencies for both complete active PiRLC `y_zcol` artifacts. -/

open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.YZcolIdentities

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.YZcolIdentities.local_rows_distinct' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms local_rows_distinct

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.YZcolIdentities.local_output_rows_distinct' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms local_output_rows_distinct

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.YZcolIdentities.data_check' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms data_check
