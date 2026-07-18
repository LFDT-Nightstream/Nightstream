import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcProjection.YZcolNormalForm
import tests.Axioms.Support

/-! Fail-closed dependencies for the active PiRLC `y_zcol` normal form. -/

open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.YZcolNormalForm

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.YZcolNormalForm.challenge_columns_shared' depends on axioms: [propext,
 Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms challenge_columns_shared

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.YZcolNormalForm.batchExact_decodedOutput_eq_sourceAggregate' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms batchExact_decodedOutput_eq_sourceAggregate

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.YZcolNormalForm.batchExact_parent_eq_sourceAggregate' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms batchExact_parent_eq_sourceAggregate

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.YZcolNormalForm.completeSourceRows_parent_eq_sourceAggregate_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms completeSourceRows_parent_eq_sourceAggregate_or_badRoot
