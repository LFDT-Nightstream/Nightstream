import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcProjection.RhoEvaluations
import tests.Axioms.Support

/-! Fail-closed dependencies for the active shared-rho refinement. -/

open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.RhoEvaluations.Refinement

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.RhoEvaluations.Refinement.ownedSourceRows_outputs_correct' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ownedSourceRows_outputs_correct

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.RhoEvaluations.Refinement.fullRows_outputs_correct' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms fullRows_outputs_correct
