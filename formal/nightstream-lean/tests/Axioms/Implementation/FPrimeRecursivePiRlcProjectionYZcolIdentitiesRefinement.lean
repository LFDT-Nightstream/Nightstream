import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcProjection.YZcolIdentities
import tests.Axioms.Support

/-!
Fail-closed dependencies for the complete active PiRLC `y_zcol` refinement.
-/

open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.YZcolIdentities.Refinement

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.YZcolIdentities.Refinement.limb0_localDefinitions_permutation' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms limb0_localDefinitions_permutation

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.YZcolIdentities.Refinement.completeSourceRows_batchAccepted' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms completeSourceRows_batchAccepted

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.YZcolIdentities.Refinement.completeSourceRows_batchExact_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms completeSourceRows_batchExact_or_badRoot

/-- info: 'Nightstream.SuperNeo.ProjectionCheck.batchAccepted_implies_exact_or_badRoot' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.ProjectionCheck.batchAccepted_implies_exact_or_badRoot
