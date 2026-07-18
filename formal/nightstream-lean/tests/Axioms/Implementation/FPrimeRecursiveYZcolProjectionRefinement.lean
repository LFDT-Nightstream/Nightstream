import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.YZcolProjection
import tests.Axioms.Support

/-!
Fail-closed kernel dependency expectations for the fixed-profile two-limb
parent `y_zcol` evaluator refinement.
-/

open Nightstream.Implementation.R1CS.FPrimeRecursiveYZcolProjection.Refinement

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursiveYZcolProjection.Refinement.eval_transport' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms eval_transport

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursiveYZcolProjection.Refinement.evaluationDefinitions_refine_parentProjection' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms evaluationDefinitions_refine_parentProjection

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursiveYZcolProjection.Refinement.ownedSourceRows_refine_parentProjection' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ownedSourceRows_refine_parentProjection

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursiveYZcolProjection.Refinement.fullRows_refine_parentProjection' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms fullRows_refine_parentProjection
