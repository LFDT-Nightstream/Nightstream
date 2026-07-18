import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcProjection.BetaLadder
import tests.Axioms.Support

/-! Fail-closed dependencies for the active beta-ladder refinement. -/

open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.BetaLadder.Refinement

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.BetaLadder.Refinement.ownedSourceRows_ladder_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ownedSourceRows_ladder_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.BetaLadder.Refinement.ownedSourceRows_y_zcol_sharedPowers' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ownedSourceRows_y_zcol_sharedPowers

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.BetaLadder.Refinement.fullRows_y_zcol_sharedPowers' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms fullRows_y_zcol_sharedPowers
