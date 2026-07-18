import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcChallenge.Sampler.FirstAccepted
import tests.Axioms.Support

/-! Fail-closed dependencies for the active field-derived sampler closure. -/

open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.Sampler

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.Sampler.ScalarSemantics.counterChain' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ScalarSemantics.counterChain

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.Sampler.TailSources.accepted_sourceBindings' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TailSources.accepted_sourceBindings

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.Sampler.FirstAccepted.enoughAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms FirstAccepted.enoughAccepted

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.Sampler.FirstAccepted.outputAt_refines' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms FirstAccepted.outputAt_refines

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.Sampler.FirstAccepted.accepted_refines' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms FirstAccepted.accepted_refines

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.Sampler.FirstAccepted.embeddedRows_refine_firstAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms FirstAccepted.embeddedRows_refine_firstAccepted
