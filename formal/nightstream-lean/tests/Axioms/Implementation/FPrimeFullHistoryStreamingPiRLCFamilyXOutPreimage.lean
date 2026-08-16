import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyXOutPreimage
import tests.Axioms.Support

/-! Fail-closed axiom guard for the exact family full-XOut preimage role map. -/

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyXOutPreimage

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyXOutPreimage.x_out_preimage_refines' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms x_out_preimage_refines
