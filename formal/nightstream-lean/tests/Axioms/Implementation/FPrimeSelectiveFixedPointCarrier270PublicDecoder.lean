import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Carrier270
import tests.Axioms.Support

/-! Kernel dependency report for the bounded fixed-point public decoder. -/

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicDecoder.generatedCoordinate_exact' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PublicDecoder.generatedCoordinate_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicAssignment.artifactPublicInput_eq_projectPublicInput' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PublicAssignment.artifactPublicInput_eq_projectPublicInput
