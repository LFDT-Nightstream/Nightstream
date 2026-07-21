import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Nifs.PiDec.SourceRefinement
import tests.Axioms.Support

/-! Fail-closed dependency gate for the bounded active strict-`PiDEC`
source artifact. `Lean.trustCompiler` is intentional for generated sparse-data
certificates and is not claimed as a kernel-only result. -/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Nifs.PiDec.SourceRefinement.SourceArtifact.sourceRows_exact' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Nifs.PiDec.SourceRefinement.SourceArtifact.sourceRows_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Nifs.PiDec.SourceRefinement.sourceRows_imply_paperAccepted' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Nifs.PiDec.SourceRefinement.sourceRows_imply_paperAccepted
