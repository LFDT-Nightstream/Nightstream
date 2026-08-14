import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingFPrimeProgramArtifact
import tests.Axioms.Support

/-! Fail-closed axiom guard for the Rust streaming F-prime program. -/

namespace NightstreamTests.Axioms.Implementation.FPrimeFullHistoryStreamingProgram

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramArtifact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramArtifact.artifact_valid' does not depend on any axioms -/
#guard_msgs in
#audit_axioms artifact_valid

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramArtifact.rust_program_exact' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms rust_program_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramArtifact.rust_program_length_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rust_program_length_exact

end NightstreamTests.Axioms.Implementation.FPrimeFullHistoryStreamingProgram
