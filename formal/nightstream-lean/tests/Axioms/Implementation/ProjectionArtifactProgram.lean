import Nightstream.Implementation.R1CS.Correspondence.Projection.ArtifactProgram
import tests.Axioms.Support

/-! Fail-closed kernel dependency expectation for exact projection-row
satisfaction transport. -/

/-- info: 'Nightstream.Implementation.R1CS.ProjectionArtifactProgram.Certificate.traceRowsHold_of_embedded' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.ProjectionArtifactProgram.Certificate.traceRowsHold_of_embedded
