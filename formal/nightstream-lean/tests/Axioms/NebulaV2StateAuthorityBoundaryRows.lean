import Nightstream.Implementation.NebulaV2.StateAuthorityBoundaryRows
import tests.Axioms.Support

open Nightstream.Implementation.NebulaV2

/-- info: 'Nightstream.Implementation.NebulaV2.AuthoritativeStateOutputBinding.typedFrame_canonical_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms AuthoritativeStateOutputBinding.typedFrame_canonical_of_rows

/-- info: 'Nightstream.Implementation.NebulaV2.StateAuthorityBoundaryRows.digest_eq_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms StateAuthorityBoundaryRows.digest_eq_of_rows

/-- info: 'Nightstream.Implementation.NebulaV2.StateAuthorityBoundaryRows.Boundary.digest_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms StateAuthorityBoundaryRows.Boundary.digest_eq

/-- info: 'Nightstream.Implementation.NebulaV2.StateAuthorityBoundaryRows.Boundary.sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms StateAuthorityBoundaryRows.Boundary.sound

/-- info: 'Nightstream.Implementation.NebulaV2.StateAuthorityBoundaryRows.candidate_sound_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms StateAuthorityBoundaryRows.candidate_sound_or_collision
