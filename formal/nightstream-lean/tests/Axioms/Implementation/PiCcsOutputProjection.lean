import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.Projection.Necessity
import tests.Axioms.Support

/-!
Fail-closed kernel dependency expectations for the fixed-profile `Pi_CCS`
output reconstruction and representation-necessity theorems.
-/

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Projection.FixedProfile.supportedProjection_unique' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Projection.FixedProfile.supportedProjection_unique

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Projection.Necessity.alterCt_sameProjection_notCanonical' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Projection.Necessity.alterCt_sameProjection_notCanonical

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Projection.Necessity.alterYRingPadding_sameProjection_notCanonical' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Projection.Necessity.alterYRingPadding_sameProjection_notCanonical

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Projection.Necessity.alterYZcolPadding_sameProjection_notCanonical' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Projection.Necessity.alterYZcolPadding_sameProjection_notCanonical
