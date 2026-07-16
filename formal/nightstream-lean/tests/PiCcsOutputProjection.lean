import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.Projection.Necessity

/-!
Kernel-facing surface for the fixed-profile output projection, deterministic
legacy expansion, and its three representation-necessity witnesses.
-/

namespace tests.PiCcsOutputProjection

open Nightstream.Implementation.R1CS.PiCcsOutputDigest.Projection

#check FixedProfile.activeWidth_eq_54
#check FixedProfile.yRingRows_eq_3
#check FixedProfile.extractActivePayload_canonicalExpand
#check FixedProfile.canonicalExpansion_iff_eq
#check FixedProfile.supportedProjection_unique
#check Necessity.alterCt_sameProjection_notCanonical
#check Necessity.alterYRingPadding_sameProjection_notCanonical
#check Necessity.alterYZcolPadding_sameProjection_notCanonical

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Projection.FixedProfile.supportedProjection_unique' depends on axioms: [propext] -/
#guard_msgs in
#print axioms FixedProfile.supportedProjection_unique

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Projection.Necessity.alterCt_sameProjection_notCanonical' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms Necessity.alterCt_sameProjection_notCanonical

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Projection.Necessity.alterYRingPadding_sameProjection_notCanonical' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms Necessity.alterYRingPadding_sameProjection_notCanonical

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Projection.Necessity.alterYZcolPadding_sameProjection_notCanonical' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms Necessity.alterYZcolPadding_sameProjection_notCanonical

end tests.PiCcsOutputProjection
