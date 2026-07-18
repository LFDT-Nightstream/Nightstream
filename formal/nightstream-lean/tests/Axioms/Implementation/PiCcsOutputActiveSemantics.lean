import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.ActiveProfile
import tests.Axioms.Support

/-!
Fail-closed kernel dependency expectations for the complete shape-indexed
`Pi_CCS` output field encoding.
-/

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Encoding.encodeKVectorFamily_injective' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Encoding.encodeKVectorFamily_injective

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSemantics.serialize_injective' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSemantics.serialize_injective

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSemantics.serialize_length_15_sources_13_matrices' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSemantics.serialize_length_15_sources_13_matrices

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveProfile.serialize_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveProfile.serialize_length

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveProfile.relationShape_eq' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveProfile.relationShape_eq

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveProfile.selectiveShape_profile_eq_steadyFixedPoint' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveProfile.selectiveShape_profile_eq_steadyFixedPoint

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveProfile.selectiveShape_not_diagnosticProfile' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveProfile.selectiveShape_not_diagnosticProfile

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveProfile.selective_serialize_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveProfile.selective_serialize_length
