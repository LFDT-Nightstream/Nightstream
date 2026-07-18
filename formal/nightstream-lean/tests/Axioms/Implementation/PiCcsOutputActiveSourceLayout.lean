import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.ActiveSourceLayout.Counts
import tests.Axioms.Support

/-!
Fail-closed kernel dependency expectations for the typed active `Pi_CCS`
output role tree.
-/

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout.sourceRoles_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout.sourceRoles_length

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout.ownerFieldCounts_reconcile' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout.ownerFieldCounts_reconcile

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout.ownerFieldCount_verifierShape' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout.ownerFieldCount_verifierShape

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout.ownerFieldCount_yRingOutput' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout.ownerFieldCount_yRingOutput

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout.ownerFieldCount_yZcolOutput' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout.ownerFieldCount_yZcolOutput

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout.sourceRoleValues_eq_serialize' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout.sourceRoleValues_eq_serialize

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout.decodedFields_eq_serialize' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout.decodedFields_eq_serialize
