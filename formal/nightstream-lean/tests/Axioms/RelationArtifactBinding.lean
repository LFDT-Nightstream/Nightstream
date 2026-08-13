import Nightstream.Assurance.RelationArtifactBinding
import tests.Axioms.Support

/-! Axiom gate for exact verifier-key relation artifact binding. -/

/-- info: 'Nightstream.Assurance.RelationArtifactBinding.accepted_eq_authoritative' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.RelationArtifactBinding.accepted_eq_authoritative

/-- info: 'Nightstream.Assurance.RelationArtifactBinding.changed_rejects' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.RelationArtifactBinding.changed_rejects

/-- info: 'Nightstream.Assurance.RelationArtifactBinding.accepted_selectedProfile' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.RelationArtifactBinding.accepted_selectedProfile

/-- info: 'Nightstream.Assurance.RelationArtifactBinding.accepted_paddedIdentityWidth' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.RelationArtifactBinding.accepted_paddedIdentityWidth
