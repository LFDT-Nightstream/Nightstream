import Nightstream.Implementation.Rust.CanonicalConformance.PiCcsPaddedRowIdentity.Conformance
import tests.Axioms.Support

open Nightstream.Implementation.Rust.CanonicalConformance.PiCcsPaddedRowIdentity

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.PiCcsPaddedRowIdentity.Artifact.carried_gamma_slots_match' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Artifact.carried_gamma_slots_match

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.PiCcsPaddedRowIdentity.Artifact.sample_proof_codec_matches' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Artifact.sample_proof_codec_matches

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.PiCcsPaddedRowIdentity.Artifact.production_output_field_count_matches' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Artifact.production_output_field_count_matches
