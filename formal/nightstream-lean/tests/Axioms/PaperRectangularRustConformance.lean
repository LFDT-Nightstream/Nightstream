import Nightstream.Implementation.Rust.CanonicalConformance.PiCcsPaperRectangular.Conformance
import tests.Axioms.Support

open Nightstream.Implementation.Rust.CanonicalConformance.PiCcsPaperRectangular

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.PiCcsPaperRectangular.Artifact.carried_gamma_slots_match' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Artifact.carried_gamma_slots_match

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.PiCcsPaperRectangular.Artifact.covers_both_rectangular_directions' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Artifact.covers_both_rectangular_directions
