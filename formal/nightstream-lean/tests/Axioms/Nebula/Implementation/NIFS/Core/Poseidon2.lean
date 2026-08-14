import Nightstream.Implementation.Nebula.NIFS.Core.Poseidon2
import tests.Axioms.Support

/-! Fail-closed dependency guard for the exact V2 product transcript. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductPoseidon2.eventSchedule_length' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPoseidon2.eventSchedule_length

/-- info: 'Nightstream.Implementation.Nebula.ProductPoseidon2.bundleFields_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPoseidon2.bundleFields_length

/-- info: 'Nightstream.Implementation.Nebula.ProductPoseidon2.publicNifsFields_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPoseidon2.publicNifsFields_length

/-- info: 'Nightstream.Implementation.Nebula.ProductPoseidon2.outputFields_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPoseidon2.outputFields_length

/-- info: 'Nightstream.Implementation.Nebula.ProductPoseidon2.piDecOutputFields_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPoseidon2.piDecOutputFields_length

/-- info: 'Nightstream.Implementation.Nebula.ProductPoseidon2.piRlcResponse_valid' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductPoseidon2.piRlcResponse_valid
