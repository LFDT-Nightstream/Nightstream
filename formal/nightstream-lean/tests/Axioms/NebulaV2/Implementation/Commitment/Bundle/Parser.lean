import Nightstream.Implementation.NebulaV2.Commitment.Bundle.Parser
import tests.Axioms.Support

open Nightstream.Implementation.NebulaV2.CommitmentBundleParser

/-! Axiom gates for the executable mandatory-bundle parser. -/

/-- info: 'Nightstream.Implementation.NebulaV2.CommitmentBundleParser.parse_success_reencodes' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms parse_success_reencodes

/-- info: 'Nightstream.Implementation.NebulaV2.CommitmentBundleParser.parse_blockOfBundle' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms parse_blockOfBundle

/-- info: 'Nightstream.Implementation.NebulaV2.CommitmentBundleParser.rejects_modulus_alias' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rejects_modulus_alias
