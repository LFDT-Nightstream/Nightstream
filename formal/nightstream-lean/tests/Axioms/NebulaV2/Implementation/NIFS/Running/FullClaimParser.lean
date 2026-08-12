import Nightstream.Implementation.NebulaV2.NIFS.Running.FullClaimParser
import tests.Axioms.Support

open Nightstream.Implementation.NebulaV2.ProductFullClaimParser

/-! Axiom gates for the complete executable V2 full-claim parser. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductFullClaimParser.parseValue_block' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms parseValue_block

/-- info: 'Nightstream.Implementation.NebulaV2.ProductFullClaimParser.parseValue_success' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms parseValue_success

/-- info: 'Nightstream.Implementation.NebulaV2.ProductFullClaimParser.decode_success' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms decode_success

/-- info: 'Nightstream.Implementation.NebulaV2.ProductFullClaimParser.parseValue_rejects_profile_mismatch' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms parseValue_rejects_profile_mismatch
