import Nightstream.Implementation.NebulaV2.NIFS.Running.FullClaimDecoder
import tests.Axioms.Support

open Nightstream.Implementation.NebulaV2.ProductFullClaimDecoder

/-- info: 'Nightstream.Implementation.NebulaV2.ProductFullClaimDecoder.widths_totalBits' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms widths_totalBits

/-- info: 'Nightstream.Implementation.NebulaV2.ProductFullClaimDecoder.decodeValue_block' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms decodeValue_block

/-- info: 'Nightstream.Implementation.NebulaV2.ProductFullClaimDecoder.decodeValue_success' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms decodeValue_success

/-- info: 'Nightstream.Implementation.NebulaV2.ProductFullClaimDecoder.decode_block' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms decode_block

/-- info: 'Nightstream.Implementation.NebulaV2.ProductFullClaimDecoder.decode_success' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms decode_success
