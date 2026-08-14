import Nightstream.Implementation.Nebula.NIFS.Running.FullClaimDecoder
import tests.Axioms.Support

open Nightstream.Implementation.Nebula.ProductFullClaimDecoder

/-- info: 'Nightstream.Implementation.Nebula.ProductFullClaimDecoder.widths_totalBits' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms widths_totalBits

/-- info: 'Nightstream.Implementation.Nebula.ProductFullClaimDecoder.decodeValue_block' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms decodeValue_block

/-- info: 'Nightstream.Implementation.Nebula.ProductFullClaimDecoder.decodeValue_success' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms decodeValue_success

/-- info: 'Nightstream.Implementation.Nebula.ProductFullClaimDecoder.decode_block' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms decode_block

/-- info: 'Nightstream.Implementation.Nebula.ProductFullClaimDecoder.decode_success' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms decode_success
