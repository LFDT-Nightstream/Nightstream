import Nightstream.Implementation.NebulaV2.NIFS.Running.Codec
import tests.Axioms.Support

open Nightstream.Implementation.NebulaV2.ProductNifsCodec

/-- info: 'Nightstream.Implementation.NebulaV2.ProductNifsCodec.runningCodec_width' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms runningCodec_width

/-- info: 'Nightstream.Implementation.NebulaV2.ProductNifsCodec.runningBits_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms runningBits_injective

/-- info: 'Nightstream.Implementation.NebulaV2.ProductNifsCodec.decodeRunning_blockOfRunning' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms decodeRunning_blockOfRunning

/-- info: 'Nightstream.Implementation.NebulaV2.ProductNifsCodec.decodeRunning_success_reencodes' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms decodeRunning_success_reencodes

/-- info: 'Nightstream.Implementation.NebulaV2.ProductNifsCodec.codecBundle_injective' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms codecBundle_injective

/-- info: 'Nightstream.Implementation.NebulaV2.ProductNifsCodec.codecBundle_protocolBundleOf' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms codecBundle_protocolBundleOf

/-- info: 'Nightstream.Implementation.NebulaV2.ProductNifsCodec.protocolBundleOf_codecBundle' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms protocolBundleOf_codecBundle

/-- info: 'Nightstream.Implementation.NebulaV2.ProductNifsCodec.publicInputOf_injective' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms publicInputOf_injective

/-- info: 'Nightstream.Implementation.NebulaV2.ProductNifsCodec.freshOf_pair_injective' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms freshOf_pair_injective

/-- info: 'Nightstream.Implementation.NebulaV2.ProductNifsCodec.runningCodecFor_width' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms runningCodecFor_width

/-- info: 'Nightstream.Implementation.NebulaV2.ProductNifsCodec.freshOfFor_pair_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms freshOfFor_pair_injective
