import Nightstream.Implementation.NebulaV2.Commitment.Terminal.BundleOpeningRows
import tests.Axioms.Support

open Nightstream.Implementation.NebulaV2

/-- info: 'Nightstream.Implementation.NebulaV2.TerminalBundleOpeningRows.sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalBundleOpeningRows.sound

/-- info: 'Nightstream.Implementation.NebulaV2.TerminalBundleOpeningRows.sound_opens_codec_bundle' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalBundleOpeningRows.sound_opens_codec_bundle
