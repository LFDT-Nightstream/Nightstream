import Nightstream.Implementation.Nebula.Commitment.Terminal.BundleOpeningRows
import tests.Axioms.Support

open Nightstream.Implementation.Nebula

/-- info: 'Nightstream.Implementation.Nebula.TerminalBundleOpeningRows.sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalBundleOpeningRows.sound

/-- info: 'Nightstream.Implementation.Nebula.TerminalBundleOpeningRows.sound_opens_codec_bundle' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TerminalBundleOpeningRows.sound_opens_codec_bundle
