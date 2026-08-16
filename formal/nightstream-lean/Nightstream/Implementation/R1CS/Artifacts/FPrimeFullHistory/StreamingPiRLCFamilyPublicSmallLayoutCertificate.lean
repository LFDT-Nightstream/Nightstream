import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPublic

/-!
Contract: bounded XOut, public-word, and digest-pin layout certificates for
both Rust-emitted PiRLC public-family arms.

Owns only fixed lists with at most 32 columns. It owns no state schedule,
hash-round schedule, or row data.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicSmallLayoutCertificate

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPublic
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublic.Artifact

theorem evenArm_xOutColumnLayout_valid : evenArm.XOutColumnLayoutValid := by
  norm_num [RawArm.XOutColumnLayoutValid, columnsValid, evenArm]

theorem oddArm_xOutColumnLayout_valid : oddArm.XOutColumnLayoutValid := by
  norm_num [RawArm.XOutColumnLayoutValid, columnsValid, oddArm]

theorem evenArm_publicAndPinLayout_valid :
    evenArm.PublicAndPinLayoutValid := by
  norm_num [RawArm.PublicAndPinLayoutValid, columnsValid, evenArm]

theorem oddArm_publicAndPinLayout_valid :
    oddArm.PublicAndPinLayoutValid := by
  norm_num [RawArm.PublicAndPinLayoutValid, columnsValid, oddArm]

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicSmallLayoutCertificate
