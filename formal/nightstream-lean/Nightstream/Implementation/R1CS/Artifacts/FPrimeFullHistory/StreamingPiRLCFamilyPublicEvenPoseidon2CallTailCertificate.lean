import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPublicEvenPoseidon2CallCertificate

/-!
Contract: second bounded Poseidon2 call-geometry certificate for the
Rust-emitted even PiRLC public-family arm.

Owns four 64-call leaves, one 32-call leaf, and their structural composition
with the four earlier leaves. It owns no Poseidon2 semantics or row ownership.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicEvenPoseidon2CallTailCertificate

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPublic
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublic.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicCertificateSupport
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicEvenPoseidon2CallCertificate

theorem evenChunk4_valid :
    ∀ call ∈ evenChunk4,
      PoseidonCallValid 1301126 call ∧ 310646 ≤ call.rowStart := by
  norm_num [evenChunk4, evenTail4, evenTail3, evenTail2, evenTail1, evenTail0,
    PoseidonCallValid, evenArm]

theorem evenChunk5_valid :
    ∀ call ∈ evenChunk5,
      PoseidonCallValid 1301126 call ∧ 310646 ≤ call.rowStart := by
  norm_num [evenChunk5, evenTail5, evenTail4, evenTail3, evenTail2, evenTail1,
    evenTail0, PoseidonCallValid, evenArm]

theorem evenChunk6_valid :
    ∀ call ∈ evenChunk6,
      PoseidonCallValid 1301126 call ∧ 310646 ≤ call.rowStart := by
  norm_num [evenChunk6, evenTail6, evenTail5, evenTail4, evenTail3, evenTail2,
    evenTail1, evenTail0, PoseidonCallValid, evenArm]

theorem evenChunk7_valid :
    ∀ call ∈ evenChunk7,
      PoseidonCallValid 1301126 call ∧ 310646 ≤ call.rowStart := by
  norm_num [evenChunk7, evenTail7, evenTail6, evenTail5, evenTail4, evenTail3, evenTail2,
    evenTail1, evenTail0, PoseidonCallValid, evenArm]

theorem evenTail8_valid :
    ∀ call ∈ evenTail8,
      PoseidonCallValid 1301126 call ∧ 310646 ≤ call.rowStart := by
  norm_num [evenTail8, evenTail7, evenTail6, evenTail5, evenTail4, evenTail3, evenTail2,
    evenTail1, evenTail0, PoseidonCallValid, evenArm]

theorem evenArm_poseidon2Calls_valid : evenArm.Poseidon2CallsValid := by
  unfold RawArm.Poseidon2CallsValid
  change ∀ call ∈ evenTail0,
    PoseidonCallValid 1301126 call ∧ 310646 ≤ call.rowStart
  exact valid_of_take_drop evenChunk0_valid
    (valid_of_take_drop evenChunk1_valid
      (valid_of_take_drop evenChunk2_valid
        (valid_of_take_drop evenChunk3_valid
          (valid_of_take_drop evenChunk4_valid
            (valid_of_take_drop evenChunk5_valid
              (valid_of_take_drop evenChunk6_valid
                (valid_of_take_drop evenChunk7_valid evenTail8_valid)))))))

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicEvenPoseidon2CallTailCertificate
