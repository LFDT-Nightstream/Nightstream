import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPublicOddPoseidon2CallCertificate

/-!
Contract: second bounded Poseidon2 call-geometry certificate for the
Rust-emitted odd PiRLC public-family arm.

Owns three 64-call leaves, one 42-call leaf, and their structural composition
with the four earlier leaves. It owns no Poseidon2 semantics or row ownership.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicOddPoseidon2CallTailCertificate

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPublic
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublic.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicCertificateSupport
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicOddPoseidon2CallCertificate

theorem oddChunk4_valid :
    ∀ call ∈ oddChunk4,
      PoseidonCallValid 1234286 call ∧ 276206 ≤ call.rowStart := by
  norm_num [oddChunk4, oddTail4, oddTail3, oddTail2, oddTail1, oddTail0,
    PoseidonCallValid, oddArm]

theorem oddChunk5_valid :
    ∀ call ∈ oddChunk5,
      PoseidonCallValid 1234286 call ∧ 276206 ≤ call.rowStart := by
  norm_num [oddChunk5, oddTail5, oddTail4, oddTail3, oddTail2, oddTail1,
    oddTail0, PoseidonCallValid, oddArm]

theorem oddChunk6_valid :
    ∀ call ∈ oddChunk6,
      PoseidonCallValid 1234286 call ∧ 276206 ≤ call.rowStart := by
  norm_num [oddChunk6, oddTail6, oddTail5, oddTail4, oddTail3, oddTail2,
    oddTail1, oddTail0, PoseidonCallValid, oddArm]

theorem oddTail7_valid :
    ∀ call ∈ oddTail7,
      PoseidonCallValid 1234286 call ∧ 276206 ≤ call.rowStart := by
  norm_num [oddTail7, oddTail6, oddTail5, oddTail4, oddTail3, oddTail2,
    oddTail1, oddTail0, PoseidonCallValid, oddArm]

theorem oddArm_poseidon2Calls_valid : oddArm.Poseidon2CallsValid := by
  unfold RawArm.Poseidon2CallsValid
  change ∀ call ∈ oddTail0,
    PoseidonCallValid 1234286 call ∧ 276206 ≤ call.rowStart
  exact valid_of_take_drop oddChunk0_valid
    (valid_of_take_drop oddChunk1_valid
      (valid_of_take_drop oddChunk2_valid
        (valid_of_take_drop oddChunk3_valid
          (valid_of_take_drop oddChunk4_valid
            (valid_of_take_drop oddChunk5_valid
              (valid_of_take_drop oddChunk6_valid oddTail7_valid))))))

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicOddPoseidon2CallTailCertificate
