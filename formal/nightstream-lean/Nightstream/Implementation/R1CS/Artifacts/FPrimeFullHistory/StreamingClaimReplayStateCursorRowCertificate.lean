import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay
import Nightstream.Implementation.R1CS.Core.ConstantPins

/-!
Contract: exact four-row state-pin and three-row cursor slices for both
Rust-emitted claim-replay arms.

Assurance tier: Rust-to-Lean artifact row certificate.

Owns only seven glue rows after the nine expected-carry rows. It does not
inspect later glue rows or any repeated leaf program.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayStateCursorRowCertificate

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay

def fullStatePins : List (Nat × Nat) :=
  [(9, 3), (419, 3), (18, 0), (428, 0)]

def finalStatePins : List (Nat × Nat) :=
  [(9, 3), (419, 3), (18, 0), (428, 3)]

def fullStatePinRows : List Row :=
  [⟨[(0, goldilocksP - 3), (9, 1)], [(0, 1)], []⟩,
   ⟨[(0, goldilocksP - 3), (419, 1)], [(0, 1)], []⟩,
   ⟨[(18, 1)], [(0, 1)], []⟩,
   ⟨[(428, 1)], [(0, 1)], []⟩]

def finalStatePinRows : List Row :=
  [⟨[(0, goldilocksP - 3), (9, 1)], [(0, 1)], []⟩,
   ⟨[(0, goldilocksP - 3), (419, 1)], [(0, 1)], []⟩,
   ⟨[(18, 1)], [(0, 1)], []⟩,
   ⟨[(0, goldilocksP - 3), (428, 1)], [(0, 1)], []⟩]

theorem fullArm_statePinRows_exact :
    ((fullArm.glueRows.map IndexedRow.row).drop 9).take 4 =
      fullStatePinRows := by
  rfl

theorem finalArm_statePinRows_exact :
    ((finalArm.glueRows.map IndexedRow.row).drop 9).take 4 =
      finalStatePinRows := by
  rfl

def fullCursorRows : List Row :=
  [⟨[(0, 97280), (19, 1), (344, goldilocksP - 1024)], [(0, 1)], []⟩,
   ⟨[(0, goldilocksP - 1024), (19, goldilocksP - 1), (429, 1)],
      [(0, 1)], []⟩,
   ⟨[(0, goldilocksP - 1), (344, goldilocksP - 1), (754, 1)],
      [(0, 1)], []⟩]

def finalCursorRows : List Row :=
  [⟨[(0, 97280), (19, 1), (344, goldilocksP - 1024)], [(0, 1)], []⟩,
   ⟨[(0, goldilocksP - 575), (19, goldilocksP - 1), (429, 1)],
      [(0, 1)], []⟩,
   ⟨[(0, goldilocksP - 1), (344, goldilocksP - 1), (754, 1)],
      [(0, 1)], []⟩]

theorem fullArm_cursorRows_exact :
    ((fullArm.glueRows.map IndexedRow.row).drop 13).take 3 =
      fullCursorRows := by
  rfl

theorem finalArm_cursorRows_exact :
    ((finalArm.glueRows.map IndexedRow.row).drop 13).take 3 =
      finalCursorRows := by
  rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayStateCursorRowCertificate
