import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay
import Nightstream.Implementation.R1CS.Core.EqualityPins

/-!
Contract: exact nine-row expected-state carry prefix for both Rust-emitted
claim-replay arms.

Assurance tier: Rust-to-Lean artifact row certificate.

Owns only the first nine glue rows. It does not inspect or validate the
remaining glue rows or any repeated leaf program.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayExpectedCarryRowCertificate

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.EqualityPins
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay

def expectedCarryPairs : List (Nat × Nat) :=
  (List.range 9).map fun offset => (1 + offset, 411 + offset)

def expectedCarryRows : List Row :=
  EqualityPins.rows expectedCarryPairs

theorem fullArm_expectedCarryRows_exact :
    (fullArm.glueRows.map IndexedRow.row).take 9 = expectedCarryRows := by
  rfl

theorem finalArm_expectedCarryRows_exact :
    (finalArm.glueRows.map IndexedRow.row).take 9 = expectedCarryRows := by
  rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayExpectedCarryRowCertificate
