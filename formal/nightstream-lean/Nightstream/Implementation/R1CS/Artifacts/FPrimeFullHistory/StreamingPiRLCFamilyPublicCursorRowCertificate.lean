import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPublic

/-!
Contract: exact two-row cursor prefix for both Rust-emitted PiRLC public-family
arms.

Owns only the two derived program-cursor rows for each arm. It owns no cursor
semantics, later glue rows, or complete artifact validity.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicCursorRowCertificate

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublic.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPublic

def evenCursorRows : List Row :=
  [⟨[(0, 18446744069414584098), (165662, 18446744069414584320),
      (310880, 1)], [(0, 1)], []⟩,
   ⟨[(0, 18446744069414584098), (165663, 18446744069414584320),
      (310881, 1)], [(0, 1)], []⟩]

def oddCursorRows : List Row :=
  [⟨[(0, 18446744069414584098), (165662, 18446744069414584320),
      (312080, 1)], [(0, 1)], []⟩,
   ⟨[(0, 18446744069414584098), (165663, 18446744069414584320),
      (312081, 1)], [(0, 1)], []⟩]

theorem evenArm_cursorRows_exact :
    (evenArm.glueRows.map IndexedRow.row).take 2 = evenCursorRows := by
  rfl

theorem oddArm_cursorRows_exact :
    (oddArm.glueRows.map IndexedRow.row).take 2 = oddCursorRows := by
  rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicCursorRowCertificate
