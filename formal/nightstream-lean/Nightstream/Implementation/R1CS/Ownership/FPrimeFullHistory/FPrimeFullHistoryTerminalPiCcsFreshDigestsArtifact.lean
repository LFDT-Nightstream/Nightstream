import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFreshDigestsPieces0
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFreshDigestsPieces1
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFreshDigestsPieces2

/-! Exact ordered row certificate for the terminal Pi_CCS fresh-claim digest owner. Hash is drift metadata only. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFreshDigests

open Nightstream.Implementation.R1CS.OwnerCertificate

set_option maxRecDepth 1048576

def rangeSha256 : String := "56bfda02e85bf05c873220e0f4ee86ead24c1bf9ecb23437c42d5789f4eb1fb9"
def rowStart : Nat := 1136175
def rowEnd : Nat := 1313780
def rowCount : Nat := 177605

def pieces : List Piece :=
  Generated.pieces0 ++
    Generated.pieces1 ++
    Generated.pieces2

def owner : Owner := ⟨rowStart, rowEnd, pieces⟩

theorem owner_valid : owner.Valid := by native_decide

def rows := owner.rows

theorem rows_length : rows.length = rowCount := by
  simpa [rows, rowCount, rowStart, rowEnd] using Owner.rows_length owner_valid

/-- Independent executable semantics for every compact piece. -/
def Accepted (assignment : Nat → Nat) : Prop := owner.Accepted assignment

def check (assignment : Nat → Nat) : Bool := owner.check assignment

theorem check_eq_true_iff (assignment : Nat → Nat) :
    check assignment = true ↔ Accepted assignment :=
  Owner.check_eq_true_iff owner assignment

theorem sound {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies rows assignment) :
    Accepted assignment :=
  Owner.sound canonical one satisfies

theorem complete {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : Accepted assignment) :
    Satisfies rows assignment :=
  Owner.complete canonical one accepted

end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFreshDigests
