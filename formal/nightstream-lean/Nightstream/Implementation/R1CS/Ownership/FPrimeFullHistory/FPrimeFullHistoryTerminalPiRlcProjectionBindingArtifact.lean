import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiRlcProjectionBindingPieces0
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiRlcProjectionBindingPieces1
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiRlcProjectionBindingPieces2
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiRlcProjectionBindingPieces3
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiRlcProjectionBindingPieces4
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiRlcProjectionBindingPieces5
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiRlcProjectionBindingPieces6
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiRlcProjectionBindingPieces7
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiRlcProjectionBindingPieces8

/-! Exact ordered row certificate for the terminal Pi_RLC projection-preimage binding owner. Hash is drift metadata only. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiRlcProjectionBinding

open Nightstream.Implementation.R1CS.OwnerCertificate

set_option maxRecDepth 1048576

def rangeSha256 : String := "9b2c25f1d0eb11799f4f548e89b35e1ab90d8661714da12e98e2118712920e62"
def rowStart : Nat := 2854212
def rowEnd : Nat := 3329745
def rowCount : Nat := 475533

def pieces : List Piece :=
  Generated.pieces0 ++
    Generated.pieces1 ++
    Generated.pieces2 ++
    Generated.pieces3 ++
    Generated.pieces4 ++
    Generated.pieces5 ++
    Generated.pieces6 ++
    Generated.pieces7 ++
    Generated.pieces8

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

end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiRlcProjectionBinding
