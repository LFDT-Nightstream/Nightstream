import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces0
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces1
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces2
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces3
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces4
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces5
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces6
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces7
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces8
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces9
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces10
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces11
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces12
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces13
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces14
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces15
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces16
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces17
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces18
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces19
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces20
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces21
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces22
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces23
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces24
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces25
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces26
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces27
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces28
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces29
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces30
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces31
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces32

/-! Exact ordered row certificate for the terminal Pi_CCS FE-terminal owner. Hash is drift metadata only. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminal

open Nightstream.Implementation.R1CS.OwnerCertificate

set_option maxRecDepth 1048576

def rangeSha256 : String := "618fbb9c0635d0dcbf1d737c1ce5aed293cf29655d37851cc89dd27f9ed9de14"
def rowStart : Nat := 1634049
def rowEnd : Nat := 1648892
def rowCount : Nat := 14843

def pieces : List Piece :=
  Generated.pieces0 ++
    Generated.pieces1 ++
    Generated.pieces2 ++
    Generated.pieces3 ++
    Generated.pieces4 ++
    Generated.pieces5 ++
    Generated.pieces6 ++
    Generated.pieces7 ++
    Generated.pieces8 ++
    Generated.pieces9 ++
    Generated.pieces10 ++
    Generated.pieces11 ++
    Generated.pieces12 ++
    Generated.pieces13 ++
    Generated.pieces14 ++
    Generated.pieces15 ++
    Generated.pieces16 ++
    Generated.pieces17 ++
    Generated.pieces18 ++
    Generated.pieces19 ++
    Generated.pieces20 ++
    Generated.pieces21 ++
    Generated.pieces22 ++
    Generated.pieces23 ++
    Generated.pieces24 ++
    Generated.pieces25 ++
    Generated.pieces26 ++
    Generated.pieces27 ++
    Generated.pieces28 ++
    Generated.pieces29 ++
    Generated.pieces30 ++
    Generated.pieces31 ++
    Generated.pieces32

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

end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminal
