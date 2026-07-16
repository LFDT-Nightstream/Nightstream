import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeInitialPieces0
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeInitialPieces1
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeInitialPieces2
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeInitialPieces3
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeInitialPieces4
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeInitialPieces5
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeInitialPieces6
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeInitialPieces7
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeInitialPieces8
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeInitialPieces9
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeInitialPieces10
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeInitialPieces11
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeInitialPieces12
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeInitialPieces13
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeInitialPieces14
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeInitialPieces15
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeInitialPieces16
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeInitialPieces17
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeInitialPieces18
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeInitialPieces19
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeInitialPieces20
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeInitialPieces21
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeInitialPieces22
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeInitialPieces23
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeInitialPieces24
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeInitialPieces25
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeInitialPieces26
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeInitialPieces27
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeInitialPieces28
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeInitialPieces29
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeInitialPieces30
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeInitialPieces31
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeInitialPieces32

/-! Exact ordered row certificate for the terminal Pi_CCS FE-initial owner. Hash is drift metadata only. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitial

open Nightstream.Implementation.R1CS.OwnerCertificate

set_option maxRecDepth 1048576

def rangeSha256 : String := "b15c4a78ae4f86421209107a5b169b1067405b959ec071814e9058c0c63f04c2"
def rowStart : Nat := 1583784
def rowEnd : Nat := 1598427
def rowCount : Nat := 14643

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

end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitial
