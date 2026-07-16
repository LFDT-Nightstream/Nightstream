import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces0
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces1
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces2
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces3
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces4
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces5
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces6
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces7
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces8
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces9
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces10
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces11
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces12
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces13
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces14
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces15
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces16
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces17
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces18
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces19
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces20
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces21
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces22
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces23
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces24
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces25
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces26
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces27
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces28
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces29
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces30
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces31
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalPiCcsFeTerminalPieces32

/-! Exact ordered row certificate for the terminal Pi_CCS FE-terminal owner. Hash is drift metadata only. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeTerminal

open Nightstream.Implementation.R1CS.OwnerCertificate

set_option maxRecDepth 1048576

def rangeSha256 : String := "e9d4a96ce0533ccfc0e46af102d09cc7048dc52e3d6c97ea1149d6c3e5ae7225"
def rowStart : Nat := 1851871
def rowEnd : Nat := 1866714
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
