import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitialPieces0
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitialPieces1
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitialPieces2
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitialPieces3
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitialPieces4
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitialPieces5
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitialPieces6
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitialPieces7
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitialPieces8
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitialPieces9
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitialPieces10
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitialPieces11
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitialPieces12
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitialPieces13
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitialPieces14
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitialPieces15
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitialPieces16
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitialPieces17
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitialPieces18
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitialPieces19
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitialPieces20
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitialPieces21
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitialPieces22
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitialPieces23
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitialPieces24
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitialPieces25
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitialPieces26
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitialPieces27
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitialPieces28
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitialPieces29
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitialPieces30
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitialPieces31
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitialPieces32

/-! Exact ordered row certificate for the terminal Pi_CCS FE-initial owner. Hash is drift metadata only. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiCcsFeInitial

open Nightstream.Implementation.R1CS.OwnerCertificate

set_option maxRecDepth 1048576

def rangeSha256 : String := "a607a000d6e3e9b73deb7e51ee78d772c2a9d31d459f88db262fe0a68d8337e4"
def rowStart : Nat := 1365967
def rowEnd : Nat := 1380610
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
