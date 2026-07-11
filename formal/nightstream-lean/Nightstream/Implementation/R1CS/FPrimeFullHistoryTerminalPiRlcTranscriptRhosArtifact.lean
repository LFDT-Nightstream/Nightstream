import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiRlcTranscriptRhosPieces0
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiRlcTranscriptRhosPieces1
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiRlcTranscriptRhosPieces2
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiRlcTranscriptRhosPieces3
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiRlcTranscriptRhosPieces4
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiRlcTranscriptRhosPieces5
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiRlcTranscriptRhosPieces6
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiRlcTranscriptRhosPieces7
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiRlcTranscriptRhosPieces8
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiRlcTranscriptRhosPieces9
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiRlcTranscriptRhosPieces10
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiRlcTranscriptRhosPieces11
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiRlcTranscriptRhosPieces12
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiRlcTranscriptRhosPieces13
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiRlcTranscriptRhosPieces14
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiRlcTranscriptRhosPieces15
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiRlcTranscriptRhosPieces16
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiRlcTranscriptRhosPieces17
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiRlcTranscriptRhosPieces18

/-! Exact ordered row certificate for the terminal Pi_RLC transcript and rho owner. Hash is drift metadata only. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiRlcTranscriptRhos

open Nightstream.Implementation.R1CS.OwnerCertificate

set_option maxRecDepth 1048576

def rangeSha256 : String := "0fab994f9c95a9c9623621449bc9d3314394fab1469f0b1726a107b7f8916bc8"
def rowStart : Nat := 1655551
def rowEnd : Nat := 2953021
def rowCount : Nat := 1297470

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
    Generated.pieces18

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

end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalPiRlcTranscriptRhos
