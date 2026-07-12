import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryRecursivePiCcsTranscriptPieces0

/-! Exact ordered row certificate for the recursive Pi_CCS transcript owner. Hash is drift metadata only. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursivePiCcsTranscript

open Nightstream.Implementation.R1CS.OwnerCertificate

set_option maxRecDepth 1048576

def rangeSha256 : String := "d9191da170e6ff5c32b16eefcfca3189282d6b5522c034e5b5d0df1bffc632ab"
def rowStart : Nat := 208439
def rowEnd : Nat := 225920
def rowCount : Nat := 17481

def pieces : List Piece :=
  Generated.pieces0

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

end Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursivePiCcsTranscript
