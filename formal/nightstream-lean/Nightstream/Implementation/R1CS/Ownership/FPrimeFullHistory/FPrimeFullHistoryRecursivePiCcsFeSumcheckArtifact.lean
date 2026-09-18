import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryRecursivePiCcsFeSumcheckPieces0

/-! Exact ordered row certificate for the recursive Pi_CCS FE SumCheck owner. Hash is drift metadata only. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursivePiCcsFeSumcheck

open Nightstream.Implementation.R1CS.OwnerCertificate

set_option maxRecDepth 1048576

def rangeSha256 : String := "5e58c0596f4c8f7b71aa577d5bb218f598f44fc4b70193a6a278d18ab48e23b0"
def rowStart : Nat := 225922
def rowEnd : Nat := 240553
def rowCount : Nat := 14631

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

end Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursivePiCcsFeSumcheck
