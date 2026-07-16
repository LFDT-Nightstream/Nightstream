import Nightstream.Implementation.R1CS.Core.TranscriptCertificate

/-! Generated exact terminal transcript-initialization owner. Hashes below are drift metadata only. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalTranscriptArtifact

open Nightstream.Implementation.R1CS

set_option maxRecDepth 1048576

def rangeSha256 : String := "a2bf4d06bf0a94252cd9e679b4d8be05931d6856eed7db62c0f64c9201bc1bcb"
def rowStart : Nat := 1124097
def rowEnd : Nat := 1124105
def rowCount : Nat := 8


def constantPins : List (Nat × Nat) := [(1132300, 13350748695060858), (1132301, 12662), (1132302, 1834773423675177288), (1132303, 13672990098899735520), (1132304, 17674766951204602997), (1132305, 17089493294232106953), (1132306, 14279209731735639343), (1132307, 1234493824114689751)]



def calls : List Poseidon2Call.Call := []

def trace : TranscriptCertificate.Trace := ⟨constantPins, calls⟩

def schedule : List TranscriptCertificate.PieceRef :=
  [.pin 0,
   .pin 1,
   .pin 2,
   .pin 3,
   .pin 4,
   .pin 5,
   .pin 6,
   .pin 7]

def rowPieces : List (List Row) :=
  schedule.map fun piece => piece.rows trace

def ownerRows : List Row := trace.orderedRows schedule

theorem ownerRows_length : ownerRows.length = rowCount := by native_decide

def pinIndicesBoundedCheck : Bool :=
  schedule.all fun piece =>
    match piece with
    | .pin index => decide (index < trace.pins.length)
    | .call _ => true

def callIndicesBoundedCheck : Bool :=
  schedule.all fun piece =>
    match piece with
    | .pin _ => true
    | .call index => decide (index < trace.calls.length)

def everyPinScheduledCheck : Bool :=
  (List.range trace.pins.length).all fun index =>
    decide (.pin index ∈ schedule)

def everyCallScheduledCheck : Bool :=
  (List.range trace.calls.length).all fun index =>
    decide (.call index ∈ schedule)

theorem pinIndicesBounded_checked : pinIndicesBoundedCheck = true := by native_decide
theorem callIndicesBounded_checked : callIndicesBoundedCheck = true := by native_decide
theorem everyPinScheduled_checked : everyPinScheduledCheck = true := by native_decide
theorem everyCallScheduled_checked : everyCallScheduledCheck = true := by native_decide

theorem traceValid : trace.OrderedValid schedule ownerRows where
  pinIndicesBounded := by
    intro index member
    exact of_decide_eq_true
      ((List.all_eq_true.mp pinIndicesBounded_checked) (.pin index) member)
  callIndicesBounded := by
    intro index member
    exact of_decide_eq_true
      ((List.all_eq_true.mp callIndicesBounded_checked) (.call index) member)
  everyPinScheduled := by
    intro index indexLt
    exact of_decide_eq_true
      ((List.all_eq_true.mp everyPinScheduled_checked) index
        (List.mem_range.mpr indexLt))
  everyCallScheduled := by
    intro index indexLt
    exact of_decide_eq_true
      ((List.all_eq_true.mp everyCallScheduled_checked) index
        (List.mem_range.mpr indexLt))
  pinValuesCanonical := by native_decide
  exactRows := rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalTranscriptArtifact
