import Nightstream.Protocol.Nebula.IdealFingerprint

set_option autoImplicit false

namespace tests.NebulaIdealFingerprint

open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.Fingerprint
open Nightstream.Protocol.Nebula.IdealFingerprint
open Nightstream.Protocol.Nebula.Memory

def zeroImage : Fin scannedCells → Nat := fun _ => 0
def zeroSnapshot : Snapshot := Snapshot.ofImage zeroImage

def zeroImageInRange : Snapshot.ImageInRange zeroImage := by
  intro _index
  change 0 < valueLimit
  decide

def emptyBounds : RecordBounds zeroSnapshot [] zeroSnapshot where
  left := by
    intro entry member
    rcases Multiset.mem_add.mp member with member | member
    · have valid : zeroSnapshot.ValidAt 0 := by
        exact Snapshot.ofImage_validAt_zero zeroImageInRange
      have tupleValid := zeroSnapshot.tuple_mem_validAt valid member
      exact
        ⟨by
          have positive : 0 < timestampLimit := by decide
          omega,
          zeroSnapshot.tuple_mem_has_bounded_index member,
          tupleValid.1⟩
    · simp [writeTuples] at member
  right := by
    intro entry member
    rcases Multiset.mem_add.mp member with member | member
    · simp [readTuples] at member
    · have valid : zeroSnapshot.ValidAt 0 := by
        exact Snapshot.ofImage_validAt_zero zeroImageInRange
      have tupleValid := zeroSnapshot.tuple_mem_validAt valid member
      exact
        ⟨by
          have positive : 0 < timestampLimit := by decide
          omega,
          zeroSnapshot.tuple_mem_has_bounded_index member,
          tupleValid.1⟩

def rationalEncoding (value : Nat) : ℚ := value

theorem rationalEncoding_injective :
    InjectiveBelowGoldilocks rationalEncoding := by
  intro left right _ _ equal
  change (left : ℚ) = (right : ℚ) at equal
  exact_mod_cast equal

def check : Check rationalEncoding zeroSnapshot [] zeroSnapshot where
  bounds := emptyBounds
  challenges := fun repetition =>
    { gamma1 := repetition.val + 1
      gamma2 := repetition.val + 3 }

theorem empty_balance :
    Balanced zeroSnapshot.tuples [] zeroSnapshot.tuples := by
  simp [Balanced, readTuples, writeTuples]

theorem honest_empty_check_accepts : check.Accepts :=
  check.accepts_of_balance empty_balance

theorem honest_empty_check_recovers_balance :
    Balanced zeroSnapshot.tuples [] zeroSnapshot.tuples := by
  rcases balance_or_evaluationFailure rationalEncoding_injective check
      honest_empty_check_accepts with balance | failure
  · exact balance
  · exact False.elim (failure.unbalanced empty_balance)

end tests.NebulaIdealFingerprint
