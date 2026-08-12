import Mathlib.Data.Multiset.MapFold
import Nightstream.Protocol.NebulaV2.Fingerprint

/-!
Contract: concrete ideal-verifier fingerprint check for Nebula V2.

Assurance tier: model-level.

Owns bounded lifting of the two memory multisets, evaluation of the exact
two-variable fingerprint polynomial at two challenge pairs, and the reduction
from an accepted check to exact balance or an explicit nonzero-polynomial
evaluation failure.

Does not own challenge unpredictability, Fiat-Shamir, Poseidon2, circuit rows,
or a probability bound.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.NebulaV2.IdealFingerprint

open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.Fingerprint
open Nightstream.Protocol.NebulaV2.Memory

def leftRecords
    (initial : Snapshot) (accesses : List Access) : Multiset MemTuple :=
  initial.tuples + (writeTuples accesses : Multiset MemTuple)

def rightRecords
    (accesses : List Access) (final : Snapshot) : Multiset MemTuple :=
  (readTuples accesses : Multiset MemTuple) + final.tuples

/-- Range checks that the ideal verifier must obtain before it interprets
record coordinates in the challenge field. -/
structure RecordBounds
    (initial : Snapshot) (accesses : List Access) (final : Snapshot) : Prop where
  left : ∀ entry ∈ leftRecords initial accesses, TupleInRange entry
  right : ∀ entry ∈ rightRecords accesses final, TupleInRange entry

private theorem access_globalIndex_lt_scannedCells
    {access : Access} (wellFormed : access.WellFormed) :
    globalIndex access.space access.address < scannedCells := by
  have addressBound := wellFormed.addressInRange
  cases spaceExact : access.space with
  | rom =>
      have addressBound' : access.address < romCells := by
        simpa [spaceExact, MemorySpace.capacity] using addressBound
      simp only [globalIndex, scannedCells]
      omega
  | ram =>
      have addressBound' : access.address < ramCells := by
        simpa [spaceExact, MemorySpace.capacity] using addressBound
      simp only [globalIndex, scannedCells]
      omega

theorem accessReadTupleInRange
    {access : Access} {timestamp : Nat}
    (valid : access.ValidAt timestamp) :
    TupleInRange access.read := by
  refine ⟨valid.readBeforeWrite.trans valid.timestampOutRange, ?_,
    valid.wellFormed.readValueInRange⟩
  rw [valid.wellFormed.readIndex]
  exact access_globalIndex_lt_scannedCells valid.wellFormed

theorem accessWriteTupleInRange
    {access : Access} {timestamp : Nat}
    (valid : access.ValidAt timestamp) :
    TupleInRange access.write := by
  refine ⟨?_, ?_, valid.wellFormed.writeValueInRange⟩
  · rw [valid.writeTimestamp]
    exact valid.timestampOutRange
  · rw [valid.wellFormed.writeIndex]
    exact access_globalIndex_lt_scannedCells valid.wellFormed

theorem orderedReadTupleInRange
    {timestampIn timestampOut : Nat} {accesses : List Access}
    (ordered : Ordered timestampIn accesses timestampOut)
    {entry : MemTuple}
    (member : entry ∈ (Memory.readTuples accesses : Multiset MemTuple)) :
    TupleInRange entry := by
  induction ordered generalizing entry with
  | nil => simp [Memory.readTuples] at member
  | @cons timestampIn timestampOut access rest valid tail
      inductionHypothesis =>
      have split : entry = access.read ∨
          entry ∈ (Memory.readTuples rest : Multiset MemTuple) := by
        simpa [Memory.readTuples] using member
      rcases split with headEqual | tailMember
      · rw [headEqual]
        exact accessReadTupleInRange valid
      · exact inductionHypothesis tailMember

theorem orderedWriteTupleInRange
    {timestampIn timestampOut : Nat} {accesses : List Access}
    (ordered : Ordered timestampIn accesses timestampOut)
    {entry : MemTuple}
    (member : entry ∈ (Memory.writeTuples accesses : Multiset MemTuple)) :
    TupleInRange entry := by
  induction ordered generalizing entry with
  | nil => simp [Memory.writeTuples] at member
  | @cons timestampIn timestampOut access rest valid tail
      inductionHypothesis =>
      have split : entry = access.write ∨
          entry ∈ (Memory.writeTuples rest : Multiset MemTuple) := by
        simpa [Memory.writeTuples] using member
      rcases split with headEqual | tailMember
      · rw [headEqual]
        exact accessWriteTupleInRange valid
      · exact inductionHypothesis tailMember

theorem Snapshot.tupleInRangeOfValidAt
    {snapshot : Snapshot} {boundary : Nat}
    (valid : snapshot.ValidAt boundary)
    (boundaryInRange : boundary < timestampLimit)
    {entry : MemTuple} (member : entry ∈ snapshot.tuples) :
    TupleInRange entry := by
  have valueAndTimestamp := snapshot.tuple_mem_validAt valid member
  exact ⟨valueAndTimestamp.2.trans_lt boundaryInRange,
    snapshot.tuple_mem_has_bounded_index member,
    valueAndTimestamp.1⟩

/-- Snapshot validity and the strict operation schedule derive every range
condition used by the concrete fingerprint. No caller supplies bounded record
multisets. -/
theorem RecordBounds.ofValidAt
    {initial final : Snapshot} {accesses : List Access}
    {timestampIn timestampOut initialBoundary finalBoundary : Nat}
    (initialValid : initial.ValidAt initialBoundary)
    (initialBoundaryInRange : initialBoundary < timestampLimit)
    (finalValid : final.ValidAt finalBoundary)
    (finalBoundaryInRange : finalBoundary < timestampLimit)
    (ordered : Ordered timestampIn accesses timestampOut) :
    RecordBounds initial accesses final where
  left := by
    intro entry member
    rw [leftRecords, Multiset.mem_add] at member
    rcases member with initialMember | writeMember
    · exact Snapshot.tupleInRangeOfValidAt initialValid
        initialBoundaryInRange initialMember
    · exact orderedWriteTupleInRange ordered writeMember
  right := by
    intro entry member
    rw [rightRecords, Multiset.mem_add] at member
    rcases member with readMember | finalMember
    · exact orderedReadTupleInRange ordered readMember
    · exact Snapshot.tupleInRangeOfValidAt finalValid
        finalBoundaryInRange finalMember

/-- Attach the checked range proof to every multiset occurrence. This operation
preserves multiplicity. -/
noncomputable def boundedRecords
    (records : Multiset MemTuple)
    (bounds : ∀ entry ∈ records, TupleInRange entry) :
    Multiset BoundedTuple :=
  records.pmap (fun entry proof => ⟨entry, proof⟩) bounds

@[simp]
theorem boundedRecords_values
    (records : Multiset MemTuple)
    (bounds : ∀ entry ∈ records, TupleInRange entry) :
    (boundedRecords records bounds).map Subtype.val = records := by
  unfold boundedRecords
  rw [Multiset.map_pmap]
  simpa using
    (Multiset.pmap_eq_map TupleInRange (fun entry => entry) records bounds)

@[simp]
theorem boundedRecords_card
    (records : Multiset MemTuple)
    (bounds : ∀ entry ∈ records, TupleInRange entry) :
    (boundedRecords records bounds).card = records.card := by
  unfold boundedRecords
  simp

structure ChallengePair (ChallengeField : Type) where
  gamma1 : ChallengeField
  gamma2 : ChallengeField
deriving DecidableEq, Repr

def ChallengePair.point
    {ChallengeField : Type} (challenge : ChallengePair ChallengeField) :
    Fin 2 → ChallengeField :=
  Fin.cases challenge.gamma2 (fun _ => challenge.gamma1)

/-- Recover a challenge pair from the two polynomial coordinates. Coordinate
zero is `gamma2`; coordinate one is `gamma1`. -/
def ChallengePair.ofPoint
    {ChallengeField : Type} (point : Fin 2 → ChallengeField) :
    ChallengePair ChallengeField :=
  { gamma1 := point 1
    gamma2 := point 0 }

@[simp]
theorem ChallengePair.ofPoint_point
    {ChallengeField : Type} (challenge : ChallengePair ChallengeField) :
    ChallengePair.ofPoint challenge.point = challenge := by
  cases challenge
  rfl

@[simp]
theorem ChallengePair.point_ofPoint
    {ChallengeField : Type} (point : Fin 2 → ChallengeField) :
    (ChallengePair.ofPoint point).point = point := by
  funext coordinate
  fin_cases coordinate <;> rfl

/-- No challenge pairs are lost by the polynomial-coordinate representation. -/
def ChallengePair.pointEquiv
    {ChallengeField : Type} :
    ChallengePair ChallengeField ≃ (Fin 2 → ChallengeField) where
  toFun := ChallengePair.point
  invFun := ChallengePair.ofPoint
  left_inv := ChallengePair.ofPoint_point
  right_inv := ChallengePair.point_ofPoint

noncomputable def evaluate
    {ChallengeField : Type} [CommSemiring ChallengeField]
    (challenge : ChallengePair ChallengeField)
    (polynomial : MvPolynomial (Fin 2) ChallengeField) : ChallengeField :=
  polynomial.eval challenge.point

/-- The ideal check uses the actual typed memory records. No caller supplies a
separate claimed multiset or a claimed difference polynomial. -/
structure Check
    {ChallengeField : Type} [Field ChallengeField]
    (encode : Nat → ChallengeField)
    (initial : Snapshot) (accesses : List Access) (final : Snapshot) where
  bounds : RecordBounds initial accesses final
  challenges : Fin 2 → ChallengePair ChallengeField

namespace Check

variable {ChallengeField : Type} [Field ChallengeField]
variable {encode : Nat → ChallengeField}
variable {initial final : Snapshot} {accesses : List Access}

noncomputable def left
    (check : Check encode initial accesses final) : Multiset BoundedTuple :=
  boundedRecords (leftRecords initial accesses) check.bounds.left

noncomputable def right
    (check : Check encode initial accesses final) : Multiset BoundedTuple :=
  boundedRecords (rightRecords accesses final) check.bounds.right

noncomputable def polynomial
    (check : Check encode initial accesses final) :
    MvPolynomial (Fin 2) ChallengeField :=
  difference (packedCoordinate encode) (valueCoordinate encode)
    check.left check.right

def Accepts (check : Check encode initial accesses final) : Prop :=
  ∀ repetition,
    evaluate (check.challenges repetition) check.polynomial = 0

theorem left_values (check : Check encode initial accesses final) :
    check.left.map Subtype.val = leftRecords initial accesses := by
  exact boundedRecords_values _ _

theorem right_values (check : Check encode initial accesses final) :
    check.right.map Subtype.val = rightRecords accesses final := by
  exact boundedRecords_values _ _

theorem left_card (check : Check encode initial accesses final) :
    check.left.card = scannedCells + accesses.length := by
  unfold left leftRecords
  simp [Snapshot.tuples, Snapshot.tupleList_length, Memory.writeTuples]

theorem right_card (check : Check encode initial accesses final) :
    check.right.card = accesses.length + scannedCells := by
  unfold right rightRecords
  simp [Snapshot.tuples, Snapshot.tupleList_length, Memory.readTuples]

theorem degree_le_maxSegmentFactors
    (check : Check encode initial accesses final)
    (accessBound : accesses.length ≤ 63 * 1088) :
    check.polynomial.totalDegree ≤ maxSegmentFactors := by
  unfold polynomial
  apply difference_totalDegree_le
  · rw [check.left_card]
    unfold maxSegmentFactors
    omega
  · rw [check.right_card]
    unfold maxSegmentFactors
    omega

theorem bounded_eq_of_balance
    (check : Check encode initial accesses final)
    (balance : Balanced initial.tuples accesses final.tuples) :
    check.left = check.right := by
  apply Multiset.map_injective Subtype.val_injective
  rw [check.left_values, check.right_values]
  exact balance

theorem balance_of_bounded_eq
    (check : Check encode initial accesses final)
    (equal : check.left = check.right) :
    Balanced initial.tuples accesses final.tuples := by
  have mapped := congrArg (Multiset.map Subtype.val) equal
  simpa only [check.left_values, check.right_values] using mapped

theorem accepts_of_balance
    (check : Check encode initial accesses final)
    (balance : Balanced initial.tuples accesses final.tuples) :
    check.Accepts := by
  have equal := check.bounded_eq_of_balance balance
  intro repetition
  simp [polynomial, equal, difference, evaluate]

end Check

/-- This is the exact public-coin algebraic failure that remains when unequal
typed multisets pass both concrete evaluations. The nonzero field polynomial
is recorded, so the later probability proof cannot price a zero polynomial. -/
structure EvaluationFailure
    {ChallengeField : Type} [Field ChallengeField]
    {encode : Nat → ChallengeField}
    {initial final : Snapshot} {accesses : List Access}
    (check : Check encode initial accesses final) : Prop where
  unbalanced : ¬ Balanced initial.tuples accesses final.tuples
  polynomialNonzero : check.polynomial ≠ 0
  accepted : check.Accepts

/-- An accepted ideal fingerprint check gives exact multiset balance unless a
named nonzero-polynomial evaluation failure occurred. -/
theorem balance_or_evaluationFailure
    {ChallengeField : Type} [Field ChallengeField]
    {encode : Nat → ChallengeField}
    (encodeInjective : InjectiveBelowGoldilocks encode)
    {initial final : Snapshot} {accesses : List Access}
    (check : Check encode initial accesses final)
    (accepted : check.Accepts) :
    Balanced initial.tuples accesses final.tuples ∨
      EvaluationFailure check := by
  by_cases balance : Balanced initial.tuples accesses final.tuples
  · exact Or.inl balance
  · right
    have unequal : check.left ≠ check.right := by
      intro equal
      exact balance (check.balance_of_bounded_eq equal)
    exact
      { unbalanced := balance
        polynomialNonzero :=
          boundedDifference_ne_zero encode encodeInjective unequal
        accepted := accepted }

end Nightstream.Protocol.NebulaV2.IdealFingerprint
