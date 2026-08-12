import Nightstream.Protocol.NebulaV2.IdealFingerprint

/-!
Contract: exact two-repetition fingerprint product state for Nebula V2.

Assurance tier: model-level.

Owns the four running products for each of the two fixed repetitions, their
canonical all-one opening value, the exact products determined by typed memory
records and transcript challenges, and the equivalence between the closing
product equation and the ideal polynomial check.

Does not own generated-row updates, challenge unpredictability, Fiat-Shamir,
or a probability bound.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.NebulaV2.ProductState

open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.Fingerprint
open Nightstream.Protocol.NebulaV2.IdealFingerprint
open Nightstream.Protocol.NebulaV2.Memory

/-- The four authority-bearing accumulators in one fingerprint repetition. -/
structure Four (ChallengeField : Type) where
  initialSnapshot : ChallengeField
  writes : ChallengeField
  reads : ChallengeField
  finalSnapshot : ChallengeField
deriving DecidableEq, Repr

@[ext]
theorem Four.ext
    {ChallengeField : Type} {left right : Four ChallengeField}
    (initialSnapshot : left.initialSnapshot = right.initialSnapshot)
    (writes : left.writes = right.writes)
    (reads : left.reads = right.reads)
    (finalSnapshot : left.finalSnapshot = right.finalSnapshot) :
    left = right := by
  cases left
  cases right
  simp_all

/-- V2 fixes exactly two independent repetitions. -/
abbrev State (ChallengeField : Type) := Fin 2 → Four ChallengeField

/-- The complete active challenge state contains exactly two challenge pairs. -/
abbrev Challenges (ChallengeField : Type) :=
  Fin 2 → ChallengePair ChallengeField

/-- The only valid product state before any committed record is consumed. -/
def one {ChallengeField : Type} [One ChallengeField] :
    State ChallengeField :=
  fun _ =>
    { initialSnapshot := 1
      writes := 1
      reads := 1
      finalSnapshot := 1 }

def Four.Balanced
    {ChallengeField : Type} [Mul ChallengeField]
    (products : Four ChallengeField) : Prop :=
  products.initialSnapshot * products.writes =
    products.reads * products.finalSnapshot

def Balanced
    {ChallengeField : Type} [Mul ChallengeField]
    (products : State ChallengeField) : Prop :=
  ∀ repetition, (products repetition).Balanced

/-- Evaluation of one typed memory record at one V2 challenge pair. The
variable order is `(gamma2, gamma1)`, exactly as in `Fingerprint.factor`. -/
def recordFactor
    {ChallengeField : Type} [Ring ChallengeField]
    (encode : Nat → ChallengeField)
    (challenge : ChallengePair ChallengeField)
    (entry : MemTuple) : ChallengeField :=
  challenge.gamma2 -
    (encode (packedNat entry) + encode entry.value * challenge.gamma1)

def recordsProduct
    {ChallengeField : Type} [CommRing ChallengeField]
    (encode : Nat → ChallengeField)
    (challenge : ChallengePair ChallengeField)
    (records : Multiset MemTuple) : ChallengeField :=
  (records.map (recordFactor encode challenge)).prod

/-- Exact typed records consumed by one checked memory step. Physical port
holes are absent here only after the fixed-position port decoder has extracted
the active records. Multiplicity and order within the operations lane remain
owned by the application-port refinement. -/
structure Chunk where
  initialSnapshot : Multiset MemTuple
  writes : Multiset MemTuple
  reads : Multiset MemTuple
  finalSnapshot : Multiset MemTuple
deriving DecidableEq

/-- One generated checked step must implement this update for both fixed
fingerprint repetitions and all four products. -/
def update
    {ChallengeField : Type} [Field ChallengeField]
    (encode : Nat → ChallengeField)
    (challenges : Challenges ChallengeField)
    (before : State ChallengeField)
    (chunk : Chunk) : State ChallengeField :=
  fun repetition =>
    let challenge := challenges repetition
    { initialSnapshot :=
        (before repetition).initialSnapshot *
          recordsProduct encode challenge chunk.initialSnapshot
      writes :=
        (before repetition).writes *
          recordsProduct encode challenge chunk.writes
      reads :=
        (before repetition).reads *
          recordsProduct encode challenge chunk.reads
      finalSnapshot :=
        (before repetition).finalSnapshot *
          recordsProduct encode challenge chunk.finalSnapshot }

/-- Recursive product accumulation in exact checked-step order. -/
def accumulate
    {ChallengeField : Type} [Field ChallengeField]
    (encode : Nat → ChallengeField)
    (challenges : Challenges ChallengeField) :
    State ChallengeField → List Chunk → State ChallengeField
  | state, [] => state
  | state, chunk :: rest =>
      accumulate encode challenges (update encode challenges state chunk) rest

/-- The checked-step chunks cover every semantic record exactly once. These
equalities preserve multiset multiplicity. -/
structure Covers
    (initial : Snapshot) (accesses : List Access) (final : Snapshot)
    (chunks : List Chunk) : Prop where
  initialSnapshot :
    (chunks.map Chunk.initialSnapshot).sum = initial.tuples
  writes :
    (chunks.map Chunk.writes).sum =
      (Memory.writeTuples accesses : Multiset MemTuple)
  reads :
    (chunks.map Chunk.reads).sum =
      (Memory.readTuples accesses : Multiset MemTuple)
  finalSnapshot :
    (chunks.map Chunk.finalSnapshot).sum = final.tuples

theorem recordsProduct_add
    {ChallengeField : Type} [Field ChallengeField]
    (encode : Nat → ChallengeField)
    (challenge : ChallengePair ChallengeField)
    (left right : Multiset MemTuple) :
    recordsProduct encode challenge (left + right) =
      recordsProduct encode challenge left *
        recordsProduct encode challenge right := by
  simp [recordsProduct, Multiset.map_add, Multiset.prod_add]

theorem accumulate_initialSnapshot
    {ChallengeField : Type} [Field ChallengeField]
    (encode : Nat → ChallengeField)
    (challenges : Challenges ChallengeField)
    (start : State ChallengeField) (chunks : List Chunk)
    (repetition : Fin 2) :
    (accumulate encode challenges start chunks repetition).initialSnapshot =
      (start repetition).initialSnapshot *
        recordsProduct encode (challenges repetition)
          (chunks.map Chunk.initialSnapshot).sum := by
  induction chunks generalizing start with
  | nil => simp [accumulate, recordsProduct]
  | cons chunk rest inductionHypothesis =>
      rw [accumulate, inductionHypothesis]
      simp only [List.map_cons, List.sum_cons]
      rw [recordsProduct_add]
      simp [update, mul_assoc]

theorem accumulate_writes
    {ChallengeField : Type} [Field ChallengeField]
    (encode : Nat → ChallengeField)
    (challenges : Challenges ChallengeField)
    (start : State ChallengeField) (chunks : List Chunk)
    (repetition : Fin 2) :
    (accumulate encode challenges start chunks repetition).writes =
      (start repetition).writes *
        recordsProduct encode (challenges repetition)
          (chunks.map Chunk.writes).sum := by
  induction chunks generalizing start with
  | nil => simp [accumulate, recordsProduct]
  | cons chunk rest inductionHypothesis =>
      rw [accumulate, inductionHypothesis]
      simp only [List.map_cons, List.sum_cons]
      rw [recordsProduct_add]
      simp [update, mul_assoc]

theorem accumulate_reads
    {ChallengeField : Type} [Field ChallengeField]
    (encode : Nat → ChallengeField)
    (challenges : Challenges ChallengeField)
    (start : State ChallengeField) (chunks : List Chunk)
    (repetition : Fin 2) :
    (accumulate encode challenges start chunks repetition).reads =
      (start repetition).reads *
        recordsProduct encode (challenges repetition)
          (chunks.map Chunk.reads).sum := by
  induction chunks generalizing start with
  | nil => simp [accumulate, recordsProduct]
  | cons chunk rest inductionHypothesis =>
      rw [accumulate, inductionHypothesis]
      simp only [List.map_cons, List.sum_cons]
      rw [recordsProduct_add]
      simp [update, mul_assoc]

theorem accumulate_finalSnapshot
    {ChallengeField : Type} [Field ChallengeField]
    (encode : Nat → ChallengeField)
    (challenges : Challenges ChallengeField)
    (start : State ChallengeField) (chunks : List Chunk)
    (repetition : Fin 2) :
    (accumulate encode challenges start chunks repetition).finalSnapshot =
      (start repetition).finalSnapshot *
        recordsProduct encode (challenges repetition)
          (chunks.map Chunk.finalSnapshot).sum := by
  induction chunks generalizing start with
  | nil => simp [accumulate, recordsProduct]
  | cons chunk rest inductionHypothesis =>
      rw [accumulate, inductionHypothesis]
      simp only [List.map_cons, List.sum_cons]
      rw [recordsProduct_add]
      simp [update, mul_assoc]

/-- The final four products are a deterministic function of the exact typed
records and the two transcript-derived challenge pairs. -/
def expected
    {ChallengeField : Type} [Field ChallengeField]
    {encode : Nat → ChallengeField}
    {initial final : Snapshot} {accesses : List Access}
    (check : Check encode initial accesses final) : State ChallengeField :=
  fun repetition =>
    let challenge := check.challenges repetition
    { initialSnapshot := recordsProduct encode challenge initial.tuples
      writes := recordsProduct encode challenge
        (writeTuples accesses : Multiset MemTuple)
      reads := recordsProduct encode challenge
        (readTuples accesses : Multiset MemTuple)
      finalSnapshot := recordsProduct encode challenge final.tuples }

/-- Exact chunk coverage makes the product state computed by all checked-step
updates equal the products of the complete semantic multisets. -/
theorem accumulate_one_eq_expected
    {ChallengeField : Type} [Field ChallengeField]
    {encode : Nat → ChallengeField}
    {initial final : Snapshot} {accesses : List Access}
    (check : Check encode initial accesses final)
    {chunks : List Chunk}
    (coverage : Covers initial accesses final chunks) :
    accumulate encode check.challenges one chunks = expected check := by
  funext repetition
  apply Four.ext
  · rw [accumulate_initialSnapshot, coverage.initialSnapshot]
    simp [one, expected]
  · rw [accumulate_writes, coverage.writes]
    simp [one, expected]
  · rw [accumulate_reads, coverage.reads]
    simp [one, expected]
  · rw [accumulate_finalSnapshot, coverage.finalSnapshot]
    simp [one, expected]

@[simp]
theorem evaluate_factor
    {ChallengeField : Type} [Field ChallengeField]
    (encode : Nat → ChallengeField)
    (challenge : ChallengePair ChallengeField)
    (entry : BoundedTuple) :
    evaluate challenge
        (Fingerprint.factor (packedCoordinate encode)
          (valueCoordinate encode) entry) =
      recordFactor encode challenge entry.1 := by
  have pointZero : challenge.point (0 : Fin 2) = challenge.gamma2 := rfl
  have pointOne : challenge.point (1 : Fin 2) = challenge.gamma1 := by
    rw [show (1 : Fin 2) = Fin.succ (0 : Fin 1) by decide]
    rfl
  simp only [evaluate, Fingerprint.factor, map_sub, MvPolynomial.eval_X,
    map_add, MvPolynomial.eval_C, map_mul, packedCoordinate,
    valueCoordinate, recordFactor, pointZero, pointOne]

@[simp]
theorem evaluate_mul
    {ChallengeField : Type} [Field ChallengeField]
    (challenge : ChallengePair ChallengeField)
    (left right : MvPolynomial (Fin 2) ChallengeField) :
    evaluate challenge (left * right) =
      evaluate challenge left * evaluate challenge right := by
  simp [evaluate]

@[simp]
theorem evaluate_sub
    {ChallengeField : Type} [Field ChallengeField]
    (challenge : ChallengePair ChallengeField)
    (left right : MvPolynomial (Fin 2) ChallengeField) :
    evaluate challenge (left - right) =
      evaluate challenge left - evaluate challenge right := by
  simp [evaluate]

theorem evaluate_product
    {ChallengeField : Type} [Field ChallengeField]
    (encode : Nat → ChallengeField)
    (challenge : ChallengePair ChallengeField)
    (records : Multiset BoundedTuple) :
    evaluate challenge
        (Fingerprint.product (packedCoordinate encode)
          (valueCoordinate encode) records) =
      recordsProduct encode challenge (records.map Subtype.val) := by
  induction records using Multiset.induction_on with
  | empty => simp [evaluate, Fingerprint.product, recordsProduct]
  | cons entry rest inductionHypothesis =>
      simp only [Fingerprint.product, Multiset.map_cons,
        Multiset.prod_cons]
      rw [evaluate_mul, evaluate_factor]
      unfold Fingerprint.product at inductionHypothesis
      simp only [recordsProduct, Multiset.map_cons, Multiset.prod_cons,
        Multiset.map_map]
      rw [inductionHypothesis]
      simp [recordsProduct, Multiset.map_map]

theorem accepts_iff_expected_balanced
    {ChallengeField : Type} [Field ChallengeField]
    {encode : Nat → ChallengeField}
    {initial final : Snapshot} {accesses : List Access}
    (check : Check encode initial accesses final) :
    check.Accepts ↔ Balanced (expected check) := by
  constructor
  · intro accepted repetition
    have atRepetition := accepted repetition
    unfold Check.polynomial Fingerprint.difference at atRepetition
    rw [evaluate_sub] at atRepetition
    rw [evaluate_product, evaluate_product] at atRepetition
    rw [check.left_values, check.right_values] at atRepetition
    simp only [sub_eq_zero] at atRepetition
    simpa [expected, Four.Balanced, leftRecords, rightRecords,
      recordsProduct, Multiset.map_add, Multiset.prod_add] using atRepetition
  · intro balanced repetition
    have atRepetition := balanced repetition
    unfold Check.polynomial Fingerprint.difference
    rw [evaluate_sub]
    rw [evaluate_product, evaluate_product]
    rw [check.left_values, check.right_values]
    simp only [sub_eq_zero]
    simpa [expected, Four.Balanced, leftRecords, rightRecords,
      recordsProduct, Multiset.map_add, Multiset.prod_add] using atRepetition

theorem balanced_expected_of_memory_balance
    {ChallengeField : Type} [Field ChallengeField]
    {encode : Nat → ChallengeField}
    {initial final : Snapshot} {accesses : List Access}
    (check : Check encode initial accesses final)
    (balance : Memory.Balanced initial.tuples accesses final.tuples) :
    Balanced (expected check) :=
  (accepts_iff_expected_balanced check).mp
    (check.accepts_of_balance balance)

end Nightstream.Protocol.NebulaV2.ProductState
