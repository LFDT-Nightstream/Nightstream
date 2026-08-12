import Nightstream.Implementation.NebulaV2.ProductPoseidon2
import Nightstream.Implementation.R1CS.Canonical.SymbolicDuplex
import Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexCount

/-!
Contract: exact indexed Poseidon2 rows for the V2 full-field PiRLC sampler.

Owns all 15 x 54 x 3 independently framed candidate calls. Each call starts
from the same complete post-PiCCS state with the profile-fixed full cursor,
absorbs the exact fixed-width two-field candidate frame, applies the
challenge gate, and exposes lane zero as the candidate. Physical call windows
are disjoint.

The row family is indexed instead of materialized as one 14-million-row Lean
list. `RowsHold` means that every indexed physical row list is satisfied.
The exact aggregate count is still a theorem.

Does not own candidate classification, modulo-five decoding, first-accepted
selection, PiRLC algebra, honest witnesses, cryptographic distribution, Rust,
or the surrounding NIFS result.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.NebulaV2.ProductPiRlcTranscriptRows

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.SuperNeo.Folding.Nifs

def scalarCount : Nat := 15
def coefficientCount : Nat := 54
def attemptCount : Nat := 3

theorem scalarCount_eq : scalarCount = 15 := by rfl
theorem coefficientCount_eq : coefficientCount = 54 := by rfl
theorem attemptCount_eq : attemptCount = 3 := by rfl

theorem scalarCount_profile : scalarCount = PaperProfile.arity.total := by rfl
theorem coefficientCount_profile :
    coefficientCount = ProductPoseidon2.samplerCoefficientCount := by rfl
theorem attemptCount_profile :
    attemptCount = ProductPoseidon2.samplerAttemptCount := by rfl

/-- One exact domain-expansion coordinate. -/
structure CandidateIndex where
  source : Fin scalarCount
  coefficient : Fin coefficientCount
  attempt : Fin attemptCount
deriving DecidableEq

/-- Source-major, coefficient-major, attempt-minor physical order. -/
def CandidateIndex.flat (index : CandidateIndex) : Nat :=
  (index.source.val * coefficientCount + index.coefficient.val) *
      attemptCount + index.attempt.val

def candidateCount : Nat := scalarCount * coefficientCount * attemptCount

theorem candidateCount_eq : candidateCount = 2430 := by decide

theorem CandidateIndex.flat_lt (index : CandidateIndex) :
    index.flat < candidateCount := by
  have sourceLt := index.source.isLt
  have coefficientLt := index.coefficient.isLt
  have attemptLt := index.attempt.isLt
  norm_num [CandidateIndex.flat, candidateCount, scalarCount,
    coefficientCount, attemptCount] at *
  omega

theorem CandidateIndex.flat_injective : Function.Injective CandidateIndex.flat := by
  intro left right equal
  have attemptEqual : left.attempt.val = right.attempt.val := by
    have modulo := congrArg (fun value => value % attemptCount) equal
    simpa [CandidateIndex.flat, Nat.add_mod, Nat.mul_mod,
      Nat.mod_eq_of_lt left.attempt.isLt,
      Nat.mod_eq_of_lt right.attempt.isLt] using modulo
  have prefixEqual :
      left.source.val * coefficientCount + left.coefficient.val =
        right.source.val * coefficientCount + right.coefficient.val := by
    norm_num [CandidateIndex.flat, attemptCount] at equal
    omega
  have coefficientEqual :
      left.coefficient.val = right.coefficient.val := by
    have modulo := congrArg (fun value => value % coefficientCount) prefixEqual
    simpa [Nat.add_mod, Nat.mul_mod,
      Nat.mod_eq_of_lt left.coefficient.isLt,
      Nat.mod_eq_of_lt right.coefficient.isLt] using modulo
  have sourceEqual : left.source.val = right.source.val := by
    norm_num [coefficientCount] at prefixEqual
    omega
  have sourceFin : left.source = right.source := Fin.ext sourceEqual
  have coefficientFin : left.coefficient = right.coefficient :=
    Fin.ext coefficientEqual
  have attemptFin : left.attempt = right.attempt := Fin.ext attemptEqual
  rcases left with ⟨leftSource, leftCoefficient, leftAttempt⟩
  rcases right with ⟨rightSource, rightCoefficient, rightAttempt⟩
  simp only at sourceFin coefficientFin attemptFin
  subst rightSource
  subst rightCoefficient
  subst rightAttempt
  rfl

/-- Physical values shared by all candidate calls. The absorbed cursor is
fixed to four by the selected complete PiCCS output serialization. -/
structure Input where
  postPiCcsLanes : Poseidon2Core.State
  transcriptBase : Nat

def word (value : Nat) : LinComb := [(0, value % goldilocksP)]

def candidateFields (index : CandidateIndex) : List LinComb :=
  (ProductPoseidon2.candidateFields
    (Fin.cast scalarCount_profile index.source)
    (Fin.cast coefficientCount_profile index.coefficient)
    (Fin.cast attemptCount_profile index.attempt)).map word

theorem candidateFields_length (index : CandidateIndex) :
    (candidateFields index).length = 2 := by
  unfold candidateFields ProductPoseidon2.candidateFields
  simp only [List.length_map]
  rfl

/-- The verifier-key-bound two-field frame gives every candidate fork a
different absorbed field list. -/
theorem candidateFields_injective : Function.Injective candidateFields := by
  intro left right fieldsEqual
  have leftLt : left.flat < goldilocksP := by
    have bounded := left.flat_lt
    norm_num [candidateCount, scalarCount, coefficientCount, attemptCount,
      goldilocksP] at bounded ⊢
    omega
  have rightLt : right.flat < goldilocksP := by
    have bounded := right.flat_lt
    norm_num [candidateCount, scalarCount, coefficientCount, attemptCount,
      goldilocksP] at bounded ⊢
    omega
  have flatEqual : left.flat = right.flat := by
    unfold candidateFields ProductPoseidon2.candidateFields
      Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityPoseidon2.candidateFields
      at fieldsEqual
    simp only [List.map_cons, List.map_nil, List.cons.injEq, and_true] at fieldsEqual
    have secondEqual := fieldsEqual.2
    simp only [word, List.cons.injEq, Prod.mk.injEq, true_and] at secondEqual
    have secondValueEqual := secondEqual.1
    have leftFlatDef :
        Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityPoseidon2.candidateFlat
            (Fin.cast scalarCount_profile left.source)
            (Fin.cast coefficientCount_profile left.coefficient)
            (Fin.cast attemptCount_profile left.attempt) = left.flat := by
      rfl
    have rightFlatDef :
        Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityPoseidon2.candidateFlat
            (Fin.cast scalarCount_profile right.source)
            (Fin.cast coefficientCount_profile right.coefficient)
            (Fin.cast attemptCount_profile right.attempt) = right.flat := by
      rfl
    simp only [
      Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityPoseidon2.word]
      at secondValueEqual
    rw [leftFlatDef, rightFlatDef] at secondValueEqual
    simp only [Nightstream.SuperNeo.Concrete.goldilocksModulus, goldilocksP]
      at secondValueEqual
    have leftLtModulus : left.flat < 18446744069414584321 := by
      simpa [goldilocksP] using leftLt
    have rightLtModulus : right.flat < 18446744069414584321 := by
      simpa [goldilocksP] using rightLt
    simpa only [Nat.mod_eq_of_lt leftLtModulus,
      Nat.mod_eq_of_lt rightLtModulus] using secondValueEqual
  exact CandidateIndex.flat_injective flatEqual

/-- Forget the PiCCS builder history but retain its exact physical lanes and
the selected full cursor. Candidate calls are independent forks of this state. -/
def start (input : Input) : SymbolicDuplex.Builder :=
  SymbolicDuplex.start input.postPiCcsLanes 4

@[simp] theorem start_lanes (input : Input) :
    (start input).lanes = input.postPiCcsLanes := rfl

@[simp] theorem start_absorbed (input : Input) :
    (start input).absorbed = 4 := rfl

def permutationsPerCandidate : Nat := 2
def rowsPerCandidate : Nat := permutationsPerCandidate * SymbolicDuplex.stride

theorem rowsPerCandidate_eq : rowsPerCandidate = 704 := by decide

/-- One disjoint call-local column window. -/
def candidateBase (input : Input) (index : CandidateIndex) : Nat :=
  input.transcriptBase + index.flat * rowsPerCandidate

def builder (input : Input) (index : CandidateIndex) : SymbolicDuplex.Builder :=
  let base := candidateBase input index
  SymbolicDuplex.gate base
    (SymbolicDuplex.absorbMany base (candidateFields index) (start input))

def candidate (input : Input) (index : CandidateIndex) : LinComb :=
  (builder input index).lanes ⟨0, by decide⟩

theorem builder_absorbed (input : Input) (index : CandidateIndex) :
    (builder input index).absorbed = 0 := by
  rfl

theorem builder_entries_length (input : Input) (index : CandidateIndex) :
    (builder input index).entries.length = permutationsPerCandidate := by
  let base := candidateBase input index
  let initial := start input
  let absorbed :=
    SymbolicDuplex.absorbMany base (candidateFields index) initial
  have initialCount :
      SymbolicDuplexCount.ofBuilder initial = ⟨0, 4⟩ := by
    rfl
  have absorbedCount :=
    SymbolicDuplexCount.ofBuilder_absorbMany base
      (candidateFields index) initial
  rw [candidateFields_length, initialCount] at absorbedCount
  have count2 :
      SymbolicDuplexCount.absorbMany 2 ⟨0, 4⟩ = ⟨1, 2⟩ := by
    rw [SymbolicDuplexCount.absorbMany_eq_fast]
    rfl
  rw [count2] at absorbedCount
  have finalCount := SymbolicDuplexCount.ofBuilder_gate base absorbed
  rw [absorbedCount] at finalCount
  have exactFinal :
      SymbolicDuplexCount.ofBuilder (SymbolicDuplex.gate base absorbed) =
        ⟨2, 0⟩ := by
    exact finalCount
  have entries := congrArg SymbolicDuplexCount.Control.entries exactFinal
  simpa only [SymbolicDuplexCount.ofBuilder, permutationsPerCandidate,
    builder, base, initial, absorbed] using entries

def rows (input : Input) (index : CandidateIndex) : List Row :=
  SymbolicDuplex.rows (candidateBase input index) ProductPoseidon2.constants
    (builder input index)

theorem rows_length (input : Input) (index : CandidateIndex) :
    (rows input index).length = rowsPerCandidate := by
  rw [rows, SymbolicDuplex.rows_length, builder_entries_length]
  rfl

/-- Satisfaction of the complete indexed physical family. -/
def RowsHold (input : Input) (assignment : Nat -> Nat) : Prop :=
  forall index, Satisfies (rows input index) assignment

def aggregateRowCount : Nat := candidateCount * rowsPerCandidate

theorem aggregateRowCount_eq : aggregateRowCount = 1710720 := by decide

theorem rows_window
    (input : Input) (index : CandidateIndex) (column : Nat)
    (lower : candidateBase input index <= column)
    (upper : column < candidateBase input index + rowsPerCandidate) :
    input.transcriptBase <= column /\
      column < input.transcriptBase + aggregateRowCount := by
  constructor
  · unfold candidateBase at lower
    omega
  · have flatLt := index.flat_lt
    have rowsPositive : 0 < rowsPerCandidate := by decide
    simp only [candidateBase, aggregateRowCount] at upper ⊢
    norm_num [candidateCount, rowsPerCandidate, permutationsPerCandidate,
      SymbolicDuplex.stride] at flatLt rowsPositive upper ⊢
    omega

theorem candidate_windows_disjoint
    (input : Input) (left right : CandidateIndex)
    (different : left ≠ right) :
    candidateBase input left + rowsPerCandidate <= candidateBase input right \/
      candidateBase input right + rowsPerCandidate <= candidateBase input left := by
  have flatDifferent : left.flat ≠ right.flat := by
    intro equal
    exact different (CandidateIndex.flat_injective equal)
  have rowsPositive : 0 < rowsPerCandidate := by decide
  norm_num [candidateBase, rowsPerCandidate, permutationsPerCandidate,
    SymbolicDuplex.stride] at *
  omega

end Nightstream.Implementation.NebulaV2.ProductPiRlcTranscriptRows
