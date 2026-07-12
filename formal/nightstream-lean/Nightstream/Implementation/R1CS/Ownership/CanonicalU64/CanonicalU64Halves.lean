import Nightstream.Implementation.R1CS.Correspondence.CanonicalU64.CanonicalU64Sound
import Nightstream.Implementation.R1CS.Core.Program
import Nightstream.Implementation.R1CS.Core.Relabel

/-!
Contract: reusable integer semantics for the two 32-bit limbs produced after
a canonical-u64 decomposition. This is the bridge used by Poseidon preimage
certificates for F' counters and program counters.
-/

namespace Nightstream.Implementation.R1CS.CanonicalU64Halves

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program

set_option maxHeartbeats 1000000

def sourceTerms (offset : Nat) : List (Nat × Nat) :=
  (List.range 32).map
    (fun bit => (CanonicalU64.bitCol (offset + bit), 2 ^ bit))

def halfValue (assignment : Nat → Nat) (offset : Nat) : Nat :=
  rawLcEval assignment (sourceTerms offset)

def definition (columnMap : List Nat) (output offset : Nat) : Definition where
  output := output
  rhs := .linear (Relabel.terms columnMap (sourceTerms offset))

private theorem range32 :
    List.range 32 =
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] := by
  decide

private theorem range64 :
    List.range 64 =
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
       32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
       48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] := by
  decide

theorem bitsValue_eq_halves (assignment : Nat → Nat) :
    bitsValue assignment =
      halfValue assignment 0 + 2 ^ 32 * halfValue assignment 32 := by
  simp only [bitsValue, range64, List.foldl_cons, List.foldl_nil]
  simp only [halfValue, sourceTerms, range32, List.map_cons, List.map_nil,
    rawLcEval, CanonicalU64.bitCol]
  simp only [Nat.zero_add, Nat.add_zero, Nat.reduceAdd]
  omega

theorem lcEval_sourceTerms (assignment : Nat → Nat) (offset : Nat) :
    lcEval assignment (sourceTerms offset) =
      halfValue assignment offset % goldilocksP := by
  exact lcEval_eq_raw_mod assignment (sourceTerms offset)

/-- A mapped canonical-u64 gadget fixing its field value to one makes its two
half-word definitions evaluate to one and zero. -/
theorem one_halves_sound
    {columnMap : List Nat} {lowColumn highColumn : Nat}
    {assignment : Nat → Nat}
    (fieldOne :
      Relabel.assignment columnMap assignment CanonicalU64.varCol = 1)
    (canonicalResult :
      Relabel.assignment columnMap assignment CanonicalU64.varCol =
          bitsValue (Relabel.assignment columnMap assignment) ∧
        bitsValue (Relabel.assignment columnMap assignment) < goldilocksP)
    (lowHolds : (definition columnMap lowColumn 0).Holds assignment)
    (highHolds : (definition columnMap highColumn 32).Holds assignment) :
    assignment lowColumn = 1 ∧ assignment highColumn = 0 := by
  let pulled := Relabel.assignment columnMap assignment
  have total : bitsValue pulled = 1 := canonicalResult.1.symm.trans fieldOne
  have split := bitsValue_eq_halves pulled
  have lowBound : halfValue pulled 0 < goldilocksP := by omega
  have highBound : halfValue pulled 32 < goldilocksP := by omega
  have lowEquation :
      assignment lowColumn = halfValue pulled 0 := by
    unfold definition Definition.Holds at lowHolds
    simp only [Rhs.eval] at lowHolds
    rw [Relabel.lcEval_terms, lcEval_sourceTerms] at lowHolds
    change assignment lowColumn = halfValue pulled 0 % goldilocksP at lowHolds
    rwa [Nat.mod_eq_of_lt lowBound] at lowHolds
  have highEquation :
      assignment highColumn = halfValue pulled 32 := by
    unfold definition Definition.Holds at highHolds
    simp only [Rhs.eval] at highHolds
    rw [Relabel.lcEval_terms, lcEval_sourceTerms] at highHolds
    change assignment highColumn = halfValue pulled 32 % goldilocksP at highHolds
    rwa [Nat.mod_eq_of_lt highBound] at highHolds
  omega

/-- A canonical-u64 gadget whose field value is a 32-bit integer makes its
two half-word definitions evaluate to that integer and zero.  This is the
profile-independent form used by generated counter transitions. -/
theorem small_halves_sound
    {columnMap : List Nat} {lowColumn highColumn value : Nat}
    {assignment : Nat → Nat}
    (fieldValue :
      Relabel.assignment columnMap assignment CanonicalU64.varCol = value)
    (valueLt : value < 2 ^ 32)
    (canonicalResult :
      Relabel.assignment columnMap assignment CanonicalU64.varCol =
          bitsValue (Relabel.assignment columnMap assignment) ∧
        bitsValue (Relabel.assignment columnMap assignment) < goldilocksP)
    (lowHolds : (definition columnMap lowColumn 0).Holds assignment)
    (highHolds : (definition columnMap highColumn 32).Holds assignment) :
    assignment lowColumn = value ∧ assignment highColumn = 0 := by
  let pulled := Relabel.assignment columnMap assignment
  have total : bitsValue pulled = value :=
    canonicalResult.1.symm.trans fieldValue
  have split := bitsValue_eq_halves pulled
  have lowBound : halfValue pulled 0 < goldilocksP := by omega
  have highBound : halfValue pulled 32 < goldilocksP := by omega
  have lowEquation : assignment lowColumn = halfValue pulled 0 := by
    unfold definition Definition.Holds at lowHolds
    simp only [Rhs.eval] at lowHolds
    rw [Relabel.lcEval_terms, lcEval_sourceTerms] at lowHolds
    change assignment lowColumn = halfValue pulled 0 % goldilocksP at lowHolds
    rwa [Nat.mod_eq_of_lt lowBound] at lowHolds
  have highEquation : assignment highColumn = halfValue pulled 32 := by
    unfold definition Definition.Holds at highHolds
    simp only [Rhs.eval] at highHolds
    rw [Relabel.lcEval_terms, lcEval_sourceTerms] at highHolds
    change assignment highColumn = halfValue pulled 32 % goldilocksP at highHolds
    rwa [Nat.mod_eq_of_lt highBound] at highHolds
  omega

end Nightstream.Implementation.R1CS.CanonicalU64Halves
