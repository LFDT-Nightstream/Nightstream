import Nightstream.Implementation.R1CS.ShiftedTernaryComplete

namespace NightstreamTests.ShiftedTernary

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ShiftedTernary
open Nightstream.Implementation.R1CS.ShiftedTernarySound
open Nightstream.Implementation.R1CS.ShiftedTernaryComplete

def honestAssignment : Nat → Nat := fun column => honestWitness.getD column 0

def forgedAssignment : Nat → Nat := fun column => forgedWitness.getD column 0

/-- The exact Rust-generated honest witness satisfies all 180 rows. -/
theorem honest_satisfies : Satisfies rows honestAssignment := by native_decide

/-- The `x + p` alternate opening fails at the terminal borrow row. -/
example : ¬ Satisfies rows forgedAssignment := by native_decide

/-- The compact seeded-commitment compiler expands to the exact generated
commitment suffix, so its generic theorem applies to this artifact. -/
example : commitmentBlock.Valid ∧ commitmentBlock.rows = rows.drop 126 := by
  exact ⟨commitmentBlock_valid, commitmentRows_eq_artifact⟩

/-- The production Goldilocks fixture receives the full semantic theorem:
canonical field opening, fixed shape, and exact seeded Phi81 commitment. -/
example (prime : EuclidPrime goldilocksP)
    (canonical : ∀ column, honestAssignment column < goldilocksP) :
    OneFieldSound honestAssignment := by
  apply oneField_sound prime
  · exact canonical
  · rfl
  · exact honest_satisfies

/-- Completeness is driven by the native witness-generator relation, not by
an assumed acceptance result or a duplicate row predicate. -/
example (witness : CanonicalWitness honestAssignment) :
    Satisfies ShiftedTernaryCompiler.canonicalRows honestAssignment :=
  canonicalRows_complete witness

end NightstreamTests.ShiftedTernary
