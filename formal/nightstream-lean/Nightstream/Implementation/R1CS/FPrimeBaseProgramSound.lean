import Nightstream.Implementation.R1CS.FPrimeBaseProgramArtifact

/-!
Contract: exact-row soundness and completeness of the complete production
plain F' base-step circuit.

Unlike a definition-only gadget, this 12,498-row program contains 1,598
assertions. The checked-program theorem preserves them as executable validity
conditions; no assertion is solved for a prover-controlled input. This module
therefore proves exact circuit functionality, output uniqueness, and witness
construction without yet claiming that the extracted checks refine every
field of the high-level `Step.BaseLocalHolds` predicate.
-/

namespace Nightstream.Implementation.R1CS.FPrimeBaseProgramSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.FPrimeBaseProgram

def interpret (state : Nat → Nat) : Nat → Nat :=
  CheckedProgram.interpret state instructions

def ValidInput (state : Nat → Nat) : Prop :=
  ChecksHold state instructions

theorem xOutColumns_known :
    ∀ column ∈ xOutColumns,
      column ∈ knownAfter inputColumns (definitions instructions) := by
  native_decide

/-- Every satisfying assignment of all exact production base rows agrees with
the deterministic interpreter and satisfies every retained verifier check. -/
theorem fPrimeBaseProgram_sound {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (satisfies : Satisfies rows assignment) :
    SoundResult inputColumns instructions assignment assignment := by
  exact sound definitions_wellFormed definitions_canonical checks_reference
    (by intro _ _; rfl) canonical constantOne satisfies

/-- The four raw x_out lanes are uniquely fixed by the checked program inputs
for every pair of satisfying exact-row assignments. -/
theorem fPrimeBaseProgram_xOut_unique
    {left right : Nat → Nat}
    (leftCanonical : ∀ column, left column < goldilocksP)
    (rightCanonical : ∀ column, right column < goldilocksP)
    (leftOne : left 0 = 1) (rightOne : right 0 = 1)
    (leftSat : Satisfies rows left) (rightSat : Satisfies rows right)
    (inputsEqual : AgreeOn left right inputColumns) :
    ∀ column ∈ xOutColumns, left column = right column := by
  have leftResult := sound definitions_wellFormed definitions_canonical
    checks_reference (state := left) (assignment := left)
    (by intro _ _; rfl) leftCanonical leftOne leftSat
  have rightResult := sound definitions_wellFormed definitions_canonical
    checks_reference (state := left) (assignment := right)
    inputsEqual rightCanonical rightOne rightSat
  intro column member
  have known := xOutColumns_known column member
  exact (leftResult.agreement column known).symm.trans
    (rightResult.agreement column known)

/-- Every canonical input satisfying the extracted verifier checks yields a
satisfying witness for all 12,498 exact rows. -/
theorem fPrimeBaseProgram_complete {state : Nat → Nat}
    (canonical : ∀ column, state column < goldilocksP)
    (constantOne : state 0 = 1)
    (valid : ValidInput state) : Satisfies rows (interpret state) := by
  exact complete definitions_wellFormed definitions_canonical canonical
    (by native_decide) constantOne valid

end Nightstream.Implementation.R1CS.FPrimeBaseProgramSound
