import Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseArtifact

/-!
Contract: universal checked-program soundness and completeness for the base
owner of the exact composed two-step full-history profile.

This owner includes both the generated base F' step and verifier-derived base
state pins.  The theorem retains all assertions as checks; it does not infer a
protocol fact from the owner label or row hash.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.FPrimeFullHistoryBase

def interpret (state : Nat → Nat) : Nat → Nat :=
  CheckedProgram.interpret state instructions

def ValidInput (state : Nat → Nat) : Prop :=
  ChecksHold state instructions

theorem xOutColumns_known :
    ∀ column ∈ xOutColumns,
      column ∈ knownAfter inputColumns (definitions instructions) := by
  native_decide

theorem fPrimeFullHistoryBase_sound {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (satisfies : Satisfies rows assignment) :
    SoundResult inputColumns instructions assignment assignment := by
  exact sound definitions_wellFormed definitions_canonical checks_reference
    (by intro _ _; rfl) canonical constantOne satisfies

theorem fPrimeFullHistoryBase_xOut_unique
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

theorem fPrimeFullHistoryBase_complete {state : Nat → Nat}
    (canonical : ∀ column, state column < goldilocksP)
    (constantOne : state 0 = 1)
    (valid : ValidInput state) : Satisfies rows (interpret state) := by
  exact complete definitions_wellFormed definitions_canonical canonical
    (by native_decide) constantOne valid

end Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseSound
