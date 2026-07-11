import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalRunningLinkArtifact

/-!
Contract: exact continuity of the four-lane running accumulator handle at the
terminal fold boundary.  This digest is compression only; full parent/child
authority continuity is proved by the separate terminal parent and CE owners.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalRunningLinkSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalRunningLink

def terminalRunningDigest (assignment : Nat → Nat) : List Nat :=
  pairs.map fun pair => assignment pair.1

def priorAccumulatorDigest (assignment : Nat → Nat) : List Nat :=
  pairs.map fun pair => assignment pair.2

def Holds (assignment : Nat → Nat) : Prop :=
  terminalRunningDigest assignment = priorAccumulatorDigest assignment

private theorem values_eq_iff (assignment : Nat → Nat) :
    Holds assignment ↔
      ∀ pair ∈ pairs, assignment pair.1 = assignment pair.2 := by
  unfold Holds terminalRunningDigest priorAccumulatorDigest
  induction pairs with
  | nil => simp
  | cons head tail ih =>
      simp only [List.map_cons, List.cons.injEq]
      constructor
      · rintro ⟨headEq, tailEq⟩ pair member
        rcases List.mem_cons.mp member with rfl | tailMember
        · exact headEq
        · exact (ih.mp tailEq) pair tailMember
      · intro pointwise
        exact ⟨pointwise head (by simp), ih.mpr (fun pair member =>
          pointwise pair (by simp [member]))⟩

theorem sound {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies rows assignment) :
    Holds assignment :=
  (values_eq_iff assignment).mpr
    (EqualityPins.rows_sound canonical one satisfies)

theorem complete {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Holds assignment) :
    Satisfies rows assignment :=
  EqualityPins.rows_complete canonical one
    ((values_eq_iff assignment).mp holds)

theorem satisfies_iff_holds {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) :
    Satisfies rows assignment ↔ Holds assignment :=
  ⟨sound canonical one, complete canonical one⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalRunningLinkSound
