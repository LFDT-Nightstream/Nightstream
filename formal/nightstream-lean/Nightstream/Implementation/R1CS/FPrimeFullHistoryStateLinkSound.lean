import Nightstream.Implementation.R1CS.FPrimeFullHistoryStateLinkArtifact
import Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseArtifact

/-!
Contract: exact adjacent-state equality for the generated two-step
full-history profile. The Rust exporter checks that these 31 rows are byte-for-
byte the production `decider.state_link` owner.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStateLinkSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStateLink

theorem rowsIncluded :
    rowsIncluded (EqualityPins.rows pairs) rows = true := by
  native_decide

theorem sound {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies rows assignment) :
    ∀ pair ∈ pairs, assignment pair.1 = assignment pair.2 :=
  EqualityPins.sound rowsIncluded canonical one satisfies

/-- The semantic adjacent-state equalities directly satisfy every exact
generated state-link row; there are no auxiliary witness columns. -/
theorem complete {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (links : ∀ pair ∈ pairs, assignment pair.1 = assignment pair.2) :
    Satisfies rows assignment :=
  EqualityPins.rows_complete canonical one links

theorem satisfies_iff_links {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) :
    Satisfies rows assignment ↔
      ∀ pair ∈ pairs, assignment pair.1 = assignment pair.2 :=
  ⟨sound canonical one, complete canonical one⟩

/-- The recursive input state columns, in the exact state-coordinate order,
are derived from the right side of the generated adjacent-state pairs. -/
def recursiveStateInColumns : List Nat := pairs.map Prod.snd

theorem baseStateOutColumns :
    pairs.map Prod.fst = FPrimeFullHistoryBase.stateOutColumns := by
  native_decide

private theorem map_pair_equal
    (assignment : Nat → Nat) (xs : List (Nat × Nat))
    (links : ∀ pair ∈ xs,
      assignment pair.1 = assignment pair.2) :
    xs.map (fun pair => assignment pair.1) =
      xs.map (fun pair => assignment pair.2) := by
  induction xs with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      have headEqual := links head (by simp)
      have tailLinks : ∀ pair ∈ tail,
          assignment pair.1 = assignment pair.2 := by
        intro pair member
        exact links pair (by simp [member])
      simp [headEqual, inductionHypothesis tailLinks]

/-- Exact adjacent-state rows equate the whole base output vector with the
recursive input vector; no coordinate is selected or omitted by hand. -/
theorem stateVectors_sound {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies rows assignment) :
    FPrimeFullHistoryBase.stateOutColumns.map assignment =
      recursiveStateInColumns.map assignment := by
  have links := sound canonical one satisfies
  have mapped := map_pair_equal assignment pairs links
  rw [← baseStateOutColumns]
  simpa [recursiveStateInColumns, List.map_map, Function.comp_def] using mapped

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStateLinkSound
