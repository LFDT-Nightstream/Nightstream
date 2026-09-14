import NightstreamFPrime.Layout.MatrixProgram

/-!
Find a source index in the existing range-list schedule without expanding
its indices. The first matching range determines the returned local ordinal.
The value theorem needs no validity, ordering or uniqueness premise.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECSourceIndex

open NightstreamFPrime.Layout.MatrixProgram

/-- Scan only range headers and retain the cumulative local offset. -/
def findRange : List IndexRange → Nat → Option Nat
  | [], _ => none
  | range :: rest, source =>
      if range.start ≤ source ∧ source < range.start + range.count then
        some (source - range.start)
      else
        (findRange rest source).map fun ordinal => range.count + ordinal

/-- Every returned local ordinal selects the original source index through
the existing interpreter. Overlapping and empty ranges do not weaken this
direction: findRange returns the first match and preserves preceding counts. -/
theorem findRange_value (ranges : List IndexRange) (source : Nat)
    {ordinal : Nat} (found : findRange ranges source = some ordinal) :
    (IndexSchedule.rangeList ranges).index? ordinal = some source := by
  induction ranges generalizing ordinal with
  | nil =>
      simp only [findRange] at found
      cases found
  | cons range rest inductionHypothesis =>
      by_cases inside : range.start ≤ source ∧ source < range.start + range.count
      · rw [findRange, if_pos inside] at found
        have equal := Option.some.inj found
        subst ordinal
        have bounded : source - range.start < range.count := by omega
        have restored : range.start + (source - range.start) = source := by omega
        simp only [IndexSchedule.index?, IndexSchedule.index?.select,
          if_pos bounded, restored]
      · rw [findRange, if_neg inside] at found
        cases tail : findRange rest source with
        | none =>
            simp only [tail, Option.map_none] at found
            cases found
        | some next =>
            simp only [tail, Option.map_some, Option.some.injEq] at found
            subst ordinal
            have outside : ¬ range.count + next < range.count := by omega
            have offset : range.count + next - range.count = next := by omega
            change IndexSchedule.index?.select (range :: rest)
              (range.count + next) = some source
            rw [IndexSchedule.index?.select, if_neg outside, offset]
            exact inductionHypothesis tail

end NightstreamFPrime.Export.Stage1.PiDECSourceIndex
