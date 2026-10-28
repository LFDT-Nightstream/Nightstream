import NightstreamFPrime.Layout.MatrixProgram

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECSourceIndex

open NightstreamFPrime.Layout.MatrixProgram

def findRange : List IndexRange → Nat → Option Nat
  | [], _ => none
  | range :: rest, source =>
      if range.start ≤ source ∧ source < range.start + range.count then
        some (source - range.start)
      else
        (findRange rest source).map fun ordinal => range.count + ordinal

theorem findRange_value (ranges : List IndexRange) (source : Nat)
    {ordinal : Nat} (found : findRange ranges source = some ordinal) :
    (IndexSchedule.rangeList ranges).index? ordinal = some source := by
  induction ranges generalizing ordinal with
  | nil =>
      simp only [findRange] at found
      cases found
  | cons range rest inductionHypothesis =>
      by_cases inside : range.start ≤ source ∧ source < range.start + range.count
      · rw [findRange, ite_eq_left inside] at found
        have equal := Option.some.inj found
        subst ordinal
        have bounded : source - range.start < range.count := by omega
        have restored : range.start + (source - range.start) = source := by omega
        simp only [IndexSchedule.index?, IndexSchedule.index?.select,
          ite_eq_left bounded, restored]
      · rw [findRange, ite_eq_right inside] at found
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
            rw [IndexSchedule.index?.select, ite_eq_right outside, offset]
            exact inductionHypothesis tail

end NightstreamFPrime.Export.Stage1.PiDECSourceIndex
