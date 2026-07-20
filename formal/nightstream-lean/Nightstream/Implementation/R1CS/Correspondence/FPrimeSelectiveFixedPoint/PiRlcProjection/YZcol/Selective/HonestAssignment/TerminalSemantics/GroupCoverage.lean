import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.TerminalSemantics.GroupTransport

/-!
Exact rewrite-group coverage for the bounded `y_zcol` program.

Owns: proving that 49 width-22 evaluation groups followed by 86 width-2
product groups flatten to exactly the 1,250 decoded rewrite steps.

Does not own: terminal proposition composition, selected-row completeness,
producer authority, security events, or permission to remove rows.

Emits constraints: no.

| Leaf | Mathematical obligation | Authority class |
|---|---|---|
| `honest.group_schedule` | `49 × 22 + 86 × 2` positions cover all 1,250 rewrite steps | derived from exact counts |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.TerminalSemantics

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.SourceDecode
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement

@[irreducible] def rewriteGroups : List (List DecodedRewriteStep) :=
  QuadraticRefinement.evaluationGroups ++ QuadraticRefinement.productGroups

private theorem flattenRangeChunksFrom {α : Type}
    (values : List α) (offset width count : Nat) :
    ((List.range count).map fun index =>
        (values.drop (offset + width * index)).take width).flatten =
      (values.drop offset).take (width * count) := by
  have flattenRangeChunks : ∀ count,
      ((List.range count).map fun index =>
          ((values.drop offset).drop (width * index)).take width).flatten =
        (values.drop offset).take (width * count) := by
    intro chunkCount
    induction chunkCount with
    | zero => simp
    | succ previous inductionHypothesis =>
        rw [List.range_succ, List.map_append, List.flatten_append,
          List.map_singleton, List.flatten_singleton,
          inductionHypothesis]
        simpa [Nat.mul_succ] using
          (List.take_add (l := values.drop offset)
            (i := width * previous) (j := width)).symm
  simpa [List.drop_drop] using flattenRangeChunks count

private theorem chunkScheduleCovers {α : Type}
    (values : List α) (lengthExact : values.length = 1250) :
    ((List.range 49).map fun index =>
        (values.drop (22 * index)).take 22).flatten ++
      ((List.range 86).map fun index =>
        (values.drop (1078 + 2 * index)).take 2).flatten = values := by
  have evaluationCoverage :
      ((List.range 49).map fun index =>
        (values.drop (22 * index)).take 22).flatten =
        values.take 1078 := by
    simpa only [Nat.zero_add] using
      flattenRangeChunksFrom values 0 22 49
  have productCoverage :
      ((List.range 86).map fun index =>
        (values.drop (1078 + 2 * index)).take 2).flatten =
        (values.drop 1078).take 172 := by
    exact flattenRangeChunksFrom values 1078 2 86
  rw [evaluationCoverage, productCoverage]
  have tailLength : (values.drop 1078).length = 172 := by
    rw [List.length_drop, lengthExact]
  rw [List.take_of_length_le (Nat.le_of_eq tailLength)]
  exact List.take_append_drop 1078 values

theorem rewriteGroupsExact :
    rewriteGroups.flatten = decodedRewriteSteps := by
  rw [rewriteGroups, List.flatten_append,
    QuadraticRefinement.evaluationGroups,
    QuadraticRefinement.productGroups]
  exact chunkScheduleCovers decodedRewriteSteps decodedRewriteStepsLengthExact

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.TerminalSemantics
