import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement.Core
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement.Evaluation.Chunk0
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement.Evaluation.Chunk1
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement.Evaluation.Chunk2
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement.Evaluation.Chunk3
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement.Evaluation.Chunk4
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement.Evaluation.Chunk5
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement.Evaluation.Chunk6
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement.Product

/-!
Structural aggregation of the bounded certificates for the focused compact
`y_zcol` quadratic refinement.

Owns: proof that the bounded evaluation slices and product certificate cover
their complete typed pair lists.

Does not own: native certificate computation, field semantics, symbolic
execution, selected-row materialization, protocol authority, or security
events.

Emits constraints: no.

| Child path | Mathematical obligation | Authority class |
|---|---|---|
| `Evaluation.Chunk*` | the evaluation slices jointly cover the canonical evaluation-pair list | checked |
| `Product` | every canonical extension-product pair satisfies its expected group match | checked |
| this module | append the child certificates without changing pair order or meaning | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement

/-- Half-open positions owned by the seven evaluation certificates. -/
def evaluationChunkRanges : List (Nat × Nat) :=
  [(0, 7), (7, 14), (14, 21), (21, 28),
    (28, 35), (35, 42), (42, 49)]

theorem evaluationChunkRangesOrdered :
    evaluationChunkRanges.Pairwise
      (fun left right => left.2 ≤ right.1) := by
  decide

theorem evaluationChunkRangesCoverCount :
    (evaluationChunkRanges.map fun range => range.2 - range.1).sum = 49 := by
  decide

private theorem sevenChunksExact {α : Type}
    (values : List α) (length : values.length = 49) :
    (values.drop 0).take 7 ++
      (values.drop 7).take 7 ++
      (values.drop 14).take 7 ++
      (values.drop 21).take 7 ++
      (values.drop 28).take 7 ++
      (values.drop 35).take 7 ++
      (values.drop 42).take 7 = values := by
  simp only [List.drop_zero]
  rw [← List.take_add]
  rw [← List.take_add]
  rw [← List.take_add]
  rw [← List.take_add]
  rw [← List.take_add]
  rw [← List.take_add]
  apply List.take_of_length_le
  omega

private theorem evaluationChunksExact :
    Evaluation.Chunk0.pairs ++
      Evaluation.Chunk1.pairs ++
      Evaluation.Chunk2.pairs ++
      Evaluation.Chunk3.pairs ++
      Evaluation.Chunk4.pairs ++
      Evaluation.Chunk5.pairs ++
      Evaluation.Chunk6.pairs = evaluationPairs := by
  simpa [Evaluation.Chunk0.pairs, Evaluation.Chunk1.pairs,
    Evaluation.Chunk2.pairs, Evaluation.Chunk3.pairs,
    Evaluation.Chunk4.pairs, Evaluation.Chunk5.pairs,
    Evaluation.Chunk6.pairs] using
    sevenChunksExact evaluationPairs evaluationPairsLengthExact

set_option maxRecDepth 100000
set_option maxHeartbeats 1000000

private theorem appendMatches {α : Type} {predicate : α → Prop}
    {left right : List α}
    (leftMatches : ∀ value ∈ left, predicate value)
    (rightMatches : ∀ value ∈ right, predicate value) :
    ∀ value ∈ left ++ right, predicate value := by
  intro value member
  rcases List.mem_append.mp member with leftMember | rightMember
  · exact leftMatches value leftMember
  · exact rightMatches value rightMember

/-- The bounded certificates cover every independent evaluation target. -/
theorem evaluationGroupsExact :
    evaluationPairs.length = 49 ∧
      ∀ pair ∈ evaluationPairs,
        GroupMatches pair.1 (evaluationExpected pair.2) := by
  constructor
  · exact evaluationPairsLengthExact
  · rw [← evaluationChunksExact]
    have chunks01 :
        ∀ pair ∈ Evaluation.Chunk0.pairs ++ Evaluation.Chunk1.pairs,
          GroupMatches pair.1 (evaluationExpected pair.2) :=
      appendMatches Evaluation.Chunk0.pairsMatch Evaluation.Chunk1.pairsMatch
    have chunks012 :
        ∀ pair ∈ Evaluation.Chunk0.pairs ++ Evaluation.Chunk1.pairs ++
            Evaluation.Chunk2.pairs,
          GroupMatches pair.1 (evaluationExpected pair.2) :=
      appendMatches chunks01 Evaluation.Chunk2.pairsMatch
    have chunks0123 :
        ∀ pair ∈ Evaluation.Chunk0.pairs ++ Evaluation.Chunk1.pairs ++
            Evaluation.Chunk2.pairs ++ Evaluation.Chunk3.pairs,
          GroupMatches pair.1 (evaluationExpected pair.2) :=
      appendMatches chunks012 Evaluation.Chunk3.pairsMatch
    have chunks01234 :
        ∀ pair ∈ Evaluation.Chunk0.pairs ++ Evaluation.Chunk1.pairs ++
            Evaluation.Chunk2.pairs ++ Evaluation.Chunk3.pairs ++
            Evaluation.Chunk4.pairs,
          GroupMatches pair.1 (evaluationExpected pair.2) :=
      appendMatches chunks0123 Evaluation.Chunk4.pairsMatch
    have chunks012345 :
        ∀ pair ∈ Evaluation.Chunk0.pairs ++ Evaluation.Chunk1.pairs ++
            Evaluation.Chunk2.pairs ++ Evaluation.Chunk3.pairs ++
            Evaluation.Chunk4.pairs ++ Evaluation.Chunk5.pairs,
          GroupMatches pair.1 (evaluationExpected pair.2) :=
      appendMatches chunks01234 Evaluation.Chunk5.pairsMatch
    exact appendMatches chunks012345 Evaluation.Chunk6.pairsMatch

/-- The product certificate covers every independent extension-product
target. -/
theorem productGroupsExact :
    productPairs.length = 86 ∧
      ∀ pair ∈ productPairs,
        GroupMatches pair.1 (productExpected pair.2) := by
  constructor
  · exact productPairsLengthExact
  · intro pair member
    apply Product.pairsMatch pair
    simpa [Product.pairs] using member

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement
