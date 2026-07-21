import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RoundArtifact
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.QuarticEmbedding

/-!
Semantic parent of the bounded production-round artifact certificates.

Owns: transport of source-row satisfaction across the certificate's exact
per-row sparse-term permutations and generated column relabeling, yielding
the independent five-coefficient quartic production-round equations.

Does not own: source-to-selective rewrite refinement, transcript order, round
continuity, terminal-NC semantics, parent or raw-child authority, or
commitment binding.

Emits constraints: none.  The artifact certificate owns exactly one 30-row
production round.
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RoundArtifactSemantics

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionIndexedRows
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc

private theorem rowHolds_iff_of_permutation
    (assignment : Nat → Nat) {left right : Row}
    (permutation : RowsPermutationEquivalent left right) :
    RowHolds assignment left ↔ RowHolds assignment right := by
  unfold RowHolds
  rw [Program.lcEval_eq_of_perm assignment permutation.1,
    Program.lcEval_eq_of_perm assignment permutation.2.1,
    Program.lcEval_eq_of_perm assignment permutation.2.2]

private theorem satisfies_cons_iff
    (head : Row) (tail : List Row) (assignment : Nat → Nat) :
    Satisfies (head :: tail) assignment ↔
      RowHolds assignment head ∧ Satisfies tail assignment := by
  simp [Satisfies]

/-- Lockstep sparse-term permutation preserves the meaning of every row in
the exact schedule.  This is a generic kernel theorem; no generated data are
evaluated here. -/
theorem satisfies_iff_of_rowsPermutationEquivalentList
    (assignment : Nat → Nat) {left right : List Row}
    (permutation : RowsPermutationEquivalentList left right) :
    Satisfies left assignment ↔ Satisfies right assignment := by
  induction left generalizing right with
  | nil =>
      cases right with
      | nil => simp [Satisfies]
      | cons _ _ =>
          simp [RowsPermutationEquivalentList] at permutation
  | cons leftHead leftTail inductionHypothesis =>
      cases right with
      | nil =>
          simp [RowsPermutationEquivalentList] at permutation
      | cons rightHead rightTail =>
          change RowsPermutationEquivalent leftHead rightHead ∧
            RowsPermutationEquivalentList leftTail rightTail at permutation
          rw [satisfies_cons_iff, satisfies_cons_iff,
            rowHolds_iff_of_permutation assignment permutation.1,
            inductionHypothesis permutation.2]

/-- A generated 30-row certificate plus satisfaction of those exact source
rows yields the independently stated production-round equations under the
certificate's column map.  The returned map is the one found at the certified
generated index, not caller-provided metadata. -/
theorem productionAccepted_of_certificate
    {index : Nat} {rows : List RawSourceRow} {assignment : Nat → Nat}
    (certificate : RoundArtifact.Certificate index rows)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (RoundArtifact.rawRows rows) assignment) :
    ∃ round,
      RoundMaps.values[index]? = some round ∧
      ProductionRound.Accepted
        (Relabel.assignment round.columnMap assignment) := by
  rcases RoundArtifact.certificate_lookup certificate with
    ⟨round, lookup, valid⟩
  have mappedSatisfies :
      Satisfies
        (ProductionRound.rows.map (Relabel.row round.columnMap)) assignment :=
    (satisfies_iff_of_rowsPermutationEquivalentList assignment
      valid.2.2.2.2.2).mp satisfies
  have mapsOne : Relabel.column round.columnMap 0 = 0 := by
    simpa [Relabel.column] using valid.1.2.2.2.1
  exact ⟨round, lookup,
    ProductionRound.mapped_sound round.columnMap mapsOne canonical one
      mappedSatisfies⟩

/-- The certified production round directly implies the quartic claimed-round
equations for the exact five assignment-derived coefficient pairs. -/
theorem quarticAccepted_of_certificate
    {index : Nat} {rows : List RawSourceRow} {assignment : Nat → Nat}
    (certificate : RoundArtifact.Certificate index rows)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (RoundArtifact.rawRows rows) assignment) :
    ∃ round,
      RoundMaps.values[index]? = some round ∧
      QuarticEmbedding.ClaimedRoundAccepted
        (ProductionRound.coefficientValues
          (Relabel.assignment round.columnMap assignment))
        (ProductionRound.claimInValue
          (Relabel.assignment round.columnMap assignment))
        (ProductionRound.challengeValue
          (Relabel.assignment round.columnMap assignment))
        (ProductionRound.claimOutValue
          (Relabel.assignment round.columnMap assignment)) := by
  rcases productionAccepted_of_certificate certificate canonical one
      satisfies with ⟨round, lookup, accepted⟩
  refine ⟨round, lookup, ?_⟩
  exact (QuarticEmbedding.productionAccepted_iff_claimedRoundAccepted
    (Relabel.assignment round.columnMap assignment)).mp accepted

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RoundArtifactSemantics
