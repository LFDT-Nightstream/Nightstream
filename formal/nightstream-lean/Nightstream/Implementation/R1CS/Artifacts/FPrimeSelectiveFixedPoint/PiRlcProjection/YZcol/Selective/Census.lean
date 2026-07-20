import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Census
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Generated.SelectiveRows
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Generated.StagePaths

/-!
Kernel-checked census for the compact source-to-selective interval artifact.

Owns: exact agreement with the 14 source-stage owners, complete fragment
coverage, unique emitted-row ownership, rewrite-family counts, and the
5,724-source-row to 1,254-selective-row total.

Does not own: rewrite semantics, final matrix coefficients, column ownership,
selector truth, protocol authority, or permission to remove rows.

Emits constraints: no.

| Compiler family | Fragments | Source rows | Selective rows |
|---|---:|---:|---:|
| polynomial evaluation | 49 | 5,288 | 1,078 |
| product sum | 86 | 430 | 172 |
| linear definition | 2 | 2 | 0 |
| retained final checks | 2 | 4 | 4 |
| total | 139 | 5,724 | 1,254 |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Census

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective

private abbrev sourceArtifact : YZcol.Artifact := Generated.Metadata.artifact
private abbrev sourceLeaves : List SourceStageLeaf :=
  sourceArtifact.sourceStageLeaves Generated.StagePaths.paths
private abbrev artifact : Selective.Artifact := Generated.SelectiveRows.artifact

theorem stagePathAgreement :
    artifact.leaves.map LoweredStageLeaf.stagePath =
      sourceLeaves.map SourceStageLeaf.stagePath := by
  decide

theorem stagePathsUnique :
    (artifact.leaves.map LoweredStageLeaf.stagePath).Nodup := by
  rw [stagePathAgreement]
  exact YZcol.Census.sourceStagePathsUnique

theorem stageLeafCount : artifact.leaves.length = 14 := by
  decide

/-- Every generated selective leaf names exactly the source definition/check
indices owned by the corresponding source-stage leaf. -/
theorem sourceIndexAgreement :
    artifact.leaves.map (fun leaf =>
        (leaf.stagePath, leaf.sourceIndices)) =
      sourceLeaves.map (fun leaf =>
        (leaf.stagePath, leaf.rowIndices)) := by
  set_option maxRecDepth 100000 in
    decide

/-- Within each leaf, the 139 exclusive compiler fragments cover every source
index once and in source order. -/
theorem fragmentSourceCoverage :
    artifact.leaves.map LoweredStageLeaf.fragmentSourceIndices =
      artifact.leaves.map LoweredStageLeaf.sourceIndices := by
  set_option maxRecDepth 100000 in
    decide

theorem stageLeafCounts :
    artifact.leaves.map (fun leaf =>
      (leaf.sourceRowCount, leaf.emittedRowCount, leaf.fragments.length)) =
      [ (272, 108, 56),
        (1620, 330, 15),
        (1620, 330, 15),
        (75, 30, 15),
        (108, 22, 1),
        (106, 22, 1),
        (5, 2, 1),
        (2, 2, 1),
        (1620, 330, 15),
        (75, 30, 15),
        (108, 22, 1),
        (106, 22, 1),
        (5, 2, 1),
        (2, 2, 1) ] := by
  set_option maxRecDepth 100000 in
    decide

theorem sourceRowCount : artifact.sourceRowCount = 5724 := by
  decide

theorem emittedRowCount : artifact.emittedRowCount = 1254 := by
  decide

theorem fragmentCount : artifact.fragments.length = 139 := by
  set_option maxRecDepth 100000 in
    decide

private def isPolynomialEvaluation (fragment : LoweredFragment) : Bool :=
  match fragment.disposition with
  | .rewrite _ .polynomialEvaluation => true
  | _ => false

private def isProductSum (fragment : LoweredFragment) : Bool :=
  match fragment.disposition with
  | .rewrite _ .productSum => true
  | _ => false

private def isLinearDefinition (fragment : LoweredFragment) : Bool :=
  match fragment.disposition with
  | .rewrite _ .linearDefinition => true
  | _ => false

private def isRetained (fragment : LoweredFragment) : Bool :=
  match fragment.disposition with
  | .retained => true
  | _ => false

theorem polynomialEvaluationCount :
    (artifact.fragments.filter isPolynomialEvaluation).length = 49 := by
  decide

theorem productSumCount :
    (artifact.fragments.filter isProductSum).length = 86 := by
  decide

theorem linearDefinitionCount :
    (artifact.fragments.filter isLinearDefinition).length = 2 := by
  decide

theorem retainedCount :
    (artifact.fragments.filter isRetained).length = 2 := by
  decide

private def intervalWithin (outer inner : RowBlock) : Prop :=
  outer.start ≤ inner.start ∧ inner.stop ≤ outer.stop

private instance (outer inner : RowBlock) :
    Decidable (intervalWithin outer inner) := by
  unfold intervalWithin
  infer_instance

private def emittedBoundsCheck : Bool :=
  artifact.emittedIntervals.all fun rows =>
    decide (intervalWithin artifact.steadyArmRows rows) &&
      decide (rows.stop ≤ artifact.finalRelationRowCount)

private theorem emittedBoundsCheck_true : emittedBoundsCheck = true := by
  native_decide

theorem emittedIntervalsBounded :
    ∀ rows ∈ artifact.emittedIntervals,
      intervalWithin artifact.steadyArmRows rows ∧
        rows.stop ≤ artifact.finalRelationRowCount := by
  intro rows member
  have checked := (List.all_eq_true.mp emittedBoundsCheck_true) rows member
  simpa [emittedBoundsCheck, decide_eq_true_eq] using checked

def IntervalsDisjoint (left right : RowBlock) : Prop :=
  left.stop ≤ right.start ∨ right.stop ≤ left.start

private instance (left right : RowBlock) :
    Decidable (IntervalsDisjoint left right) := by
  unfold IntervalsDisjoint
  infer_instance

def PairwiseDisjoint : List RowBlock → Prop
  | [] => True
  | first :: rest =>
      (∀ second ∈ rest, IntervalsDisjoint first second) ∧
        PairwiseDisjoint rest

private def pairwiseDisjointCheck : List RowBlock → Bool
  | [] => true
  | first :: rest =>
      (rest.all fun second => decide (IntervalsDisjoint first second)) &&
        pairwiseDisjointCheck rest

private theorem pairwiseDisjointCheck_eq_true_iff :
    ∀ intervals, pairwiseDisjointCheck intervals = true ↔
      PairwiseDisjoint intervals
  | [] => by simp [pairwiseDisjointCheck, PairwiseDisjoint]
  | first :: rest => by
      simp [pairwiseDisjointCheck, PairwiseDisjoint,
        pairwiseDisjointCheck_eq_true_iff, List.all_eq_true,
        decide_eq_true_eq]

private theorem emittedIntervalsDisjointCheck :
    pairwiseDisjointCheck artifact.emittedIntervals = true := by
  native_decide

/-- No selected emitted row is charged to two source-stage leaves or two
rewrite fragments. -/
theorem emittedIntervalsDisjoint :
    PairwiseDisjoint artifact.emittedIntervals :=
  (pairwiseDisjointCheck_eq_true_iff _).mp emittedIntervalsDisjointCheck

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Census
