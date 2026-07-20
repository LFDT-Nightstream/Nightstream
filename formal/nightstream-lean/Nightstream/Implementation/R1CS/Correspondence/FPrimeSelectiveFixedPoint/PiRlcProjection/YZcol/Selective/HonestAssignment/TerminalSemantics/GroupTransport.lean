import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.TerminalSemantics.Core

/-!
Checked-group transport for honest `y_zcol` rewrite terminals.

Owns: pairing each checked evaluation/product rewrite group with its exact
source trace and deriving terminal validity from the direct family equation.

Does not own: whole-program group coverage, selected-row completeness,
producer authority, security events, or permission to remove rows.

Emits constraints: no.

| Leaf | Mathematical obligation | Authority class |
|---|---|---|
| `honest.evaluation_group` | a checked evaluation group yields its direct terminal equations | artifact-checked + derived |
| `honest.product_group` | a checked product group yields its direct terminal equations | artifact-checked + derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.TerminalSemantics

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.SourceDecode
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.DirectSemantics
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment

private theorem sourceEvaluationDirect
    {source : Nat → Nat} (honest : HonestSourceBoundary source) :
    ∀ trace ∈ SourceSchedule.evaluationTraces,
      EvaluationDirect trace source := by
  intro trace traceMember
  rcases List.mem_map.mp traceMember with ⟨owner, ownerMember, traceEq⟩
  subst trace
  rw [← honest.sourceEq]
  exact DirectSemantics.sourceEvaluationsDirect honest.seed owner ownerMember

private theorem sourceProductDirect
    {source : Nat → Nat} (honest : HonestSourceBoundary source) :
    ∀ trace ∈ SourceSchedule.productTraces,
      ProductDirect trace source := by
  intro trace traceMember
  rcases List.mem_map.mp traceMember with ⟨owner, ownerMember, traceEq⟩
  subst trace
  rw [← honest.sourceEq]
  exact DirectSemantics.sourceProductsDirect honest.seed owner ownerMember

private theorem evaluationPairGroupsExact :
    QuadraticRefinement.evaluationPairs.map Prod.fst =
      QuadraticRefinement.evaluationGroups := by
  unfold QuadraticRefinement.evaluationPairs
  apply List.map_fst_zip
  simp [QuadraticRefinement.evaluationGroups,
    SourceSchedule.evaluation_trace_count]

private theorem productPairGroupsExact :
    QuadraticRefinement.productPairs.map Prod.fst =
      QuadraticRefinement.productGroups := by
  unfold QuadraticRefinement.productPairs
  apply List.map_fst_zip
  simp [QuadraticRefinement.productGroups,
    SourceSchedule.product_trace_count]

private theorem evaluationPairTracesExact :
    QuadraticRefinement.evaluationPairs.map Prod.snd =
      SourceSchedule.evaluationTraces := by
  unfold QuadraticRefinement.evaluationPairs
  apply List.map_snd_zip
  simp [QuadraticRefinement.evaluationGroups,
    SourceSchedule.evaluation_trace_count]

private theorem productPairTracesExact :
    QuadraticRefinement.productPairs.map Prod.snd =
      SourceSchedule.productTraces := by
  unfold QuadraticRefinement.productPairs
  apply List.map_snd_zip
  simp [QuadraticRefinement.productGroups,
    SourceSchedule.product_trace_count]

theorem evaluationGroupTerminals
    {source : Nat → Nat} (honest : HonestSourceBoundary source) :
    ∀ group ∈ QuadraticRefinement.evaluationGroups, ∀ derived,
      TerminalsHoldFrom source derived group := by
  intro group groupMember derived
  have mapped : group ∈ QuadraticRefinement.evaluationPairs.map Prod.fst := by
    rw [evaluationPairGroupsExact]
    exact groupMember
  rcases List.mem_map.mp mapped with ⟨pair, pairMember, groupEq⟩
  subst group
  apply groupTerminalsHold source derived
  · exact QuadraticRefinement.evaluationGroupsExact.2 pair pairMember
  · apply expectedEvaluationHolds_of_direct
    apply sourceEvaluationDirect honest pair.2
    have traceMapped : pair.2 ∈ QuadraticRefinement.evaluationPairs.map Prod.snd :=
      List.mem_map.mpr ⟨pair, pairMember, rfl⟩
    rw [evaluationPairTracesExact] at traceMapped
    exact traceMapped

theorem productGroupTerminals
    {source : Nat → Nat} (honest : HonestSourceBoundary source) :
    ∀ group ∈ QuadraticRefinement.productGroups, ∀ derived,
      TerminalsHoldFrom source derived group := by
  intro group groupMember derived
  have mapped : group ∈ QuadraticRefinement.productPairs.map Prod.fst := by
    rw [productPairGroupsExact]
    exact groupMember
  rcases List.mem_map.mp mapped with ⟨pair, pairMember, groupEq⟩
  subst group
  apply groupTerminalsHold source derived
  · exact QuadraticRefinement.productGroupsExact.2 pair pairMember
  · apply expectedProductHolds_of_direct
    apply sourceProductDirect honest pair.2
    have traceMapped : pair.2 ∈ QuadraticRefinement.productPairs.map Prod.snd :=
      List.mem_map.mpr ⟨pair, pairMember, rfl⟩
    rw [productPairTracesExact] at traceMapped
    exact traceMapped

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.TerminalSemantics
