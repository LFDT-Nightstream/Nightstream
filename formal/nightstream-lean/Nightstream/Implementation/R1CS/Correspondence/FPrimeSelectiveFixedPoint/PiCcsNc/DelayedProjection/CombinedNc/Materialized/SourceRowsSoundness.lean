import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RoundArtifactSemantics
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RoundChainArtifact
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.StageProgram

/-!
Semantic consequences of literal production combined-NC source-row truth.

Owns: the kernel-only split of satisfaction of the ordered 8,021 generated
source rows into the exact padding, claimed-initial, 25-round, and terminal
artifact leaves, and their composition into one typed consequence package.

Does not own: source-to-selective refinement, construction of the source
assignment, transcript scheduling, parent or raw-child authority, commitment
binding, SumCheck soundness, costs, or permission to remove rows.

The split uses `StageProgram.stageSourceRows_coverage`, whose proof joins the
bounded stage certificates.  This leaf performs no closed evaluation over the
complete generated list and introduces no stage-satisfaction premise.

Assurance tier: artifact-checked for the fixed generated source profile,
conditional on literal source-row satisfaction, canonical assignment words,
and the constant-one column.
-/

/-!
Emits constraints: none; this module proves soundness of already-generated source rows.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.source_rows_soundness` | Show exact source-row satisfaction implies the complete source program. | derived |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceRowsSoundness

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Semantics

private theorem mappedSatisfies_of_subset
    {assignment : Nat → Nat} {whole part : List RawSourceRow}
    (satisfies :
      Satisfies (whole.map SourceDecodeBridge.rawRow) assignment)
    (subset : ∀ raw ∈ part, raw ∈ whole) :
    Satisfies (part.map SourceDecodeBridge.rawRow) assignment := by
  intro row rowMember
  rcases List.mem_map.mp rowMember with ⟨raw, rawMember, rfl⟩
  exact satisfies _ (List.mem_map.mpr
    ⟨raw, subset raw rawMember, rfl⟩)

private theorem satisfies_append_left
    {assignment : Nat → Nat} {left right : List Row}
    (satisfies : Satisfies (left ++ right) assignment) :
    Satisfies left assignment := by
  intro row member
  exact satisfies row (List.mem_append_left right member)

private theorem satisfies_append_right
    {assignment : Nat → Nat} {left right : List Row}
    (satisfies : Satisfies (left ++ right) assignment) :
    Satisfies right assignment := by
  intro row member
  exact satisfies row (List.mem_append_right left member)

private theorem paddingRow_mem_stageSourceRows
    {raw : RawSourceRow}
    (member : raw ∈ StageProgram.paddingSourceRows) :
    raw ∈ StageProgram.stageSourceRows := by
  rw [StageProgram.stageSourceRows]
  simp only [List.mem_append]
  exact Or.inl (Or.inl (Or.inl member))

private theorem initialRow_mem_stageSourceRows
    {raw : RawSourceRow}
    (member : raw ∈ InitialArtifact.claimedInitialRows) :
    raw ∈ StageProgram.stageSourceRows := by
  rw [StageProgram.stageSourceRows]
  simp only [List.mem_append]
  exact Or.inl (Or.inl (Or.inr member))

private theorem roundRow_mem_stageSourceRows
    {raw : RawSourceRow}
    (member : raw ∈ StageProgram.roundSourceRows) :
    raw ∈ StageProgram.stageSourceRows := by
  rw [StageProgram.stageSourceRows]
  simp only [List.mem_append]
  exact Or.inl (Or.inr member)

private theorem terminalRow_mem_stageSourceRows
    {raw : RawSourceRow}
    (member : raw ∈ TerminalArtifact.generatedTerminalRows) :
    raw ∈ StageProgram.stageSourceRows := by
  rw [StageProgram.stageSourceRows]
  simp only [List.mem_append]
  exact Or.inr member

private theorem paddingSourceRowsSatisfy_of_stage
    {assignment : Nat → Nat}
    (satisfies :
      Satisfies
        (StageProgram.paddingSourceRows.map SourceDecodeBridge.rawRow)
        assignment) :
    PaddingArtifact.SourceRowsSatisfy assignment := by
  have joined :
      Satisfies
        (PaddingArtifact.rawRows PaddingArtifact.sourceShard0 ++
          (PaddingArtifact.rawRows PaddingArtifact.sourceShard1 ++
            PaddingArtifact.rawRows PaddingArtifact.sourceShard2))
        assignment := by
    simpa only [StageProgram.paddingSourceRows, PaddingArtifact.rawRows,
      PaddingArtifact.rawRow, PaddingArtifact.rawTerms,
      SourceDecodeBridge.rawRow, SourceDecodeBridge.rawTerms,
      List.map_append] using satisfies
  have tail := satisfies_append_right joined
  exact ⟨satisfies_append_left joined,
    satisfies_append_left tail, satisfies_append_right tail⟩

private theorem roundSourceRowsAt_mem_roundSourceRows
    (index : Fin sumcheckRoundCount) {raw : RawSourceRow}
    (member : raw ∈ StageProgram.roundSourceRowsAt index.val) :
    raw ∈ StageProgram.roundSourceRows := by
  unfold StageProgram.roundSourceRows
  apply List.mem_flatten.mpr
  refine ⟨StageProgram.roundSourceRowsAt index.val, ?_, member⟩
  unfold StageProgram.roundSourceStages
  exact List.mem_ofFn.mpr ⟨index, rfl⟩

private theorem generatedRoundRowsSatisfy_of_stage
    {assignment : Nat → Nat}
    (satisfies :
      Satisfies
        (StageProgram.roundSourceRows.map SourceDecodeBridge.rawRow)
        assignment) :
    RoundChainArtifact.GeneratedRoundRowsSatisfy assignment := by
  intro round roundMember
  rcases List.getElem?_of_mem roundMember with ⟨index, lookup⟩
  rcases List.getElem?_eq_some_iff.mp lookup with
    ⟨indexLtLength, _⟩
  have indexLt : index < sumcheckRoundCount := by
    rw [RoundArtifact.generatedRoundMapsValid.1] at indexLtLength
    exact indexLtLength
  have localSourceSatisfies :
      Satisfies
        ((StageProgram.roundSourceRowsAt index).map
          SourceDecodeBridge.rawRow)
        assignment :=
    mappedSatisfies_of_subset satisfies
      (fun _ member =>
        roundSourceRowsAt_mem_roundSourceRows
          ⟨index, indexLt⟩ member)
  have localArtifactSatisfies :
      Satisfies
        (RoundArtifact.rawRows (StageProgram.roundSourceRowsAt index))
        assignment := by
    simpa only [RoundArtifact.rawRows, RoundArtifact.rawRow,
      RoundArtifact.rawTerms, SourceDecodeBridge.rawRow,
      SourceDecodeBridge.rawTerms] using localSourceSatisfies
  rcases RoundArtifact.certificate_exact_rows
      (StageProgram.roundCertificateAt index indexLt) with
    ⟨certifiedRound, certifiedLookup, rowsExact⟩
  have certifiedRoundEq : certifiedRound = round := by
    exact (Option.some.inj (lookup.symm.trans certifiedLookup)).symm
  subst certifiedRound
  exact
    (RoundArtifactSemantics.satisfies_iff_of_rowsPermutationEquivalentList
      assignment rowsExact).mp localArtifactSatisfies

/-- Exact typed consequences currently available from the four independent
materialized stage leaves.  The initial and terminal boundary reads remain
explicit because no theorem in this layer identifies them with authoritative
parent/child state or transcript data. -/
structure Consequences (assignment : Nat → Nat) : Prop where
  padding : ∀ output : Fin outputCount,
    PaddingArtifact.OutputPaddingZero assignment output.val
  claimedInitial :
    boundaryClaimedInitial productionBoundary assignment =
      K.mul (boundaryBatchWeight productionBoundary assignment)
        (Nightstream.SuperNeo.ProjectionCheck.eval K.ops
          (boundaryPendingParentYZcol productionBoundary assignment)
          (boundaryProducerBeta productionBoundary assignment))
  roundChain :
    Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.Chain
      ClaimedChain.ops
      (ClaimedChain.initial RoundMaps.values assignment)
      (ClaimedChain.certificate RoundMaps.values assignment).rounds
      (ClaimedChain.challenges RoundMaps.values assignment)
      (ClaimedChain.terminal RoundMaps.values assignment)
  terminal : TerminalProgram.Computed assignment

/-- Literal satisfaction of all 8,021 generated source rows implies the
padding, claimed-initial, complete claimed-chain, and terminal semantics.
No stage truth is accepted as a premise: each stage is obtained by a kernel
membership restriction of the same ordered source list. -/
theorem sourceRowsSatisfy_implies_consequences
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (sourceRowsSatisfy :
      Satisfies
        (SourceRows.values.map SourceDecodeBridge.rawRow) assignment) :
    Consequences assignment := by
  have stageRowsSatisfy :
      Satisfies
        (StageProgram.stageSourceRows.map SourceDecodeBridge.rawRow)
        assignment := by
    rw [StageProgram.stageSourceRows_coverage]
    exact sourceRowsSatisfy
  have paddingRowsSatisfy :
      Satisfies
        (StageProgram.paddingSourceRows.map SourceDecodeBridge.rawRow)
        assignment :=
    mappedSatisfies_of_subset stageRowsSatisfy
      (fun _ member => paddingRow_mem_stageSourceRows member)
  have initialRowsSatisfy :
      Satisfies
        (InitialArtifact.claimedInitialRows.map SourceDecodeBridge.rawRow)
        assignment :=
    mappedSatisfies_of_subset stageRowsSatisfy
      (fun _ member => initialRow_mem_stageSourceRows member)
  have roundRowsSatisfy :
      Satisfies
        (StageProgram.roundSourceRows.map SourceDecodeBridge.rawRow)
        assignment :=
    mappedSatisfies_of_subset stageRowsSatisfy
      (fun _ member => roundRow_mem_stageSourceRows member)
  have terminalRowsSatisfy :
      Satisfies
        (TerminalArtifact.generatedTerminalRows.map
          SourceDecodeBridge.rawRow)
        assignment :=
    mappedSatisfies_of_subset stageRowsSatisfy
      (fun _ member => terminalRow_mem_stageSourceRows member)
  have paddingSatisfies :=
    paddingSourceRowsSatisfy_of_stage paddingRowsSatisfy
  have initialSatisfies :
      Satisfies
        (InitialArtifact.rawRows InitialArtifact.claimedInitialRows)
        assignment := by
    simpa only [InitialArtifact.rawRows] using initialRowsSatisfy
  have generatedRoundsSatisfy :=
    generatedRoundRowsSatisfy_of_stage roundRowsSatisfy
  have terminalSatisfies :
      TerminalArtifact.GeneratedTerminalRowsSatisfy assignment := by
    simpa only [TerminalArtifact.GeneratedTerminalRowsSatisfy,
      TerminalArtifact.Certificates.rawRows] using terminalRowsSatisfy
  exact
    { padding := fun output =>
        PaddingArtifact.outputYZcolPaddingZero paddingSatisfies constantOne
          output
      claimedInitial :=
        InitialArtifact.sourceRows_imply_boundaryClaimedInitial canonical
          constantOne initialSatisfies
      roundChain :=
        RoundChainArtifact.claimedChain_of_generated_round_rows canonical
          constantOne generatedRoundsSatisfy
      terminal :=
        TerminalArtifact.generatedTerminalRowsSatisfy_implies_computed
          canonical constantOne terminalSatisfies }

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceRowsSoundness
