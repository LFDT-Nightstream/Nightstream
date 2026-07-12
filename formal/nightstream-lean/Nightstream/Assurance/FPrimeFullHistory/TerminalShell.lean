import Nightstream.Assurance.FPrimeConcreteNifs
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryParentCeSerialization
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryPublicPinsSound
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryTerminalAccumulatorSound
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryTerminalCeSound
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryTerminalContinuitySound
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryTerminalLinkSound
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryTerminalParentLinkSound
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryTerminalRunningLinkSound

/-!
Contract: exact fixed-profile assurance for the production terminal shell.

The shell owns every terminal row family outside the recursive step: the
terminal NIFS, three boundary links, the post-fold accumulator, all fourteen
child-continuity shards, public pins, and all fourteen direct terminal-CE
programs.  Its soundness result contains only independently decoded semantic
facts.  The sampled PiRLC projection is coefficient-exact or the result names
the precise batch bad-root event.

The accumulator's raw parent source is tied to the decoded strict-PiDEC parent,
not merely to a verifier-normalized shape surrogate.  The required five shape
equalities are derived from exact generated terminal PiRLC shape pins.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalShellSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection
open Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles
open Nightstream.Implementation.R1CS.FPrimeFullHistorySumcheckArtifact

set_option maxRecDepth 1048576
set_option maxHeartbeats 8000000

/-- Exact row predicates owned by the fixed-profile terminal shell. -/
structure TerminalRows (assignment : Nat → Nat) : Prop where
  nifs : Nightstream.Assurance.FPrimeConcreteNifs.TerminalRows assignment
  runningLink : Satisfies FPrimeFullHistoryTerminalRunningLink.rows assignment
  parentLink : Satisfies
    FPrimeFullHistoryTerminalParentLink.rows assignment
  latestLink : Satisfies FPrimeFullHistoryTerminalLink.rows assignment
  accumulator : Satisfies FPrimeFullHistoryTerminalAccumulator.rows assignment
  continuity : Satisfies FPrimeFullHistoryTerminalContinuity.rows assignment
  publicPins : Satisfies FPrimeFullHistoryPublicPins.rows assignment
  terminalCe : Satisfies FPrimeFullHistoryTerminalCe.terminalCeRows assignment

/-- Five literal terminal PiDEC parent-shape pins, after the production
local-to-owner relabeling. -/
private def parentShapePins : List AffinePins.Pin :=
  [ .constant 2611276 54
  , .constant 2611277 18
  , .constant 2611278 54
  , .constant 2611279 257
  , .constant 2611280 257 ]

private theorem parentShapePins_member :
    ∀ pin ∈ parentShapePins,
      pin ∈ FPrimeFullHistoryPiRlcTerminalShape.pins := by
  native_decide

private theorem parentShapeColumns :
    Relabel.column FPrimeFullHistoryPiDec.terminalColumnMap 24391 = 2611276 ∧
    Relabel.column FPrimeFullHistoryPiDec.terminalColumnMap 24393 = 2611277 ∧
    Relabel.column FPrimeFullHistoryPiDec.terminalColumnMap 24395 = 2611278 ∧
    Relabel.column FPrimeFullHistoryPiDec.terminalColumnMap 24397 = 2611279 ∧
    Relabel.column FPrimeFullHistoryPiDec.terminalColumnMap 24399 = 2611280 := by
  native_decide

/-- Exact terminal affine rows pin every verifier-normalized parent-shape
constant to the corresponding strict-PiDEC parent wire. -/
theorem parentShapeAgrees
    {assignment : Nat → Nat}
    (affine : FPrimeFullHistoryAffineSound.Terminal.Holds assignment) :
    FPrimeFullHistoryParentCeSerialization.ShapeAgrees assignment := by
  have d := affine.piRlcShape (.constant 2611276 54)
    (parentShapePins_member _ (by simp [parentShapePins]))
  have kappa := affine.piRlcShape (.constant 2611277 18)
    (parentShapePins_member _ (by simp [parentShapePins]))
  have xRows := affine.piRlcShape (.constant 2611278 54)
    (parentShapePins_member _ (by simp [parentShapePins]))
  have xWidth := affine.piRlcShape (.constant 2611279 257)
    (parentShapePins_member _ (by simp [parentShapePins]))
  have mIn := affine.piRlcShape (.constant 2611280 257)
    (parentShapePins_member _ (by simp [parentShapePins]))
  simp only [AffinePins.Pin.Holds] at d kappa xRows xWidth mIn
  rcases parentShapeColumns with ⟨dColumn, kappaColumn, xRowsColumn,
    xWidthColumn, mInColumn⟩
  refine {
    commitmentD := ?_
    commitmentKappa := ?_
    xRows := ?_
    xWidth := ?_
    mIn := ?_ }
  · change assignment
      (Relabel.column FPrimeFullHistoryPiDec.terminalColumnMap 24391) = 54
    rw [dColumn]
    exact d
  · change assignment
      (Relabel.column FPrimeFullHistoryPiDec.terminalColumnMap 24393) = 18
    rw [kappaColumn]
    exact kappa
  · change assignment
      (Relabel.column FPrimeFullHistoryPiDec.terminalColumnMap 24395) = 54
    rw [xRowsColumn]
    exact xRows
  · change assignment
      (Relabel.column FPrimeFullHistoryPiDec.terminalColumnMap 24397) = 257
    rw [xWidthColumn]
    exact xWidth
  · change assignment
      (Relabel.column FPrimeFullHistoryPiDec.terminalColumnMap 24399) = 257
    rw [mInColumn]
    exact mIn

/-- Independent semantic conclusions for every exact terminal owner.  The
result carries no row-satisfaction proposition.  `terminalCe` ranges over the
artifact's exact fourteen-entry column-map census. -/
structure TerminalFacts (assignment : Nat → Nat) : Prop where
  nifs : Nightstream.Assurance.FPrimeConcreteNifs.TerminalArtifactAccepted
    assignment
  runningLink : FPrimeFullHistoryTerminalRunningLinkSound.Holds assignment
  parentLink : FPrimeFullHistoryTerminalParentLinkSound.Holds assignment
  latestLink : FPrimeFullHistoryTerminalLinkSound.Holds assignment
  accumulator : FPrimeFullHistoryTerminalAccumulatorSound.Facts assignment
  parentShape : FPrimeFullHistoryParentCeSerialization.ShapeAgrees assignment
  parentSourceDecoded :
    FPrimeFullHistoryTerminalAccumulator.parentCeClaimSourceColumns.map
        assignment =
      CeClaimDigestV2.noAdvPreimage
        (PiDecStrictCompiler.activeColumns FPrimeFullHistoryPiDec.layout)
        FPrimeFullHistoryPiDec.layout.extensionLimbs
        (FPrimeFullHistoryParentCeSerialization.decodedParent assignment)
  continuity : FPrimeFullHistoryTerminalContinuitySound.Holds assignment
  publicPins : FPrimeFullHistoryPublicPinsSound.Artifact.Facts assignment
  terminalCe : FPrimeFullHistoryTerminalCeSound.AllClaimsHold assignment

theorem terminalCe_child_count :
    FPrimeFullHistoryTerminalCe.columnMaps.length = 14 :=
  FPrimeFullHistoryTerminalCe.column_maps_length

/-- Soundness of the complete terminal shell.  The only non-deterministic
alternative retained by the theorem is the named nonzero-polynomial root
event from the terminal PiRLC projection batch. -/
theorem sound_or_badRoot
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (rows : TerminalRows assignment) :
    TerminalFacts assignment ∨
      Nightstream.SuperNeo.ProjectionCheck.BatchBadRoot
        ProjectionProgram.K.ops
        (ProjectionProgram.BatchIdentity terminalTraces assignment) := by
  have semantic :=
    Nightstream.Assurance.FPrimeConcreteNifs.terminal_rows_sound
      prime canonical one rows.nifs
  rcases Nightstream.Assurance.FPrimeConcreteNifs.terminal_semantic_sound_or_badRoot
      semantic with exact | bad
  · have shape := parentShapeAgrees exact.affine
    have accumulator := FPrimeFullHistoryTerminalAccumulatorSound.sound
      prime canonical one rows.accumulator
    have parentSourceDecoded :
        FPrimeFullHistoryTerminalAccumulator.parentCeClaimSourceColumns.map
            assignment =
          CeClaimDigestV2.noAdvPreimage
            (PiDecStrictCompiler.activeColumns FPrimeFullHistoryPiDec.layout)
            FPrimeFullHistoryPiDec.layout.extensionLimbs
            (FPrimeFullHistoryParentCeSerialization.decodedParent
              assignment) := by
      exact accumulator.parentClaimSource.trans
        (FPrimeFullHistoryParentCeSerialization.parentPreimage_eq_decoded shape)
    left
    exact {
      nifs := exact
      runningLink := FPrimeFullHistoryTerminalRunningLinkSound.sound canonical
        one rows.runningLink
      parentLink := FPrimeFullHistoryTerminalParentLinkSound.sound canonical
        one rows.parentLink
      latestLink := FPrimeFullHistoryTerminalLinkSound.sound canonical one
        rows.latestLink
      accumulator := accumulator
      parentShape := shape
      parentSourceDecoded := parentSourceDecoded
      continuity := FPrimeFullHistoryTerminalContinuitySound.sound canonical
        one rows.continuity
      publicPins := FPrimeFullHistoryPublicPinsSound.Artifact.sound canonical
        one rows.publicPins
      terminalCe := FPrimeFullHistoryTerminalCeSound.all_claims_sound prime
        canonical one rows.terminalCe }
  · exact Or.inr bad

/-- Independent compiler inputs for the terminal NIFS owner.  In particular,
the witness does not carry `TerminalSemanticAccepted` or generated-row
satisfaction: sampled projection acceptance is reconstructed from the native
projection executions during completeness. -/
structure NifsCompilerWitness
    (field : CanonicalU64Complete.FieldInverse)
    (assignment : Nat → Nat) where
  transcript : FPrimeFullHistoryTranscriptSound.TerminalTranscriptAccepted
    assignment
  affine : FPrimeFullHistoryAffineSound.Terminal.Holds assignment
  projection : ∀ trace ∈ terminalTraces, trace.ExecutionWitness assignment
  projectionGlue : ∀ pin ∈ terminalGluePins,
    AffinePins.Pin.Holds assignment pin
  feSumcheck : SumcheckChainSound.ExecutionWitness terminalFeMaps assignment
  ncSumcheck : SumcheckChainSound.ExecutionWitness terminalNcMaps assignment
  piDec : PiDecStrictSound.Exact.ExecutionWitness
    (Relabel.assignment FPrimeFullHistoryPiDec.terminalColumnMap assignment)
  authorityPiDec : PiDecStrictSound.Exact.ExecutionWitness
    (Relabel.assignment FPrimeFullHistoryPiDec.terminalCeColumnMap assignment)
  authorityTail :
    ∀ pin ∈ FPrimeFullHistoryPiCcsTerminalAuthorityTail.pins,
      AffinePins.Pin.Holds assignment pin
  residual : Nightstream.Assurance.FPrimeConcreteNifs.OwnersExecution field
    Nightstream.Assurance.FPrimeConcreteNifs.terminalResidualOwners assignment
  pointBinding : FPrimeFullHistoryPointBindingSound.TerminalHolds assignment

/-- Successful execution of the exact public-pin checked program. -/
abbrev PublicPinsExecution (assignment : Nat → Nat) :=
  CheckedProgram.ExecutionWitness FPrimeFullHistoryPublicPins.instructions
    assignment

theorem PublicPinsExecution.compiles
    {assignment : Nat → Nat}
    (execution : PublicPinsExecution assignment) :
    Satisfies FPrimeFullHistoryPublicPins.rows assignment := by
  exact CheckedProgram.ExecutionWitness.compiles execution
    FPrimeFullHistoryPublicPins.definitions_wellFormed
    FPrimeFullHistoryPublicPins.definitions_canonical (by native_decide)

/-- Compiler/execution witness for every terminal row owner.  It contains no
`Satisfies`, `AssignmentHolds`, `ChecksHold`, or aggregate accepted-conclusion
field. -/
structure CompilerWitness
    (field : CanonicalU64Complete.FieldInverse)
    (assignment : Nat → Nat) where
  nifs : NifsCompilerWitness field assignment
  runningLink : FPrimeFullHistoryTerminalRunningLinkSound.Holds assignment
  parentLink : FPrimeFullHistoryTerminalParentLinkSound.Holds assignment
  latestLink : FPrimeFullHistoryTerminalLinkSound.Holds assignment
  accumulator : FPrimeFullHistoryTerminalAccumulatorSound.CompilerWitness
    assignment
  continuity : FPrimeFullHistoryTerminalContinuitySound.Holds assignment
  publicPins : PublicPinsExecution assignment
  terminalCe : FPrimeFullHistoryTerminalCeSound.CompilerWitness assignment

/-- Exact CIR-COMPLETE constructor for the complete terminal shell. -/
theorem complete
    {field : CanonicalU64Complete.FieldInverse}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (witness : CompilerWitness field assignment) :
    TerminalRows assignment := by
  let nifsWitness :
      Nightstream.Assurance.FPrimeConcreteNifs.TerminalExecutionWitness
        field assignment := {
    transcript := witness.nifs.transcript
    affine := witness.nifs.affine
    projection := witness.nifs.projection
    projectionGlue := witness.nifs.projectionGlue
    feSumcheck := witness.nifs.feSumcheck
    ncSumcheck := witness.nifs.ncSumcheck
    piDec := witness.nifs.piDec
    pointBinding := witness.nifs.pointBinding
    terminalCe := witness.nifs.authorityPiDec
    authorityTail := witness.nifs.authorityTail
    residual := witness.nifs.residual }
  exact {
    nifs := Nightstream.Assurance.FPrimeConcreteNifs.terminal_rows_complete
      canonical one nifsWitness
    runningLink := FPrimeFullHistoryTerminalRunningLinkSound.complete canonical
      one witness.runningLink
    parentLink := FPrimeFullHistoryTerminalParentLinkSound.complete canonical
      one witness.parentLink
    latestLink := FPrimeFullHistoryTerminalLinkSound.complete canonical one
      witness.latestLink
    accumulator := FPrimeFullHistoryTerminalAccumulatorSound.complete canonical
      one witness.accumulator
    continuity := FPrimeFullHistoryTerminalContinuitySound.complete canonical
      one witness.continuity
    publicPins := witness.publicPins.compiles
    terminalCe := FPrimeFullHistoryTerminalCeSound.all_claims_complete
      witness.terminalCe }

end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalShellSound
