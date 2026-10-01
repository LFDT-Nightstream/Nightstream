import NightstreamFPrime.Export.Stage1.ActualPiRLCStates
import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryDirectPlanSemantics

/-! Accepted ordinary rows determine the wide reduction of the actual retained
transcript draw. The checked words feed the product plan directly. This proof
applies to arbitrary assignments; canonical witness encoding is not a premise. -/

namespace NightstreamFPrime.Export.Stage1.ActualPiRLCSampling

open NightstreamFPrime.Circuit NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1 NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1 NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Spec NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open PiRLCSamplerOrdinaryDirectPlan (Location resolvedEnv poseidonGeometry)
open PiRLCSamplerOrdinaryRetainedBlocks (sourceCount)

variable {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
  {relationLogicalWidth : Nat}
  {relationPublicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth relationLogicalWidth}

def challenge
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (source : Fin sourceCount) : RingF :=
  fun lane => ((Location.word source lane).form geometry).eval assignment - 2

private theorem decoded_location
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (location : Location) :
    (Spartan.pullback (resolvedEnv geometry assignment)) location.sourceColumn =
      (location.form geometry).eval assignment := by
  unfold Spartan.pullback resolvedEnv PiRLCSamplerOrdinaryDirectPlan.resolvedForm
    PiRLCSamplerOrdinaryDirectPlan.classifyTarget
  rw [Spartan.spartanToSource_sourceToSpartan _ location.sourceColumn_lt]
  cases location with
  | poseidon source lane =>
      simp only [PiRLCSamplerOrdinaryDirectPlan.classifySource_poseidonEntry]
  | logical source position =>
      change (match PiRLCSamplerOrdinaryDirectPlan.classifySource
          (PiRLCSamplerOrdinaryRetainedBlocks.logicalSource source position) with
        | none => SparseForm.empty | some found => found.form geometry).eval assignment = _
      rw [PiRLCSamplerOrdinaryDirectPlan.classifySource_logical]
  | fresh source position =>
      change (match PiRLCSamplerOrdinaryDirectPlan.classifySource
          (PiRLCSamplerOrdinaryRetainedBlocks.freshSource source position) with
        | none => SparseForm.empty | some found => found.form geometry).eval assignment = _
      rw [PiRLCSamplerOrdinaryDirectPlan.classifySource_fresh]
  | word source position =>
      change (match PiRLCSamplerOrdinaryDirectPlan.classifySource
          (PiRLCStarts.challengeWordStart source.val + position.val) with
        | none => SparseForm.empty | some found => found.form geometry).eval assignment = _
      rw [PiRLCSamplerOrdinaryDirectPlan.classifySource_word]

private theorem range_inputs (source : Fin sourceCount) :
    WideReduction.Assumptions
      (PiRLCSamplerOrdinaryRows.rangeInterface
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits) source.val)
      (PiRLCStarts.rangeLogicalStart source.val) := by
  intro lane
  rw [PiRLCSamplerOrdinaryDirectSource.rangeSource_eq_var]
  change PiRLCStarts.samplerSourceLogicalStart source.val + 1080 + lane.val <
    PiRLCStarts.samplerSourceLogicalStart source.val + 1096
  rw [Nat.add_assoc]
  exact Nat.add_lt_add_left (by have bound : lane.val < 4 := lane.isLt; omega) _

private theorem draw_eq_verifier
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (semantics : PiRLCSamplerPoseidonPreservation.CanonicalSemantics
      (poseidonGeometry geometry) assignment)
    (source : Fin sourceCount) :
    WideReduction.drawOf
        (WideReduction.Program.coreInterface (PiRLCSamplerOrdinaryRows.rangeInterface
          (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits) source.val)
          (PiRLCStarts.rangeLogicalStart source.val))
        (Spartan.pullback (resolvedEnv geometry assignment))
        (WideReduction.Program.coreOffset (PiRLCStarts.rangeLogicalStart source.val)) =
      Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.drawAt
        (ActualPiRLCStates.initialState (poseidonGeometry geometry) assignment) source.val := by
  funext lane
  simp only [WideReduction.drawOf, WideReduction.Program.coreInterface,
    PiRLCSamplerOrdinaryDirectSource.rangeSource_eq_var, Expr.eval]
  change (Spartan.pullback (resolvedEnv geometry assignment))
    (Location.poseidon source lane).sourceColumn = _
  rw [decoded_location]
  have stateEq := congrArg (fun state : Poseidon2.State => state.getD lane.val 0)
    (ActualPiRLCStates.entry_eq_verifier (poseidonGeometry geometry) assignment semantics source)
  rw [ActualPiRLCStates.state, List.getD_eq_get _ _
    ⟨lane.val, by have bounded : lane.val < 4 := lane.isLt; simp only [List.length_ofFn]; omega⟩] at stateEq
  simpa only [List.get_ofFn, Fin.cast_mk, Sampler.rateLane, Location.form, Location.poseidonInvocation,
    PiRLCSamplerPoseidonPreservation.outputValue, SparseLayer.evalState,
    Transcript.PiRlcSampler.enterScalar,
    Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.drawAt,
    Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.block] using stateEq

/-- Every retained coefficient is forced by the exact verifier draw. -/
theorem rowsZero_implies_challenge
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiRLCSamplerOrdinaryRetainedGeometry.oneColumn geometry) = 1)
    (rows : (PiRLCSamplerOrdinaryDirectPlan.plan relation geometry).RowsZero assignment)
    (semantics : PiRLCSamplerPoseidonPreservation.CanonicalSemantics
      (poseidonGeometry geometry) assignment)
    (source : Fin sourceCount) :
    Transcript.PiRlcSampler.sampleRingChallenge
        (ActualPiRLCStates.initialState (poseidonGeometry geometry) assignment) source.val =
      challenge geometry assignment source := by
  let env := resolvedEnv geometry assignment
  have ordinary := (PiRLCSamplerOrdinaryDirectPlan.rowsZero_iff_rowsHold
    relation geometry assignment one).mp rows
  change R1CS.RowsHold env ((PiRLCSamplerOrdinaryRows.rows
    (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)).map Rows.CompiledRow.toR1CS) at ordinary
  have packet := PiRLCSamplerOrdinaryRows.rows_imply_sourceRows
    (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits) source env (@ordinary)
  rw [PiRLCSamplerOrdinaryRows.sourceRows, List.map_append, R1CS.rowsHold_append] at packet
  have decoded := PiRLCSamplerOrdinaryRows.rangeRows_imply_spec
    (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
    source.val env (range_inputs source) packet.1
  have words := PiRLCSamplerOrdinaryRows.wordRows_imply_spec source.val env packet.2
  have draw := draw_eq_verifier (relationLogicalWidth := relationLogicalWidth)
    (relationPublicFits := relationPublicFits) geometry assignment semantics source
  funext position
  have digit := decoded position
  change WideReduction.digitValue (Spartan.pullback env)
      (WideReduction.Program.coreOffset (PiRLCStarts.rangeLogicalStart source.val)) position.val = _ at digit
  rw [draw] at digit
  have word := words position
  change (Spartan.pullback env) (Location.word source position).sourceColumn = _ at word
  rw [decoded_location] at word
  change _ = (WideReduction.Program.outputWord _ position).eval (Spartan.pullback env) at word
  rw [WideReduction.Program.outputWord_eval, digit] at word
  change _ = ((Location.word source position).form geometry).eval assignment - 2
  rw [word]
  exact (SamplerChain.centered_digit _).symm

/-- All 17 challenges and the final state agree with the total verifier batch. -/
theorem rowsZero_implies_batch
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiRLCSamplerOrdinaryRetainedGeometry.oneColumn geometry) = 1)
    (rows : (PiRLCSamplerOrdinaryDirectPlan.plan relation geometry).RowsZero assignment)
    (semantics : PiRLCSamplerPoseidonPreservation.CanonicalSemantics
      (poseidonGeometry geometry) assignment) :
    Transcript.PiRlcSampler.piRlcChallengesWithState
        (ActualPiRLCStates.initialState (poseidonGeometry geometry) assignment) sourceCount =
      ⟨challenge geometry assignment,
        ActualPiRLCStates.state (poseidonGeometry geometry) assignment
          ⟨16, by decide⟩ ⟨1, by decide⟩⟩ := by
  have challenges : (fun source : Fin sourceCount => Transcript.PiRlcSampler.sampleRingChallenge
      (ActualPiRLCStates.initialState (poseidonGeometry geometry) assignment) source.val) =
        challenge geometry assignment := by
    funext source
    exact rowsZero_implies_challenge relation geometry assignment one rows semantics source
  change Transcript.PiRlcSampler.Batch.mk
    (Transcript.PiRlcSampler.piRlcChallengesWithState
      (ActualPiRLCStates.initialState (poseidonGeometry geometry) assignment) sourceCount).challenges
    (Transcript.PiRlcSampler.piRlcChallengesWithState
      (ActualPiRLCStates.initialState (poseidonGeometry geometry) assignment) sourceCount).finalState = _
  apply congrArg₂ Transcript.PiRlcSampler.Batch.mk
  · exact (Transcript.PiRlcSampler.piRlcChallengesWithState_challenges _ _).trans challenges
  · rw [Transcript.PiRlcSampler.piRlcChallengesWithState_finalState,
      ActualPiRLCStates.final_eq_stateAt _ _ semantics]
    rfl

end NightstreamFPrime.Export.Stage1.ActualPiRLCSampling
