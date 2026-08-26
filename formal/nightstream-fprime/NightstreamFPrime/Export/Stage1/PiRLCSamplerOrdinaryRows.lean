import NightstreamFPrime.Export.Stage1.PiRLCSamplerRows
import NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestLane
import NightstreamFPrime.Layout.Stage1.PiRLCStarts

/-!
Owns the ordinary-row remainder of the compact PiRLC sampler package.

Poseidon2 entry/window rows use permutation invocations. `First54` recipe
rows use compact templates. The only ordinary rows are the 32 digest lanes
and the final fail-closed selector assertion in each scalar sampler.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryRows

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

def sourceCount : Nat := 17
def digestRoundCount : Nat := 8

def chainInterface : SamplerChain.Interface :=
  PiRLCSamplerRows.samplerInterface
    (logicalWidth := logicalWidth) (publicFits := publicFits)

def sourceInterface (source : Nat) : Sampler.Interface :=
  SamplerChain.childInterface
    (chainInterface (logicalWidth := logicalWidth) (publicFits := publicFits))
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart source

def windowInterface (source round : Nat) : DigestWindow.Interface :=
  Sampler.windowInterface
    (sourceInterface (logicalWidth := logicalWidth)
      (publicFits := publicFits) source)
    source
    (NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerSourceLogicalStart
      source)
    round

def laneInterface (source round : Nat) (lane : Fin 4) :
    DigestLane.Interface :=
  DigestWindow.laneInterface
    (windowInterface (logicalWidth := logicalWidth)
      (publicFits := publicFits) source round)
    (NightstreamFPrime.Layout.Stage1.PiRLCStarts.windowLogicalStart source round)
    lane

def laneInputs (source round : Nat) (lane : Fin 4) :
    NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestLane.InputsAffine
      (laneInterface (logicalWidth := logicalWidth) (publicFits := publicFits)
        source round lane)
      (NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestLaneLogicalStart
        source round lane.val) := by
  have windowInputs := NightstreamFPrime.Layout.PiRLC.v1_1.Sampler.windowInputs
    (sourceInterface (logicalWidth := logicalWidth)
      (publicFits := publicFits) source)
    source
    (NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerSourceLogicalStart
      source)
    round
  constructor
  simpa [laneInterface, windowInterface, DigestWindow.laneInterface,
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestLaneLogicalStart,
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.windowLogicalStart,
    DigestWindow.laneOffset] using
      windowInputs.initialState (DigestWindow.rateLane lane)

def laneConstraints (source round : Nat) (lane : Fin 4) : List Expr :=
  NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestLane.logicalConstraints
    (laneInterface (logicalWidth := logicalWidth) (publicFits := publicFits)
      source round lane)
    (NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestLaneLogicalStart
      source round lane.val)

def laneRows (source round : Nat) (lane : Fin 4) :
    List Rows.CompiledRow :=
  PiCCSArithmetic.compilePacket
    (NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestLaneRowStart
      source round lane.val)
    (NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestLaneFreshStart
      source round lane.val)
    (laneConstraints (logicalWidth := logicalWidth) (publicFits := publicFits)
      source round lane)

def windowRows (source round : Nat) : List Rows.CompiledRow :=
  (List.finRange 4).flatMap
    (laneRows (logicalWidth := logicalWidth) (publicFits := publicFits)
      source round)

def selectorFinalConstraint (source : Nat) : Expr :=
  First54.finalFull
      (NightstreamFPrime.Layout.Stage1.PiRLCStarts.selectorLogicalStart source) -
    1

def selectorFinalRows (source : Nat) : List Rows.CompiledRow :=
  PiCCSArithmetic.compilePacket
    (NightstreamFPrime.Layout.Stage1.PiRLCStarts.selectorRowStart source +
      41023)
    (NightstreamFPrime.Layout.Stage1.PiRLCStarts.selectorFreshStart source +
      34047)
    [selectorFinalConstraint source]

def sourceRows (source : Nat) : List Rows.CompiledRow :=
  (List.range digestRoundCount).flatMap
      (windowRows (logicalWidth := logicalWidth) (publicFits := publicFits)
        source) ++
    selectorFinalRows source

def rows : List Rows.CompiledRow :=
  (List.range sourceCount).flatMap
    (sourceRows (logicalWidth := logicalWidth) (publicFits := publicFits))

theorem laneRows_toR1CS (source round : Nat) (lane : Fin 4) :
    (laneRows (logicalWidth := logicalWidth) (publicFits := publicFits)
      source round lane).map Rows.CompiledRow.toR1CS =
      NightstreamFPrime.Layout.Stage1.Spartan.remapRows
        (R1CS.lowerConstraints
          (laneConstraints (logicalWidth := logicalWidth)
            (publicFits := publicFits) source round lane)
          (NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestLaneFreshStart
            source round lane.val)).rows := by
  exact PiCCSArithmetic.compilePacket_toR1CS _ _ _

theorem selectorFinalRows_toR1CS (source : Nat) :
    (selectorFinalRows source).map Rows.CompiledRow.toR1CS =
      NightstreamFPrime.Layout.Stage1.Spartan.remapRows
        (R1CS.lowerConstraints [selectorFinalConstraint source]
          (NightstreamFPrime.Layout.Stage1.PiRLCStarts.selectorFreshStart
            source + 34047)).rows := by
  exact PiCCSArithmetic.compilePacket_toR1CS _ _ _

@[simp] theorem laneRows_length (source round : Nat) (lane : Fin 4) :
    (laneRows (logicalWidth := logicalWidth) (publicFits := publicFits)
      source round lane).length = 406 := by
  rw [laneRows, PiCCSArithmetic.compilePacket_length]
  exact
    NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestLane.totalRowCount_eq
      (laneInterface (logicalWidth := logicalWidth) (publicFits := publicFits)
        source round lane)
      (NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestLaneLogicalStart
        source round lane.val)
      (laneInputs (logicalWidth := logicalWidth) (publicFits := publicFits)
        source round lane)

@[simp] theorem windowRows_length (source round : Nat) :
    (windowRows (logicalWidth := logicalWidth) (publicFits := publicFits)
      source round).length = 1624 := by
  simp [windowRows]

@[simp] theorem selectorFinalRows_length (source : Nat) :
    (selectorFinalRows source).length = 1 := by
  rw [selectorFinalRows, PiCCSArithmetic.compilePacket_length]
  rfl

@[simp] theorem sourceRows_length (source : Nat) :
    (sourceRows (logicalWidth := logicalWidth) (publicFits := publicFits)
      source).length = 12993 := by
  simp [sourceRows, digestRoundCount]

@[simp] theorem rows_length :
    (rows (logicalWidth := logicalWidth) (publicFits := publicFits)).length =
      220881 := by
  simp [rows, sourceCount]

theorem laneRows_subset_rows (source round : Nat) (lane : Fin 4)
    (sourceLt : source < sourceCount) (roundLt : round < digestRoundCount) :
    ∀ row ∈ laneRows (logicalWidth := logicalWidth)
        (publicFits := publicFits) source round lane,
      row ∈ rows (logicalWidth := logicalWidth) (publicFits := publicFits) := by
  intro row member
  unfold rows
  apply List.mem_flatMap.mpr
  refine ⟨source, List.mem_range.mpr sourceLt, ?_⟩
  unfold sourceRows
  apply List.mem_append_left
  apply List.mem_flatMap.mpr
  refine ⟨round, List.mem_range.mpr roundLt, ?_⟩
  unfold windowRows
  apply List.mem_flatMap.mpr
  exact ⟨lane, by simp, member⟩

theorem selectorFinalRows_subset_rows (source : Nat)
    (sourceLt : source < sourceCount) :
    ∀ row ∈ selectorFinalRows source,
      row ∈ rows (logicalWidth := logicalWidth) (publicFits := publicFits) := by
  intro row member
  unfold rows
  apply List.mem_flatMap.mpr
  refine ⟨source, List.mem_range.mpr sourceLt, ?_⟩
  unfold sourceRows
  exact List.mem_append_right _ member

/-- Satisfaction of the full ordinary sampler packet implies one exact
digest-lane specification under the Stage 1 source-column pullback. -/
theorem rows_imply_laneSpec (source round : Nat) (lane : Fin 4)
    (sourceLt : source < sourceCount) (roundLt : round < digestRoundCount)
    (env : Env)
    (assumptions : DigestLane.Assumptions
      (laneInterface (logicalWidth := logicalWidth) (publicFits := publicFits)
        source round lane)
      (NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestLaneLogicalStart
        source round lane.val)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env))
    (holds : R1CS.RowsHold env
      ((rows (logicalWidth := logicalWidth) (publicFits := publicFits)).map
        Rows.CompiledRow.toR1CS)) :
    DigestLane.SpecHolds
      (laneInterface (logicalWidth := logicalWidth) (publicFits := publicFits)
        source round lane)
      (NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestLaneLogicalStart
        source round lane.val)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  have laneHolds : R1CS.RowsHold env
      ((laneRows (logicalWidth := logicalWidth) (publicFits := publicFits)
        source round lane).map Rows.CompiledRow.toR1CS) := by
    intro row member
    rcases List.mem_map.mp member with ⟨compiled, compiledMember, rfl⟩
    apply holds
    apply List.mem_map.mpr
    exact ⟨compiled,
      laneRows_subset_rows source round lane sourceLt roundLt compiled
        compiledMember,
      rfl⟩
  have logical := PiCCSArithmetic.compilePacket_sound
    (NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestLaneRowStart
      source round lane.val)
    (NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestLaneFreshStart
      source round lane.val)
    (laneConstraints (logicalWidth := logicalWidth) (publicFits := publicFits)
      source round lane)
    env laneHolds
  apply DigestLane.soundness
    (laneInterface (logicalWidth := logicalWidth) (publicFits := publicFits)
      source round lane)
    (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
    (NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestLaneLogicalStart
      source round lane.val)
    assumptions
  apply holdsFlat_implies_holds
  change ConstraintsHold
    (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
    (laneConstraints (logicalWidth := logicalWidth) (publicFits := publicFits)
      source round lane)
  exact logical

/-- Satisfaction of the full ordinary sampler packet enforces the final
fail-closed `First54` full-slot assertion for one bounded scalar source. -/
theorem rows_imply_selectorFull (source : Nat) (sourceLt : source < sourceCount)
    (env : Env)
    (holds : R1CS.RowsHold env
      ((rows (logicalWidth := logicalWidth) (publicFits := publicFits)).map
        Rows.CompiledRow.toR1CS)) :
    (First54.finalFull
      (NightstreamFPrime.Layout.Stage1.PiRLCStarts.selectorLogicalStart source)
      ).eval (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) = 1 := by
  have selectorHolds : R1CS.RowsHold env
      ((selectorFinalRows source).map Rows.CompiledRow.toR1CS) := by
    intro row member
    rcases List.mem_map.mp member with ⟨compiled, compiledMember, rfl⟩
    apply holds
    apply List.mem_map.mpr
    exact ⟨compiled,
      selectorFinalRows_subset_rows source sourceLt compiled compiledMember,
      rfl⟩
  have logical := PiCCSArithmetic.compilePacket_sound
    (NightstreamFPrime.Layout.Stage1.PiRLCStarts.selectorRowStart source +
      41023)
    (NightstreamFPrime.Layout.Stage1.PiRLCStarts.selectorFreshStart source +
      34047)
    [selectorFinalConstraint source] env selectorHolds
  have finalZero := logical (selectorFinalConstraint source) (by simp)
  unfold selectorFinalConstraint at finalZero
  rw [Expr.eval_sub] at finalZero
  exact sub_eq_zero.mp finalZero

end NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryRows
