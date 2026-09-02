import NightstreamFPrime.Export.Stage1.PermutationPlan
import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryDirectPlan
import NightstreamFPrime.Export.Stage1.PiRLCSamplerPoseidonPreservation

/-!
Owns the exact source-column custody bridge from the retained sampler
Poseidon2 suffix to the canonical PiRLC sampler ordinary rows.

This module does not add rows or retained values.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerRetainedCustody

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-- The sampler suffix of the package invocation list uses the exact
Lean-authored random-access witness-start schedule. -/
theorem laterWitnessStart_sampler
    (current : Fin (PiRLCSamplerInvocations.sourceCount *
      PermutationPlan.samplerStepsPerSource)) :
    PoseidonRetainedBlock.laterWitnessStart
        ⟨7604 + current.val, by
          rw [PoseidonRetainedBlock.laterInvocationCount_eq]
          have currentLt := current.isLt
          norm_num [PiRLCSamplerInvocations.sourceCount,
            PermutationPlan.samplerStepsPerSource] at currentLt ⊢
          omega⟩ =
      PermutationPlan.samplerWitnessStartAt current := by
  unfold PoseidonRetainedBlock.laterWitnessStart
    PoseidonRetainedBlock.basePackage PerApplicationPackage.basePackage
  simp only [List.get_eq_getElem, Data.circuitPackage_permutationInvocations,
    Data.components_permutationInvocations, Data.permutationInvocations_eq]
  rw [List.getElem_append_right]
  · have prefixLength :
        (PiCCSInvocations.invocations Data.logicalWidth
          Data.publicFits).length = 7604 :=
      PiCCSInvocations.invocations_length Data.logicalWidth Data.publicFits
    have offsetEq :
        7604 + current.val -
            (PiCCSInvocations.invocations Data.logicalWidth
              Data.publicFits).length = current.val := by
      rw [prefixLength]
      omega
    have samplerBound : current.val <
        (PiRLCSamplerInvocations.invocations
          (logicalWidth := Data.logicalWidth)
          (publicFits := Data.publicFits)).length := by
      rw [PiRLCSamplerInvocations.invocations_length]
      have currentLt := current.isLt
      norm_num [PiRLCSamplerInvocations.sourceCount,
        PermutationPlan.samplerStepsPerSource] at currentLt ⊢
      exact currentLt
    have materialized := PermutationPlan.samplerWitnessStartAt_materializes
    have point :
        (List.ofFn PermutationPlan.samplerWitnessStartAt)[current.val]? =
          ((PiRLCSamplerInvocations.invocations
            (logicalWidth := Data.logicalWidth)
            (publicFits := Data.publicFits)).map
              (fun invocation => invocation.witnessStart))[current.val]? := by
      exact congrArg (fun values => values[current.val]?) materialized
    have samplerPoint :
        ((PiRLCSamplerInvocations.invocations
          (logicalWidth := Data.logicalWidth)
          (publicFits := Data.publicFits)).get
            ⟨current.val, samplerBound⟩).witnessStart =
          PermutationPlan.samplerWitnessStartAt current := by
      apply Option.some.inj
      calc
        some ((PiRLCSamplerInvocations.invocations
            (logicalWidth := Data.logicalWidth)
            (publicFits := Data.publicFits))[current.val].witnessStart) =
            ((PiRLCSamplerInvocations.invocations
              (logicalWidth := Data.logicalWidth)
              (publicFits := Data.publicFits)).map
                (fun invocation => invocation.witnessStart))[current.val]? := by
          rw [List.getElem?_map, List.getElem?_eq_getElem samplerBound]
          rfl
        _ = (List.ofFn PermutationPlan.samplerWitnessStartAt)[current.val]? :=
          point.symm
        _ = some (PermutationPlan.samplerWitnessStartAt current) := by
          simp only [List.getElem?_ofFn]
          split
          · congr 2
          · rename_i outside
            exfalso
            apply outside
            simpa [PermutationPlan.samplerStepsPerSource] using current.isLt
    let leftIndex : Fin
        (PiRLCSamplerInvocations.invocations
          (logicalWidth := Data.logicalWidth)
          (publicFits := Data.publicFits)).length :=
      ⟨7604 + current.val -
          (PiCCSInvocations.invocations Data.logicalWidth
            Data.publicFits).length,
        by omega⟩
    let rightIndex : Fin
        (PiRLCSamplerInvocations.invocations
          (logicalWidth := Data.logicalWidth)
          (publicFits := Data.publicFits)).length :=
      ⟨current.val, samplerBound⟩
    have indexEq : leftIndex = rightIndex := by
      apply Fin.ext
      exact offsetEq
    change ((PiRLCSamplerInvocations.invocations
      (logicalWidth := Data.logicalWidth)
      (publicFits := Data.publicFits)).get leftIndex).witnessStart = _
    rw [indexEq]
    exact samplerPoint
  · rw [PiCCSInvocations.invocations_length]
    omega

/-- Any exact source-classifier result lifts through the canonical Spartan
inverse to the corresponding retained form. -/
theorem resolvedForm_of_source
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) {source : Nat}
    {location : PiRLCSamplerOrdinaryDirectPlan.Location}
    (sourceBound : source < Spartan.SourceColumnCount)
    (found : PiRLCSamplerOrdinaryDirectPlan.classifySource source =
      some location) :
    PiRLCSamplerOrdinaryDirectPlan.resolvedForm geometry
        (Spartan.sourceToSpartan source) =
      location.form geometry := by
  unfold PiRLCSamplerOrdinaryDirectPlan.resolvedForm
    PiRLCSamplerOrdinaryDirectPlan.classifyTarget
  rw [Spartan.spartanToSource_sourceToSpartan source sourceBound]
  change (match PiRLCSamplerOrdinaryDirectPlan.classifySource source with
    | none => SparseForm.empty
    | some selected => selected.form geometry) = location.form geometry
  rw [found]

def stateOutputOffset : Nat := 584

/-- One of the nine complete eight-lane permutation outputs owned by each
scalar sampler. Step zero is the domain-entry permutation; steps one through
eight are the digest-window permutations. -/
structure StateLocation where
  source : Fin PiRLCSamplerPoseidonPlan.sourceCount
  step : Fin PiRLCSamplerPoseidonPlan.invocationsPerSource
  lane : Fin Spec.Poseidon2.width

namespace StateLocation

private theorem eq_of_fields {left right : StateLocation}
    (source : left.source = right.source)
    (step : left.step = right.step) (lane : left.lane = right.lane) :
    left = right := by
  cases left with
  | mk leftSource leftStep leftLane =>
      cases right with
      | mk rightSource rightStep rightLane =>
          cases source
          cases step
          cases lane
          rfl

def sourceColumn (location : StateLocation) : Nat :=
  PiRLCStarts.samplerLogicalStart +
    location.source.val * Sampler.logicalPrivateCount + stateOutputOffset +
      location.step.val * DigestWindow.logicalPrivateCount + location.lane.val

def form
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (location : StateLocation) : SparseForm logicalWidth :=
  (PiRLCSamplerPoseidonPlan.interface
    (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry)).output
      (PiRLCSamplerPoseidonPlan.invocation location.source location.step)
      location.lane

theorem sourceColumn_lt (location : StateLocation) :
    location.sourceColumn < Spartan.SourceColumnCount := by
  have sourceLt := location.source.isLt
  have stepLt := location.step.isLt
  have laneLt := location.lane.isLt
  rw [Spartan.sourceColumnCount_eq]
  norm_num [sourceColumn, stateOutputOffset,
    PiRLCSamplerPoseidonPlan.sourceCount,
    PiRLCSamplerPoseidonPlan.invocationsPerSource, Spec.Poseidon2.width,
    Sampler.logicalPrivateCount, DigestWindow.logicalPrivateCount,
    PiRLCStarts.samplerLogicalStart, PiRLCStarts.phaseLogicalStart,
    PiRLCInputs.phaseOffset,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset] at sourceLt stepLt laneLt ⊢
  omega

end StateLocation

private def stateSourceIndex (column : Nat) : Nat :=
  (column - PiRLCStarts.samplerLogicalStart) / Sampler.logicalPrivateCount

private def stateSourceOffset (column : Nat) : Nat :=
  (column - PiRLCStarts.samplerLogicalStart) % Sampler.logicalPrivateCount

private def stateStepIndex (column : Nat) : Nat :=
  (stateSourceOffset column - stateOutputOffset) /
    DigestWindow.logicalPrivateCount

private def stateLaneIndex (column : Nat) : Nat :=
  (stateSourceOffset column - stateOutputOffset) %
    DigestWindow.logicalPrivateCount

private def stateCandidate (column : Nat) : StateLocation where
  source := ⟨stateSourceIndex column % PiRLCSamplerPoseidonPlan.sourceCount,
    Nat.mod_lt _ (by decide)⟩
  step := ⟨stateStepIndex column %
      PiRLCSamplerPoseidonPlan.invocationsPerSource,
    Nat.mod_lt _ (by decide)⟩
  lane := ⟨stateLaneIndex column % Spec.Poseidon2.width,
    Nat.mod_lt _ (by decide)⟩

private def exactStateCandidate (column : Nat) (candidate : StateLocation) :
    Option StateLocation :=
  if candidate.sourceColumn = column then some candidate else none

/-- Constant-time, fail-closed classifier for all sampler state outputs. -/
def classifyStateSource (column : Nat) : Option StateLocation :=
  exactStateCandidate column (stateCandidate column)

private theorem quotient_at (outer offset stride : Nat)
    (stridePositive : 0 < stride) (offsetLt : offset < stride) :
    (outer * stride + offset) / stride = outer := by
  rw [Nat.mul_comm outer stride, Nat.mul_add_div stridePositive]
  rw [Nat.div_eq_of_lt offsetLt, Nat.add_zero]

private theorem remainder_at (outer offset stride : Nat)
    (offsetLt : offset < stride) :
    (outer * stride + offset) % stride = offset := by
  exact Nat.mul_add_mod_of_lt offsetLt

private theorem stateSourceIndex_at (location : StateLocation) :
    stateSourceIndex location.sourceColumn = location.source.val := by
  let withinSource := stateOutputOffset +
    location.step.val * DigestWindow.logicalPrivateCount + location.lane.val
  have withinSourceLt : withinSource < Sampler.logicalPrivateCount := by
    have stepLt := location.step.isLt
    have laneLt := location.lane.isLt
    norm_num [withinSource, stateOutputOffset,
      PiRLCSamplerPoseidonPlan.invocationsPerSource, Spec.Poseidon2.width,
      Sampler.logicalPrivateCount, DigestWindow.logicalPrivateCount] at stepLt laneLt ⊢
    omega
  unfold stateSourceIndex StateLocation.sourceColumn
  rw [show
      PiRLCStarts.samplerLogicalStart +
          location.source.val * Sampler.logicalPrivateCount +
          stateOutputOffset +
          location.step.val * DigestWindow.logicalPrivateCount +
          location.lane.val - PiRLCStarts.samplerLogicalStart =
        location.source.val * Sampler.logicalPrivateCount + withinSource by
    dsimp [withinSource]
    omega]
  exact quotient_at location.source.val withinSource
    Sampler.logicalPrivateCount (by norm_num [Sampler.logicalPrivateCount])
    withinSourceLt

private theorem stateSourceOffset_at (location : StateLocation) :
    stateSourceOffset location.sourceColumn =
      stateOutputOffset +
        location.step.val * DigestWindow.logicalPrivateCount +
          location.lane.val := by
  let withinSource := stateOutputOffset +
    location.step.val * DigestWindow.logicalPrivateCount + location.lane.val
  have withinSourceLt : withinSource < Sampler.logicalPrivateCount := by
    have stepLt := location.step.isLt
    have laneLt := location.lane.isLt
    norm_num [withinSource, stateOutputOffset,
      PiRLCSamplerPoseidonPlan.invocationsPerSource, Spec.Poseidon2.width,
      Sampler.logicalPrivateCount, DigestWindow.logicalPrivateCount] at stepLt laneLt ⊢
    omega
  unfold stateSourceOffset StateLocation.sourceColumn
  rw [show
      PiRLCStarts.samplerLogicalStart +
          location.source.val * Sampler.logicalPrivateCount +
          stateOutputOffset +
          location.step.val * DigestWindow.logicalPrivateCount +
          location.lane.val - PiRLCStarts.samplerLogicalStart =
        location.source.val * Sampler.logicalPrivateCount + withinSource by
    dsimp [withinSource]
    omega]
  exact remainder_at location.source.val withinSource
    Sampler.logicalPrivateCount withinSourceLt

private theorem stateStepIndex_at (location : StateLocation) :
    stateStepIndex location.sourceColumn = location.step.val := by
  unfold stateStepIndex
  rw [stateSourceOffset_at]
  rw [show
      stateOutputOffset +
          location.step.val * DigestWindow.logicalPrivateCount +
          location.lane.val - stateOutputOffset =
        location.step.val * DigestWindow.logicalPrivateCount +
          location.lane.val by omega]
  exact quotient_at location.step.val location.lane.val
    DigestWindow.logicalPrivateCount
    (by norm_num [DigestWindow.logicalPrivateCount])
    (lt_trans location.lane.isLt (by
      norm_num [Spec.Poseidon2.width, DigestWindow.logicalPrivateCount]))

private theorem stateLaneIndex_at (location : StateLocation) :
    stateLaneIndex location.sourceColumn = location.lane.val := by
  unfold stateLaneIndex
  rw [stateSourceOffset_at]
  rw [show
      stateOutputOffset +
          location.step.val * DigestWindow.logicalPrivateCount +
          location.lane.val - stateOutputOffset =
        location.step.val * DigestWindow.logicalPrivateCount +
          location.lane.val by omega]
  exact remainder_at location.step.val location.lane.val
    DigestWindow.logicalPrivateCount
    (lt_trans location.lane.isLt (by
      norm_num [Spec.Poseidon2.width, DigestWindow.logicalPrivateCount]))

private theorem stateCandidate_source (location : StateLocation) :
    stateCandidate location.sourceColumn = location := by
  apply StateLocation.eq_of_fields
  · apply Fin.ext
    change stateSourceIndex location.sourceColumn %
      PiRLCSamplerPoseidonPlan.sourceCount = location.source.val
    exact (congrArg
      (fun value => value % PiRLCSamplerPoseidonPlan.sourceCount)
      (stateSourceIndex_at location)).trans
        (Nat.mod_eq_of_lt location.source.isLt)
  · apply Fin.ext
    change stateStepIndex location.sourceColumn %
      PiRLCSamplerPoseidonPlan.invocationsPerSource = location.step.val
    exact (congrArg
      (fun value => value % PiRLCSamplerPoseidonPlan.invocationsPerSource)
      (stateStepIndex_at location)).trans
        (Nat.mod_eq_of_lt location.step.isLt)
  · apply Fin.ext
    change stateLaneIndex location.sourceColumn % Spec.Poseidon2.width =
      location.lane.val
    exact (congrArg (fun value => value % Spec.Poseidon2.width)
      (stateLaneIndex_at location)).trans
        (Nat.mod_eq_of_lt location.lane.isLt)

private theorem StateLocation.sourceColumn_injective
    {left right : StateLocation} (same : left.sourceColumn = right.sourceColumn) :
    left = right := by
  calc
    left = stateCandidate left.sourceColumn := (stateCandidate_source left).symm
    _ = stateCandidate right.sourceColumn := congrArg stateCandidate same
    _ = right := stateCandidate_source right

/-- Every canonical state output is accepted by the exact classifier. -/
theorem classifyStateSource_source (location : StateLocation) :
    classifyStateSource location.sourceColumn = some location := by
  unfold classifyStateSource
  rw [stateCandidate_source]
  simp [exactStateCandidate]

/-- Every successful state classification owns the exact requested source
column. -/
theorem classifyStateSource_sound {column : Nat} {location : StateLocation}
    (found : classifyStateSource column = some location) :
    location.sourceColumn = column := by
  unfold classifyStateSource exactStateCandidate at found
  split at found
  · rename_i owns
    have same := Option.some.inj found
    rw [← same]
    exact owns
  · cases found

def classifyStateTarget (column : Nat) : Option StateLocation :=
  match Spartan.spartanToSource column with
  | none => none
  | some source => classifyStateSource source

/-- The Spartan image of every canonical state output is accepted exactly. -/
theorem classifyStateTarget_source (location : StateLocation) :
    classifyStateTarget (Spartan.sourceToSpartan location.sourceColumn) =
      some location := by
  unfold classifyStateTarget
  rw [Spartan.spartanToSource_sourceToSpartan location.sourceColumn
    location.sourceColumn_lt]
  exact classifyStateSource_source location

private def ordinaryStateLocation
    (descriptor : PiRLCSamplerOrdinaryRetainedBlocks.Lane)
    (lane : Fin 4) : StateLocation where
  source := descriptor.source
  step := ⟨descriptor.round.val,
    lt_trans descriptor.round.isLt (by decide)⟩
  lane := DigestWindow.rateLane lane

private theorem ordinaryStateLocation_sourceColumn
    (descriptor : PiRLCSamplerOrdinaryRetainedBlocks.Lane)
    (lane : Fin 4) :
    (ordinaryStateLocation descriptor lane).sourceColumn =
      PiRLCSamplerOrdinaryDirectSource.poseidonSource descriptor.source.val
        descriptor.round.val lane := by
  unfold ordinaryStateLocation StateLocation.sourceColumn stateOutputOffset
  change PiRLCStarts.samplerLogicalStart +
      descriptor.source.val * Sampler.logicalPrivateCount + 584 +
        descriptor.round.val * DigestWindow.logicalPrivateCount + lane.val =
    PiRLCSamplerOrdinaryDirectSource.poseidonSource descriptor.source.val
      descriptor.round.val lane
  by_cases zero : descriptor.round.val = 0
  · rw [zero]
    simp [PiRLCSamplerOrdinaryDirectSource.poseidonSource,
      PiRLCStarts.samplerSourceLogicalStart, Sampler.logicalPrivateCount,
      DigestWindow.logicalPrivateCount, DigestWindow.rateLane]
  · obtain ⟨previous, roundEq⟩ := Nat.exists_eq_succ_of_ne_zero zero
    rw [roundEq]
    simp [PiRLCSamplerOrdinaryDirectSource.poseidonSource,
      PiRLCStarts.samplerSourceLogicalStart, Sampler.windowOffset,
      Sampler.windowBase, SamplerChain.sourceOffset,
      DigestWindow.permutationOffset, Sampler.logicalPrivateCount,
      Sampler.entryPrivateCount, DigestWindow.logicalPrivateCount,
      DigestLane.logicalPrivateCount, DigestWindow.rateLane]
    omega

/-- Semantic view for the complete sampler: exact retained forms own every
sampler-ordinary source, while all other columns retain the canonical Stage 1
transition view needed by the nested First54 relation. -/
def semanticEnv
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment Spec.F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → Spec.F) : Env :=
  fun column =>
    match PiRLCSamplerOrdinaryDirectPlan.classifyTarget column with
    | some _ =>
        PiRLCSamplerOrdinaryDirectPlan.resolvedEnv geometry assignment column
    | none =>
        match classifyStateTarget column with
        | some location => (location.form geometry).eval assignment
        | none => RunningTransitionDirectPlan.transitionEnv program base column

private theorem ordinaryLocation_sourceColumn_ge
    (location : PiRLCSamplerOrdinaryDirectPlan.Location) :
    PiRLCStarts.samplerLogicalStart ≤ location.sourceColumn := by
  cases location with
  | poseidon descriptor sourceLane =>
      rcases descriptor with ⟨source, round, lane⟩
      cases roundValue : round.val with
      | zero =>
          simp [PiRLCSamplerOrdinaryDirectPlan.Location.sourceColumn,
            PiRLCSamplerOrdinaryDirectSource.poseidonSource, roundValue,
            PiRLCStarts.samplerSourceLogicalStart]
          omega
      | succ previous =>
          simp [PiRLCSamplerOrdinaryDirectPlan.Location.sourceColumn,
            PiRLCSamplerOrdinaryDirectSource.poseidonSource, roundValue,
            DigestWindow.permutationOffset, Sampler.windowOffset,
            Sampler.windowBase, SamplerChain.sourceOffset,
            PiRLCStarts.samplerSourceLogicalStart]
          omega
  | logical descriptor position =>
      simp [PiRLCSamplerOrdinaryDirectPlan.Location.sourceColumn,
        PiRLCSamplerOrdinaryRetainedBlocks.logicalSource,
        PiRLCStarts.digestLaneLogicalStart, PiRLCStarts.windowLogicalStart,
        PiRLCStarts.samplerSourceLogicalStart]
      omega
  | fresh descriptor position =>
      change PiRLCStarts.samplerLogicalStart ≤
        PiRLCStarts.digestLaneFreshStart descriptor.source.val
            descriptor.round.val descriptor.lane.val + position.val
      unfold
        PiRLCStarts.digestLaneFreshStart PiRLCStarts.windowFreshStart
        PiRLCStarts.samplerSourceFreshStart PiRLCStarts.samplerFreshStart
        PiRLCStarts.phaseFreshStart PiRLCStarts.samplerLogicalStart
        PiRLCStarts.phaseLogicalStart
        NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset
      omega
  | selector source =>
      simp [PiRLCSamplerOrdinaryDirectPlan.Location.sourceColumn,
        PiRLCSamplerOrdinaryDirectSource.selectorSource,
        PiRLCStarts.selectorLogicalStart,
        PiRLCStarts.samplerSourceLogicalStart, First54.positionOffset]
      omega

private theorem stateLocation_sourceColumn_ge (location : StateLocation) :
    PiRLCStarts.samplerLogicalStart ≤ location.sourceColumn := by
  unfold StateLocation.sourceColumn
  omega

private theorem samplerLogicalStart_lt_sourceColumnCount :
    PiRLCStarts.samplerLogicalStart < Spartan.SourceColumnCount := by
  rw [Spartan.sourceColumnCount_eq]
  norm_num [PiRLCStarts.samplerLogicalStart, PiRLCStarts.phaseLogicalStart,
    PiRLCInputs.phaseOffset,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset]

private theorem ordinaryTarget_none_of_beforeSampler {column : Nat}
    (before : column < PiRLCStarts.samplerLogicalStart) :
    PiRLCSamplerOrdinaryDirectPlan.classifyTarget
        (Spartan.sourceToSpartan column) = none := by
  unfold PiRLCSamplerOrdinaryDirectPlan.classifyTarget
  rw [Spartan.spartanToSource_sourceToSpartan column
    (lt_trans before samplerLogicalStart_lt_sourceColumnCount)]
  cases found : PiRLCSamplerOrdinaryDirectPlan.classifySource column with
  | none => exact found
  | some location =>
      have owns := PiRLCSamplerOrdinaryDirectPlan.classifySource_sound found
      have lower := ordinaryLocation_sourceColumn_ge location
      exfalso
      omega

private theorem stateTarget_none_of_beforeSampler {column : Nat}
    (before : column < PiRLCStarts.samplerLogicalStart) :
    classifyStateTarget (Spartan.sourceToSpartan column) = none := by
  unfold classifyStateTarget
  rw [Spartan.spartanToSource_sourceToSpartan column
    (lt_trans before samplerLogicalStart_lt_sourceColumnCount)]
  cases found : classifyStateSource column with
  | none => exact found
  | some location =>
      have owns := classifyStateSource_sound found
      have lower := stateLocation_sourceColumn_ge location
      exfalso
      omega

/-- A source column before the sampler interval cannot alias either exact
sampler classifier, so the complete semantic view uses the canonical Stage 1
transition value. -/
theorem semanticEnv_source_eq_transitionEnv_of_beforeSampler
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment Spec.F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → Spec.F)
    {column : Nat} (before : column < PiRLCStarts.samplerLogicalStart) :
    semanticEnv geometry assignment base (Spartan.sourceToSpartan column) =
      RunningTransitionDirectPlan.transitionEnv program base
        (Spartan.sourceToSpartan column) := by
  unfold semanticEnv
  rw [ordinaryTarget_none_of_beforeSampler before,
    stateTarget_none_of_beforeSampler before]

private theorem stateColumn_eq_samplerStateColumn (location : StateLocation) :
    location.sourceColumn =
      PiRLCSamplerOrdinaryDirectPlan.samplerStateColumn
        ⟨location.source.val, by
          have sourceLt := location.source.isLt
          change location.source.val < 17 at sourceLt
          change location.source.val < 17
          exact sourceLt⟩
        location.step location.lane := by
  rfl

/-- Every complete sampler state output evaluates to the exact retained
Poseidon2 output form. The ordinary resolver owns its first four consumed
lanes; the complete state resolver owns all remaining lanes and the final
window output. -/
theorem semanticEnv_state
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment Spec.F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → Spec.F)
    (location : StateLocation) :
    semanticEnv geometry assignment base
        (Spartan.sourceToSpartan location.sourceColumn) =
      (location.form geometry).eval assignment := by
  let source : Fin PiRLCSamplerOrdinaryRetainedBlocks.sourceCount :=
    ⟨location.source.val, by
      have sourceLt := location.source.isLt
      change location.source.val < 17 at sourceLt
      simpa [PiRLCSamplerOrdinaryRetainedBlocks.sourceCount] using sourceLt⟩
  by_cases missing : location.step.val = 8 ∨ 4 ≤ location.lane.val
  · have sourceNone :=
      PiRLCSamplerOrdinaryDirectPlan.classifySource_samplerState_missing
        source location.step location.lane missing
    have targetNone :
        PiRLCSamplerOrdinaryDirectPlan.classifyTarget
            (Spartan.sourceToSpartan location.sourceColumn) = none := by
      unfold PiRLCSamplerOrdinaryDirectPlan.classifyTarget
      rw [Spartan.spartanToSource_sourceToSpartan location.sourceColumn
        location.sourceColumn_lt]
      have columnEq : location.sourceColumn =
          PiRLCSamplerOrdinaryDirectPlan.samplerStateColumn source
            location.step location.lane := by
        exact stateColumn_eq_samplerStateColumn location
      rw [columnEq]
      exact sourceNone
    unfold semanticEnv
    rw [targetNone, classifyStateTarget_source]
  · have stepLt := location.step.isLt
    change location.step.val < 9 at stepLt
    have roundLt : location.step.val < 8 := by
      by_contra notLt
      have eight : location.step.val = 8 := by omega
      exact missing (Or.inl eight)
    have laneLt : location.lane.val < 4 := by
      by_contra notLt
      exact missing (Or.inr (by omega))
    let round : Fin PiRLCSamplerOrdinaryRetainedBlocks.roundCount :=
      ⟨location.step.val, roundLt⟩
    let rateLane : Fin 4 := ⟨location.lane.val, laneLt⟩
    let descriptor : PiRLCSamplerOrdinaryRetainedBlocks.Lane :=
      ⟨source, round, ⟨0, by decide⟩⟩
    have locationEq : ordinaryStateLocation descriptor rateLane = location := by
      apply StateLocation.eq_of_fields
      · apply Fin.ext
        change source.val = location.source.val
        rfl
      · apply Fin.ext
        change round.val = location.step.val
        rfl
      · apply Fin.ext
        change rateLane.val = location.lane.val
        rfl
    have sourceColumnEq : location.sourceColumn =
        PiRLCSamplerOrdinaryDirectSource.poseidonSource source.val round.val
          rateLane := by
      rw [← locationEq]
      exact ordinaryStateLocation_sourceColumn descriptor rateLane
    have sourceFound : PiRLCSamplerOrdinaryDirectPlan.classifySource
        location.sourceColumn =
          some (.poseidon descriptor rateLane) := by
      rw [sourceColumnEq]
      by_cases zero : round.val = 0
      · have entry :=
          PiRLCSamplerOrdinaryDirectPlan.classifySource_poseidonEntry
            source rateLane
        have roundZero : round = ⟨0, by decide⟩ := by
          apply Fin.ext
          exact zero
        have descriptorEntry : descriptor =
            { source := source
              round := ⟨0, by decide⟩
              lane := ⟨0, by decide⟩ } := by
          change
            PiRLCSamplerOrdinaryRetainedBlocks.Lane.mk source round
                ⟨0, by decide⟩ =
              PiRLCSamplerOrdinaryRetainedBlocks.Lane.mk source
                ⟨0, by decide⟩ ⟨0, by decide⟩
          exact congrArg
            (fun boundedRound =>
              PiRLCSamplerOrdinaryRetainedBlocks.Lane.mk source boundedRound
                ⟨0, by decide⟩)
            roundZero
        rw [zero, descriptorEntry]
        exact entry
      · obtain ⟨previous, roundEq⟩ := Nat.exists_eq_succ_of_ne_zero zero
        have previousLt : previous + 1 <
            PiRLCSamplerOrdinaryRetainedBlocks.roundCount := by
          simpa [roundEq] using round.isLt
        have window :=
          PiRLCSamplerOrdinaryDirectPlan.classifySource_poseidonWindow
            source previous previousLt rateLane
        have roundSucc : round = ⟨previous + 1, previousLt⟩ := by
          apply Fin.ext
          exact roundEq
        have descriptorWindow : descriptor =
            { source := source
              round := ⟨previous + 1, previousLt⟩
              lane := ⟨0, by decide⟩ } := by
          change
            PiRLCSamplerOrdinaryRetainedBlocks.Lane.mk source round
                ⟨0, by decide⟩ =
              PiRLCSamplerOrdinaryRetainedBlocks.Lane.mk source
                ⟨previous + 1, previousLt⟩ ⟨0, by decide⟩
          exact congrArg
            (fun boundedRound =>
              PiRLCSamplerOrdinaryRetainedBlocks.Lane.mk source boundedRound
                ⟨0, by decide⟩)
            roundSucc
        rw [roundEq, descriptorWindow]
        exact window
    have targetFound : PiRLCSamplerOrdinaryDirectPlan.classifyTarget
        (Spartan.sourceToSpartan location.sourceColumn) =
          some (.poseidon descriptor rateLane) := by
      unfold PiRLCSamplerOrdinaryDirectPlan.classifyTarget
      rw [Spartan.spartanToSource_sourceToSpartan location.sourceColumn
        location.sourceColumn_lt]
      exact sourceFound
    unfold semanticEnv PiRLCSamplerOrdinaryDirectPlan.resolvedEnv
    rw [targetFound]
    unfold PiRLCSamplerOrdinaryDirectPlan.resolvedForm
    rw [targetFound]
    rw [← locationEq]
    rfl

/-- On the exact ordinary-row support, the complete semantic view is the
assignment-derived retained view used by the direct matrix plan. -/
theorem semanticEnv_eq_resolved_of_target
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment Spec.F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → Spec.F)
    {column : Nat}
    (support : PiRLCSamplerOrdinaryDirectSource.Target column) :
    semanticEnv geometry assignment base column =
      PiRLCSamplerOrdinaryDirectPlan.resolvedEnv geometry assignment column := by
  obtain ⟨location, found⟩ :=
    PiRLCSamplerOrdinaryDirectPlan.classifyTarget_complete support
  unfold semanticEnv
  rw [found]

/-- Canonical sampler ordinary-row satisfaction transfers to the complete
sampler semantic view without inspecting or rebuilding a row. -/
theorem rowsHold_semanticEnv
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment Spec.F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → Spec.F)
    (holds : R1CS.RowsHold
      (PiRLCSamplerOrdinaryDirectPlan.resolvedEnv geometry assignment)
      (PiRLCSamplerOrdinaryDirectSource.sourceRows
        (logicalWidth := relationLogicalWidth)
        (publicFits := relationPublicFits))) :
    R1CS.RowsHold (semanticEnv geometry assignment base)
      (PiRLCSamplerOrdinaryDirectSource.sourceRows
        (logicalWidth := relationLogicalWidth)
        (publicFits := relationPublicFits)) := by
  apply R1CS.rowsHold_of_agree _
    PiRLCSamplerOrdinaryDirectSource.Target
    (PiRLCSamplerOrdinaryDirectPlan.resolvedEnv geometry assignment)
    (semanticEnv geometry assignment base)
    (PiRLCSamplerOrdinaryDirectSource.sourceRows_varsSatisfy
      (logicalWidth := relationLogicalWidth)
      (publicFits := relationPublicFits))
  · intro column support
    exact semanticEnv_eq_resolved_of_target geometry assignment base support
  · exact holds

/-- The resolved entry-output source is the exact derived output of the
source's entry permutation. -/
theorem resolvedEnv_poseidonEntry
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment Spec.F logicalWidth)
    (source : Fin PiRLCSamplerOrdinaryRetainedBlocks.sourceCount)
    (lane : Fin 4) :
    PiRLCSamplerOrdinaryDirectPlan.resolvedEnv geometry assignment
        (Spartan.sourceToSpartan
          (PiRLCSamplerOrdinaryDirectSource.poseidonSource source.val 0 lane)) =
      PiRLCSamplerPoseidonPreservation.outputValue
        (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry) assignment
        (PiRLCSamplerPoseidonPlan.invocation source ⟨0, by decide⟩)
        (DigestWindow.rateLane lane) := by
  let location : PiRLCSamplerOrdinaryDirectPlan.Location :=
    .poseidon
      { source := source
        round := ⟨0, by decide⟩
        lane := ⟨0, by decide⟩ }
      lane
  have sourceBound :
      PiRLCSamplerOrdinaryDirectSource.poseidonSource source.val 0 lane <
        Spartan.SourceColumnCount := by
    change location.sourceColumn < Spartan.SourceColumnCount
    exact location.sourceColumn_lt
  unfold PiRLCSamplerOrdinaryDirectPlan.resolvedEnv
  rw [resolvedForm_of_source geometry sourceBound
    (PiRLCSamplerOrdinaryDirectPlan.classifySource_poseidonEntry source lane)]
  rfl

/-- Every resolved window-output source is the exact derived output of the
matching source and sampler step. -/
theorem resolvedEnv_poseidonWindow
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment Spec.F logicalWidth)
    (source : Fin PiRLCSamplerOrdinaryRetainedBlocks.sourceCount)
    (previous : Nat)
    (roundLt : previous + 1 < PiRLCSamplerOrdinaryRetainedBlocks.roundCount)
    (lane : Fin 4) :
    PiRLCSamplerOrdinaryDirectPlan.resolvedEnv geometry assignment
        (Spartan.sourceToSpartan
          (PiRLCSamplerOrdinaryDirectSource.poseidonSource source.val
            (previous + 1) lane)) =
      PiRLCSamplerPoseidonPreservation.outputValue
        (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry) assignment
        (PiRLCSamplerPoseidonPlan.invocation source
          ⟨previous + 1, lt_trans roundLt (by decide)⟩)
        (DigestWindow.rateLane lane) := by
  let location : PiRLCSamplerOrdinaryDirectPlan.Location :=
    .poseidon
      { source := source
        round := ⟨previous + 1, roundLt⟩
        lane := ⟨0, by decide⟩ }
      lane
  have sourceBound :
      PiRLCSamplerOrdinaryDirectSource.poseidonSource source.val
          (previous + 1) lane < Spartan.SourceColumnCount := by
    change location.sourceColumn < Spartan.SourceColumnCount
    exact location.sourceColumn_lt
  unfold PiRLCSamplerOrdinaryDirectPlan.resolvedEnv
  rw [resolvedForm_of_source geometry sourceBound
    (PiRLCSamplerOrdinaryDirectPlan.classifySource_poseidonWindow source
      previous roundLt lane)]
  rfl

/-- Every retained digest-lane logical source evaluates to the exact canonical
package source selected by the transition environment. -/
theorem resolvedEnv_logical
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment Spec.F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → Spec.F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → Spec.F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → Spec.F)
    (encodes : PiRLCSamplerOrdinaryRetainedGeometry.Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment program base groupValue
        products))
    (descriptor : PiRLCSamplerOrdinaryRetainedBlocks.Lane)
    (position : Fin PiRLCSamplerOrdinaryRetainedBlocks.logicalCountPerLane) :
    PiRLCSamplerOrdinaryDirectPlan.resolvedEnv geometry assignment
        (Spartan.sourceToSpartan
          (PiRLCSamplerOrdinaryRetainedBlocks.logicalSource descriptor
            position)) =
      RunningTransitionDirectPlan.transitionEnv program base
        (Spartan.sourceToSpartan
          (PiRLCSamplerOrdinaryRetainedBlocks.logicalSource descriptor
            position)) := by
  unfold PiRLCSamplerOrdinaryDirectPlan.resolvedEnv
  rw [resolvedForm_of_source geometry
    (PiRLCSamplerOrdinaryRetainedBlocks.logicalSource_lt descriptor position)
    (PiRLCSamplerOrdinaryDirectPlan.classifySource_logical descriptor position)]
  change ((PiRLCSamplerOrdinaryRetainedBlocks.logicalBlock program).form
    (PiRLCSamplerOrdinaryRetainedGeometry.logicalStart program)
    (PiRLCSamplerOrdinaryRetainedGeometry.logicalFits geometry)
    (PiRLCSamplerOrdinaryRetainedBlocks.logicalSlot descriptor position)).eval
      assignment = _
  rw [LowNormBlock.Block.form_eval _ _ _ assignment _ encodes.logical]
  rw [PiRLCSamplerOrdinaryRetainedBlocks.logicalBlock_source]
  exact RunningTransitionDirectPlan.sourceAssignment_packageSource program base
    groupValue products _ _

/-- Every retained digest-lane fresh source evaluates to the exact canonical
package source selected by the transition environment. -/
theorem resolvedEnv_fresh
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment Spec.F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → Spec.F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → Spec.F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → Spec.F)
    (encodes : PiRLCSamplerOrdinaryRetainedGeometry.Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment program base groupValue
        products))
    (descriptor : PiRLCSamplerOrdinaryRetainedBlocks.Lane)
    (position : Fin PiRLCSamplerOrdinaryRetainedBlocks.freshCountPerLane) :
    PiRLCSamplerOrdinaryDirectPlan.resolvedEnv geometry assignment
        (Spartan.sourceToSpartan
          (PiRLCSamplerOrdinaryRetainedBlocks.freshSource descriptor
            position)) =
      RunningTransitionDirectPlan.transitionEnv program base
        (Spartan.sourceToSpartan
          (PiRLCSamplerOrdinaryRetainedBlocks.freshSource descriptor
            position)) := by
  unfold PiRLCSamplerOrdinaryDirectPlan.resolvedEnv
  rw [resolvedForm_of_source geometry
    (PiRLCSamplerOrdinaryRetainedBlocks.freshSource_lt descriptor position)
    (PiRLCSamplerOrdinaryDirectPlan.classifySource_fresh descriptor position)]
  change ((PiRLCSamplerOrdinaryRetainedBlocks.freshBlock program).form
    (PiRLCSamplerOrdinaryRetainedGeometry.freshStart program)
    (PiRLCSamplerOrdinaryRetainedGeometry.freshFits geometry)
    (PiRLCSamplerOrdinaryRetainedBlocks.freshSlot descriptor position)).eval
      assignment = _
  rw [LowNormBlock.Block.form_eval _ _ _ assignment _ encodes.fresh]
  rw [PiRLCSamplerOrdinaryRetainedBlocks.freshBlock_source]
  exact RunningTransitionDirectPlan.sourceAssignment_packageSource program base
    groupValue products _ _

/-- The product-plan base environment and transition environment are the same
view of every package-private logical source. -/
theorem baseEnv_eq_transitionEnv
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → Spec.F)
    (column : Nat)
    (bound : column < PiRLCProductPlan.basePackage.layout.constantColumn) :
    PiRLCProductPlan.baseEnv program base column =
      RunningTransitionDirectPlan.transitionEnv program base
        (Spartan.sourceToSpartan column) := by
  have mappedBound :=
    PiRLCProductPlan.sourceToSpartan_lt_basePackage column bound
  rw [PiRLCProductPlan.baseEnv_eq_mappedPackageColumn program base column bound]
  unfold RunningTransitionDirectPlan.transitionEnv
  rw [dif_pos mappedBound]
  rfl

/-- The fail-closed final selector source is the exact retained First54 full
slot and the exact canonical transition value. -/
theorem resolvedEnv_selector
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment Spec.F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → Spec.F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → Spec.F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → Spec.F)
    (encodes : PiRLCRetainedPreservation.Encodes
      (PiRLCSamplerOrdinaryDirectPlan.piRlcGeometry geometry) assignment base
      groupValue products)
    (source : Fin PiRLCSamplerOrdinaryRetainedBlocks.sourceCount) :
    PiRLCSamplerOrdinaryDirectPlan.resolvedEnv geometry assignment
        (Spartan.sourceToSpartan
          (PiRLCSamplerOrdinaryDirectSource.selectorSource source.val)) =
      RunningTransitionDirectPlan.transitionEnv program base
        (Spartan.sourceToSpartan
          (PiRLCSamplerOrdinaryDirectSource.selectorSource source.val)) := by
  let location : PiRLCSamplerOrdinaryDirectPlan.Location := .selector source
  have sourceBound :
      PiRLCSamplerOrdinaryDirectSource.selectorSource source.val <
        Spartan.SourceColumnCount := by
    change location.sourceColumn < Spartan.SourceColumnCount
    exact location.sourceColumn_lt
  unfold PiRLCSamplerOrdinaryDirectPlan.resolvedEnv
  rw [resolvedForm_of_source geometry sourceBound
    (PiRLCSamplerOrdinaryDirectPlan.classifySource_selector source)]
  change ((PiRLCFirst54RetainedBlocks.positionBlock program).form
    (PiRLCRetainedGeometry.positionStart program)
    (PiRLCRetainedGeometry.positionFits
      (PiRLCSamplerOrdinaryDirectPlan.piRlcGeometry geometry))
    (PiRLCFirst54DirectSchedule.positionIndex
      (PiRLCFirst54DirectPlan.finalPositionDescriptor source))).eval
        assignment = _
  rw [LowNormBlock.Block.form_eval _ _ _ assignment _ encodes.position]
  rw [PiRLCFirst54RetainedBlocks.positionBlock_source]
  rw [PiRLCFirst54DirectSchedule.position_positionIndex]
  unfold PiRLCFirst54DirectPlan.retainedPositionColumn
  rw [PiRLCRetainedPreservation.sourceAssignment_package]
  rw [PiRLCFirst54DirectPlan.finalPositionDescriptor_positionColumn]
  have privateBound :
      PiRLCSamplerOrdinaryDirectSource.selectorSource source.val <
        PiRLCProductPlan.basePackage.layout.constantColumn := by
    have sourceLt := source.isLt
    have constant : PiRLCProductPlan.basePackage.layout.constantColumn =
        29336446 :=
      NightstreamFPrime.Export.Stage1.Package.circuitPackage_layout_values.2.2.1
    rw [constant]
    norm_num [PiRLCSamplerOrdinaryDirectSource.selectorSource,
      PiRLCStarts.selectorLogicalStart, PiRLCStarts.samplerSourceLogicalStart,
      PiRLCStarts.samplerLogicalStart, PiRLCStarts.phaseLogicalStart,
      PiRLCInputs.phaseOffset, First54.positionOffset,
      First54.candidateCount, First54.roundPrivateCount,
      First54.fullSlot, First54Step.fullSlot, First54Step.slotCount,
      First54ValueStep.outputCount,
      PiRLCSamplerOrdinaryRetainedBlocks.sourceCount,
      PiRLCSamplerOrdinaryRows.sourceCount,
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset] at sourceLt ⊢
    omega
  change PiRLCProductPlan.baseEnv program base
      (PiRLCSamplerOrdinaryDirectSource.selectorSource source.val) = _
  exact baseEnv_eq_transitionEnv program base _ privateBound

end NightstreamFPrime.Export.Stage1.PiRLCSamplerRetainedCustody
