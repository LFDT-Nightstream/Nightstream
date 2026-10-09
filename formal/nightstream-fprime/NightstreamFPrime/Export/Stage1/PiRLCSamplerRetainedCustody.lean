import NightstreamFPrime.Export.Stage1.PermutationPlan
import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryDirectPlan
import NightstreamFPrime.Export.Stage1.PiRLCSamplerPoseidonPreservation
import NightstreamFPrime.Layout.Stage1.SpartanValues

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
        ⟨948 + current.val, by
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
          Data.publicFits).length = 948 :=
      PiCCSInvocations.invocations_length Data.logicalWidth Data.publicFits
    have offsetEq :
        948 + current.val -
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
      ⟨948 + current.val -
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

def stateOutputOffset : Nat := 1080
def stateStepStride : Nat := 3117

/-- One of the two complete eight-lane permutation outputs owned by each
scalar sampler: domain entry and transcript advance. -/
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
      location.step.val * stateStepStride + location.lane.val

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
    Sampler.counts.1, stateStepStride,
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
    stateStepStride

private def stateLaneIndex (column : Nat) : Nat :=
  (stateSourceOffset column - stateOutputOffset) %
    stateStepStride

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
    location.step.val * stateStepStride + location.lane.val
  have withinSourceLt : withinSource < Sampler.logicalPrivateCount := by
    have stepLt := location.step.isLt
    have laneLt := location.lane.isLt
    norm_num [withinSource, stateOutputOffset,
      PiRLCSamplerPoseidonPlan.invocationsPerSource, Spec.Poseidon2.width,
      Sampler.counts.1, stateStepStride] at stepLt laneLt ⊢
    omega
  unfold stateSourceIndex StateLocation.sourceColumn
  rw [show
      PiRLCStarts.samplerLogicalStart +
          location.source.val * Sampler.logicalPrivateCount +
          stateOutputOffset +
          location.step.val * stateStepStride +
          location.lane.val - PiRLCStarts.samplerLogicalStart =
        location.source.val * Sampler.logicalPrivateCount + withinSource by
    dsimp [withinSource]
    omega]
  exact quotient_at location.source.val withinSource
    Sampler.logicalPrivateCount (by norm_num [Sampler.counts.1])
    withinSourceLt

private theorem stateSourceOffset_at (location : StateLocation) :
    stateSourceOffset location.sourceColumn =
      stateOutputOffset +
        location.step.val * stateStepStride +
          location.lane.val := by
  let withinSource := stateOutputOffset +
    location.step.val * stateStepStride + location.lane.val
  have withinSourceLt : withinSource < Sampler.logicalPrivateCount := by
    have stepLt := location.step.isLt
    have laneLt := location.lane.isLt
    norm_num [withinSource, stateOutputOffset,
      PiRLCSamplerPoseidonPlan.invocationsPerSource, Spec.Poseidon2.width,
      Sampler.counts.1, stateStepStride] at stepLt laneLt ⊢
    omega
  unfold stateSourceOffset StateLocation.sourceColumn
  rw [show
      PiRLCStarts.samplerLogicalStart +
          location.source.val * Sampler.logicalPrivateCount +
          stateOutputOffset +
          location.step.val * stateStepStride +
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
          location.step.val * stateStepStride +
          location.lane.val - stateOutputOffset =
        location.step.val * stateStepStride +
          location.lane.val by omega]
  exact quotient_at location.step.val location.lane.val
    stateStepStride
    (by norm_num [stateStepStride])
    (lt_trans location.lane.isLt (by
      norm_num [Spec.Poseidon2.width, stateStepStride]))

private theorem stateLaneIndex_at (location : StateLocation) :
    stateLaneIndex location.sourceColumn = location.lane.val := by
  unfold stateLaneIndex
  rw [stateSourceOffset_at]
  rw [show
      stateOutputOffset +
          location.step.val * stateStepStride +
          location.lane.val - stateOutputOffset =
        location.step.val * stateStepStride +
          location.lane.val by omega]
  exact remainder_at location.step.val location.lane.val
    stateStepStride
    (lt_trans location.lane.isLt (by
      norm_num [Spec.Poseidon2.width, stateStepStride]))

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

/-- Semantic view for the complete sampler: exact retained forms own every
sampler-ordinary source, while all other columns retain the canonical Stage 1
transition view for the other lifecycle phases. -/
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
  | poseidon source lane => rw [PiRLCSamplerOrdinaryDirectPlan.poseidonColumn]; omega
  | logical source position => rw [PiRLCSamplerOrdinaryDirectPlan.logicalColumn]; omega
  | word source position => rw [PiRLCSamplerOrdinaryDirectPlan.wordColumn]; omega
  | fresh source position =>
      change PiRLCStarts.samplerLogicalStart ≤ PiRLCStarts.samplerFreshStart + source.val * 144 + position.val
      simp only [PiRLCStarts.samplerFreshStart, PiRLCStarts.phaseFreshStart,
        PiRLCStarts.samplerLogicalStart, Formal.samplerOffset]
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

private theorem ordinaryMissing (location : StateLocation)
    (missing : location.step.val = 1 ∨ 4 ≤ location.lane.val) :
    PiRLCSamplerOrdinaryDirectPlan.classifySource location.sourceColumn = none := by
  have sourceLt : location.source.val < 17 := location.source.isLt
  have stepLt : location.step.val < 2 := location.step.isLt
  have laneLt : location.lane.val < 16 := location.lane.isLt
  have stepCases : location.step.val = 0 ∨ location.step.val = 1 := by omega
  cases found : PiRLCSamplerOrdinaryDirectPlan.classifySource location.sourceColumn with
  | none => rfl
  | some selected =>
      have owns := PiRLCSamplerOrdinaryDirectPlan.classifySource_sound found
      exfalso
      cases selected with
      | poseidon source position =>
          have selectedLt : source.val < 17 := source.isLt
          have positionLt : position.val < 4 := position.isLt
          rw [PiRLCSamplerOrdinaryDirectPlan.poseidonColumn] at owns
          norm_num [StateLocation.sourceColumn, stateOutputOffset, stateStepStride,
            PiRLCStarts.samplerLogicalStart, Formal.samplerOffset, PiRLCStarts.phaseLogicalStart_eq,
            PiRLCStarts.samplerFreshStart, PiRLCStarts.phaseFreshStart_eq,
            Sampler.counts.1] at owns
          rcases stepCases with zero | one <;>
            (rcases Nat.lt_trichotomy location.source.val source.val with before | same | after <;> omega)

      | logical source position =>
          have selectedLt : source.val < 17 := source.isLt
          have positionLt : position.val < 617 := position.isLt
          rw [PiRLCSamplerOrdinaryDirectPlan.logicalColumn] at owns
          norm_num [StateLocation.sourceColumn, stateOutputOffset, stateStepStride,
            PiRLCStarts.samplerLogicalStart, Formal.samplerOffset, PiRLCStarts.phaseLogicalStart_eq,
            PiRLCStarts.samplerFreshStart, PiRLCStarts.phaseFreshStart_eq,
            Sampler.counts.1] at owns
          rcases stepCases with zero | one <;>
            (rcases Nat.lt_trichotomy location.source.val source.val with before | same | after <;> omega)

      | word source position =>
          have selectedLt : source.val < 17 := source.isLt
          have positionLt : position.val < 54 := position.isLt
          rw [PiRLCSamplerOrdinaryDirectPlan.wordColumn] at owns
          norm_num [StateLocation.sourceColumn, stateOutputOffset, stateStepStride,
            PiRLCStarts.samplerLogicalStart, Formal.samplerOffset, PiRLCStarts.phaseLogicalStart_eq,
            PiRLCStarts.samplerFreshStart, PiRLCStarts.phaseFreshStart_eq,
            Sampler.counts.1] at owns
          rcases stepCases with zero | one <;>
            (rcases Nat.lt_trichotomy location.source.val source.val with before | same | after <;> omega)

      | fresh source position =>
          have selectedLt : source.val < 17 := source.isLt
          have positionLt : position.val < 144 := position.isLt
          change PiRLCStarts.samplerFreshStart + source.val * 144 + position.val = _ at owns
          norm_num [StateLocation.sourceColumn, stateOutputOffset, stateStepStride,
            PiRLCStarts.samplerLogicalStart, Formal.samplerOffset, PiRLCStarts.phaseLogicalStart_eq,
            PiRLCStarts.samplerFreshStart, PiRLCStarts.phaseFreshStart_eq,
            Sampler.counts.1] at owns
          rcases stepCases with zero | one <;>
            (rcases Nat.lt_trichotomy location.source.val source.val with before | same | after <;> omega)

/-- Both complete permutation outputs are owned by the retained Poseidon2 forms.
The ordinary resolver reads the first four entry lanes. -/
theorem semanticEnv_state
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment Spec.F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → Spec.F)
    (location : StateLocation) :
    semanticEnv geometry assignment base (Spartan.sourceToSpartan location.sourceColumn) =
      (location.form geometry).eval assignment := by
  by_cases missing : location.step.val = 1 ∨ 4 ≤ location.lane.val
  · have targetNone : PiRLCSamplerOrdinaryDirectPlan.classifyTarget
        (Spartan.sourceToSpartan location.sourceColumn) = none := by
      unfold PiRLCSamplerOrdinaryDirectPlan.classifyTarget
      rw [Spartan.spartanToSource_sourceToSpartan _ location.sourceColumn_lt]
      exact ordinaryMissing location missing
    unfold semanticEnv
    rw [targetNone, classifyStateTarget_source]
  · have stepLt : location.step.val < 2 := location.step.isLt
    have stepZero : location.step = ⟨0, by decide⟩ := by apply Fin.ext; change location.step.val = 0; omega
    have laneLt : location.lane.val < 4 := by omega
    let lane : Fin 4 := ⟨location.lane.val, laneLt⟩
    have sourceEq : location.sourceColumn =
        (PiRLCSamplerOrdinaryDirectPlan.Location.poseidon location.source lane).sourceColumn := by
      rw [PiRLCSamplerOrdinaryDirectPlan.poseidonColumn]
      simp only [StateLocation.sourceColumn, stepZero, Fin.val_zero, Nat.zero_mul,
        Nat.add_zero, stateOutputOffset, Sampler.counts.1]
      dsimp only [lane]
      omega
    have found : PiRLCSamplerOrdinaryDirectPlan.classifyTarget
        (Spartan.sourceToSpartan location.sourceColumn) = some (.poseidon location.source lane) := by
      unfold PiRLCSamplerOrdinaryDirectPlan.classifyTarget
      rw [Spartan.spartanToSource_sourceToSpartan _ location.sourceColumn_lt, sourceEq]
      exact PiRLCSamplerOrdinaryDirectPlan.classifySource_poseidonEntry location.source lane
    unfold semanticEnv PiRLCSamplerOrdinaryDirectPlan.resolvedEnv
      PiRLCSamplerOrdinaryDirectPlan.resolvedForm
    rw [found]
    simp only [PiRLCSamplerOrdinaryDirectPlan.Location.form, StateLocation.form, stepZero]
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


/-- The product-plan and transition views agree on private source columns
outside the PiCCS transcript-output family. Callers prove their source ranges. -/
theorem baseEnv_eq_transitionEnv
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → Spec.F)
    (column : Nat)
    (bound : column < PiRLCProductPlan.basePackage.layout.constantColumn)
    (outside : column < PiCCSStarts.statementWitnessStart ∨
      PiCCSStarts.statementWitnessStart +
          PiCCSOrdinarySourceSupport.transcriptInvocationCount * 1096 ≤ column) :
    PiRLCProductPlan.baseEnv program base column =
      RunningTransitionDirectPlan.transitionEnv program base
        (Spartan.sourceToSpartan column) := by
  have sourceBound : column < Spartan.SourceColumnCount := by
    have constant : PiRLCProductPlan.basePackage.layout.constantColumn = 11464596 :=
      Package.circuitPackage_layout_values.2.2.1
    rw [constant] at bound
    rw [Spartan.sourceColumnCount_eq]
    omega
  rw [PiRLCProductPlan.baseEnv_eq_mappedPackageColumn program base column bound,
    RunningTransitionDirectPlan.transitionEnv_of_outside program base column sourceBound outside]
  exact (SourceCompiler.sourceEnv_at base
    (PiRLCProductPlan.mappedPackageColumn program column bound)).symm


end NightstreamFPrime.Export.Stage1.PiRLCSamplerRetainedCustody
