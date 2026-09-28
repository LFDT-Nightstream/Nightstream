import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryMatrixSchedule
import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryDirectPlan
import NightstreamFPrime.Export.MatrixProgram.Ordinary

/-! Four compact source grids for the sampler ordinary matrix block.
Each grid names current canonical source columns and their retained forms. -/

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryMatrixSubstitution

open NightstreamFPrime.Layout.MatrixProgram NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1 NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open PiRLCSamplerOrdinaryRetainedBlocks PiRLCSamplerOrdinaryRetainedGeometry
open PiRLCSamplerOrdinaryDirectPlan (Location)

abbrev Program := Lifecycle.Stage1.Application.Program

def frameSourceStart : Nat := Spartan.sourceToSpartan PiRLCStarts.samplerLogicalStart
def poseidonSourceStart : Nat := frameSourceStart + 584
def logicalSourceStart : Nat := frameSourceStart + 1996
def wordSourceStart : Nat := frameSourceStart + 3205
def freshSourceStart : Nat := Spartan.sourceToSpartan PiRLCStarts.samplerFreshStart

def poseidonGrid (program : Program) : SourceGrid :=
  SourceGrid.externalOfSemantic (PiRLCSamplerPoseidonPlan.retainedBlock program)
    (PiRLCSamplerPoseidonPlan.retainedStart program)
    poseidonSourceStart 17 3259 1 3259 4 78 172 0

def logicalGrid (program : Program) : SourceGrid :=
  SourceGrid.ofSemantic (logicalBlock program) (logicalStart program)
    logicalSourceStart 17 3259 1 3259 617 0 617 0

def wordGrid (program : Program) : SourceGrid :=
  SourceGrid.ofSemantic (PiRLCRetainedGeometry.challengeBlock program)
    (PiRLCRetainedGeometry.challengeStart program)
    wordSourceStart 17 3259 1 3259 54 0 54 0

def freshGrid (program : Program) : SourceGrid :=
  SourceGrid.ofSemantic (freshBlock program) (freshStart program)
    freshSourceStart 17 1548 1 1548 1548 0 1548 0

def substitution (program : Program) : SourceSubstitution where
  ranges := []
  grids := [poseidonGrid program, logicalGrid program, wordGrid program, freshGrid program]

private theorem samplerLogical_after_piCcs : Spartan.piCcsPhaseOffset ≤ PiRLCStarts.samplerLogicalStart := by
  norm_num [Spartan.piCcsPhaseOffset, PiRLCStarts.samplerLogicalStart,
    Formal.samplerOffset, PiRLCStarts.phaseLogicalStart_eq]

private theorem samplerFresh_after_piCcs : Spartan.piCcsPhaseOffset ≤ PiRLCStarts.samplerFreshStart := by
  change Spartan.piCcsPhaseOffset ≤ PiRLCStarts.phaseFreshStart
  rw [PiRLCStarts.phaseFreshStart_eq]
  norm_num [Spartan.piCcsPhaseOffset]

private theorem scalarTarget (source offset : Nat) :
    Spartan.sourceToSpartan (PiRLCStarts.samplerLogicalStart + source * 3259 + offset) =
      frameSourceStart + source * 3259 + offset := by
  rw [Nat.add_assoc, Spartan.sourceToSpartan_add_of_piCcsLocal _ _ samplerLogical_after_piCcs]
  unfold frameSourceStart
  omega

theorem poseidonTarget (source : Fin sourceCount) (lane : Fin 4) :
    Spartan.sourceToSpartan (Location.poseidon source lane).sourceColumn =
      poseidonSourceStart + source.val * 3259 + lane.val := by
  rw [PiRLCSamplerOrdinaryDirectPlan.poseidonColumn, scalarTarget]
  unfold poseidonSourceStart
  omega

theorem logicalTarget (source : Fin sourceCount) (position : Fin logicalCountPerSource) :
    Spartan.sourceToSpartan (Location.logical source position).sourceColumn =
      logicalSourceStart + source.val * 3259 + position.val := by
  rw [PiRLCSamplerOrdinaryDirectPlan.logicalColumn, scalarTarget]
  unfold logicalSourceStart
  omega

theorem wordTarget (source : Fin sourceCount) (position : Fin ringDegree) :
    Spartan.sourceToSpartan (Location.word source position).sourceColumn =
      wordSourceStart + source.val * 3259 + position.val := by
  rw [PiRLCSamplerOrdinaryDirectPlan.wordColumn, scalarTarget]
  unfold wordSourceStart
  omega

theorem freshTarget (source : Fin sourceCount) (position : Fin freshCountPerSource) :
    Spartan.sourceToSpartan (Location.fresh source position).sourceColumn =
      freshSourceStart + source.val * 1548 + position.val := by
  change Spartan.sourceToSpartan (PiRLCStarts.samplerFreshStart + source.val * 1548 + position.val) = _
  rw [Nat.add_assoc, Spartan.sourceToSpartan_add_of_piCcsLocal _ _ samplerFresh_after_piCcs]
  unfold freshSourceStart
  omega

theorem logicalGrid_form? {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (source : Fin sourceCount) (position : Fin logicalCountPerSource) :
    (logicalGrid program).form? logicalWidth (Spartan.sourceToSpartan (Location.logical source position).sourceColumn) =
      some ((Location.logical source position).form geometry) := by
  rw [logicalTarget]
  have sourceLt : source.val < 17 := source.isLt
  have positionLt : position.val < 617 := position.isLt
  have direct := SourceGrid.form?_ofSemantic (logicalBlock program) (logicalStart program)
    logicalSourceStart 17 3259 1 3259 617 0 617 0 (logicalFits geometry)
    (by decide) (by decide) source ⟨0, by decide⟩ position (by omega) (by omega)
    (by rw [logicalBlock_slotCount]; omega)
  simp only [Nat.zero_add, Nat.zero_mul, Nat.mul_zero, Nat.add_zero] at direct
  have slot : (⟨source.val * 617 + position.val, by rw [logicalBlock_slotCount]; omega⟩ :
      Fin (logicalBlock program).slotCount) = logicalSlot source position := by
    apply Fin.ext
    simp [logicalSlot, Fin.encodeProd, logicalCountPerSource, Nat.mul_comm]
  rw [slot] at direct
  exact direct

theorem freshGrid_form? {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (source : Fin sourceCount) (position : Fin freshCountPerSource) :
    (freshGrid program).form? logicalWidth (Spartan.sourceToSpartan (Location.fresh source position).sourceColumn) =
      some ((Location.fresh source position).form geometry) := by
  rw [freshTarget]
  have sourceLt : source.val < 17 := source.isLt
  have positionLt : position.val < 1548 := position.isLt
  have direct := SourceGrid.form?_ofSemantic (freshBlock program) (freshStart program)
    freshSourceStart 17 1548 1 1548 1548 0 1548 0 (freshFits geometry)
    (by decide) (by decide) source ⟨0, by decide⟩ position (by omega) (by omega)
    (by rw [freshBlock_slotCount]; omega)
  simp only [Nat.zero_add, Nat.zero_mul, Nat.mul_zero, Nat.add_zero] at direct
  have slot : (⟨source.val * 1548 + position.val, by rw [freshBlock_slotCount]; omega⟩ :
      Fin (freshBlock program).slotCount) = freshSlot source position := by
    apply Fin.ext
    simp [freshSlot, Fin.encodeProd, freshCountPerSource, Nat.mul_comm]
  rw [slot] at direct
  exact direct

theorem wordGrid_form? {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (source : Fin sourceCount) (position : Fin ringDegree) :
    (wordGrid program).form? logicalWidth (Spartan.sourceToSpartan (Location.word source position).sourceColumn) =
      some ((Location.word source position).form geometry) := by
  rw [wordTarget]
  have sourceLt : source.val < 17 := source.isLt
  have positionLt : position.val < 54 := position.isLt
  have slotCount : (PiRLCRetainedGeometry.challengeBlock program).slotCount = 918 := rfl
  have direct := SourceGrid.form?_ofSemantic (PiRLCRetainedGeometry.challengeBlock program)
    (PiRLCRetainedGeometry.challengeStart program)
    wordSourceStart 17 3259 1 3259 54 0 54 0
    (PiRLCRetainedGeometry.challengeFits (piRlcGeometry geometry))
    (by decide) (by decide) source ⟨0, by decide⟩ position (by omega) (by omega)
    (by rw [slotCount]; omega)
  simp only [Nat.zero_add, Nat.zero_mul, Nat.mul_zero, Nat.add_zero] at direct
  have slot : (⟨source.val * 54 + position.val, by rw [slotCount]; omega⟩ :
      Fin (PiRLCRetainedGeometry.challengeBlock program).slotCount) =
      PiRLCProductSourceBlocks.challengeIndex source position := by
    apply Fin.ext
    simp [PiRLCProductSourceBlocks.challengeIndex, Fin.encodeProd, ringDegree, Nat.mul_comm]
  rw [slot] at direct
  exact direct

theorem poseidonGrid_form? {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (source : Fin sourceCount) (lane : Fin 4) :
    (poseidonGrid program).form? logicalWidth (Spartan.sourceToSpartan (Location.poseidon source lane).sourceColumn) =
      some ((Location.poseidon source lane).form geometry) := by
  rw [poseidonTarget]
  have sourceLt : source.val < 17 := source.isLt
  have laneLt := lane.isLt
  have direct := SourceGrid.form?_externalOfSemantic
    (PiRLCSamplerPoseidonPlan.retainedBlock program) (PiRLCSamplerPoseidonPlan.retainedStart program)
    poseidonSourceStart 17 3259 1 3259 4 78 172 0
    (PiRLCSamplerPoseidonPlan.retainedFits (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry))
    (by decide) (by decide) source ⟨0, by decide⟩ lane (by omega) (by omega) (by omega)
    (by intro selected; have selectedLt := selected.isLt
        rw [PiRLCSamplerPoseidonPlan.retainedBlock_slotCount]; omega)
  simp only [Nat.zero_mul, Nat.mul_zero, Nat.add_zero] at direct
  have outputEq : SparseLayer.external (fun selected : Fin 8 =>
      (PiRLCSamplerPoseidonPlan.retainedBlock program).form (PiRLCSamplerPoseidonPlan.retainedStart program)
        (PiRLCSamplerPoseidonPlan.retainedFits (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry))
        ⟨78 + source.val * 172 + selected.val, by
          have selectedLt := selected.isLt
          rw [PiRLCSamplerPoseidonPlan.retainedBlock_slotCount]; omega⟩)
      ⟨lane.val, by omega⟩ = (Location.poseidon source lane).form geometry := by
    unfold Location.form PiRLCSamplerPoseidonPlan.interface PoseidonRetainedFamily.familyInterface
      PoseidonRetainedFamily.outputState
    apply congrArg (fun state => SparseLayer.external state (Sampler.rateLane lane))
    funext selected
    apply congrArg ((PiRLCSamplerPoseidonPlan.retainedBlock program).form
      (PiRLCSamplerPoseidonPlan.retainedStart program)
      (PiRLCSamplerPoseidonPlan.retainedFits (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry)))
    apply Fin.ext
    simp [Location.poseidonInvocation, PiRLCSamplerPoseidonPlan.invocation,
      PoseidonRetainedFamily.slot, Fin.encodeProd, PoseidonRetainedSlots.finalRow_val,
      Sampler.rateLane, PiRLCSamplerPoseidonPlan.invocationsPerSource]
    omega
  rw [outputEq] at direct
  exact direct

private theorem frameGrid_none (grid : SourceGrid) (logicalWidth start offset : Nat)
    (origin : grid.sourceStart = frameSourceStart + start)
    (shape : grid.majorCount = 17 ∧ grid.majorSourceStride = 3259 ∧
      grid.minorCount = 1 ∧ grid.minorSourceStride = 3259)
    (endBound : start + grid.runCount ≤ 3259)
    (source : Fin sourceCount) (offsetBound : offset < 3259)
    (outside : offset < start ∨ start + grid.runCount ≤ offset) :
    grid.form? logicalWidth (frameSourceStart + source.val * 3259 + offset) = none := by
  have sourceLt : source.val < 17 := source.isLt
  have gap (major : Fin grid.majorCount) (distance : Nat)
      (distanceBound : distance < 3259) (afterRun : grid.runCount ≤ distance) :
      grid.form? logicalWidth
        (grid.sourceStart + major.val * 3259 + distance) = none := by
    have checked := SourceGrid.form?_eq_none_at_gap grid logicalWidth major
      ⟨0, by rw [shape.2.2.1]; decide⟩ distance
      (by rw [shape.2.1]; decide) (by rw [shape.2.2.2]; decide)
      (by simpa only [shape.2.1, Nat.zero_mul, Nat.zero_add] using distanceBound)
      (by simpa only [shape.2.2.2] using distanceBound) afterRun
    simpa only [shape.2.1, Nat.zero_mul, Nat.add_zero] using checked
  by_cases before : offset < start
  · by_cases first : source.val = 0
    · apply SourceGrid.form?_eq_none_of_before
      rw [origin, first]
      omega
    · let previous : Fin grid.majorCount := ⟨source.val - 1, by rw [shape.1]; omega⟩
      have checked := gap previous (3259 + offset - start) (by omega) (by omega)
      have address : grid.sourceStart + previous.val * 3259 + (3259 + offset - start) =
          frameSourceStart + source.val * 3259 + offset := by
        rw [origin]
        dsimp only [previous]
        omega
      rw [address] at checked
      exact checked
  · let major : Fin grid.majorCount := ⟨source.val, by rw [shape.1]; exact sourceLt⟩
    have checked := gap major (offset - start) (by omega) (by omega)
    have address : grid.sourceStart + major.val * 3259 + (offset - start) =
        frameSourceStart + source.val * 3259 + offset := by
      rw [origin]
      dsimp only [major]
      omega
    rw [address] at checked
    exact checked

private theorem freshSourceStart_eq : freshSourceStart = frameSourceStart + 107729 := by
  change Spartan.sourceToSpartan (PiRLCStarts.samplerLogicalStart + 107729) = _
  rw [Spartan.sourceToSpartan_add_of_piCcsLocal _ _ samplerLogical_after_piCcs]
  rfl

private theorem freshGrid_none_scalar (program : Program) (logicalWidth offset : Nat)
    (source : Fin sourceCount) (offsetBound : offset < 3259) :
    (freshGrid program).form? logicalWidth (frameSourceStart + source.val * 3259 + offset) = none := by
  apply SourceGrid.form?_eq_none_of_before
  change frameSourceStart + source.val * 3259 + offset < freshSourceStart
  rw [freshSourceStart_eq]
  have sourceLt : source.val < 17 := source.isLt
  omega

private theorem frameGrid_none_fresh (grid : SourceGrid) (logicalWidth start : Nat)
    (origin : grid.sourceStart = frameSourceStart + start)
    (shape : grid.majorCount = 17 ∧ grid.majorSourceStride = 3259)
    (startBound : start ≤ 3259) (source : Fin sourceCount) (position : Fin freshCountPerSource) :
    grid.form? logicalWidth (freshSourceStart + source.val * 1548 + position.val) = none := by
  apply SourceGrid.form?_eq_none_of_after
  · rw [shape.2]
    decide
  · rw [origin, shape.1, shape.2, freshSourceStart_eq]
    omega



theorem substitution_location_form? {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (location : Location) :
    (substitution program).form? logicalWidth (Spartan.sourceToSpartan location.sourceColumn) =
      some (location.form geometry) := by
  cases location with
  | poseidon source lane =>
      have valueBound : lane.val < 4 := lane.isLt
      have address : Spartan.sourceToSpartan (Location.poseidon source lane).sourceColumn =
          frameSourceStart + source.val * 3259 + (584 + lane.val) := by
        rw [poseidonTarget]
        unfold poseidonSourceStart
        omega
      have hit := poseidonGrid_form? geometry source lane
      have missLogical : (logicalGrid program).form? logicalWidth
          (Spartan.sourceToSpartan (Location.poseidon source lane).sourceColumn) = none := by
        rw [address]
        exact frameGrid_none _ logicalWidth 1996 (584 + lane.val)
          rfl ⟨rfl, rfl, rfl, rfl⟩ (by change 2613 ≤ 3259; decide) source (by omega) (by left; change 584 + lane.val < 1996; omega)
      have missWord : (wordGrid program).form? logicalWidth
          (Spartan.sourceToSpartan (Location.poseidon source lane).sourceColumn) = none := by
        rw [address]
        exact frameGrid_none _ logicalWidth 3205 (584 + lane.val)
          rfl ⟨rfl, rfl, rfl, rfl⟩ (by change 3259 ≤ 3259; decide) source (by omega) (by left; change 584 + lane.val < 3205; omega)
      have missFresh : (freshGrid program).form? logicalWidth
          (Spartan.sourceToSpartan (Location.poseidon source lane).sourceColumn) = none := by
        rw [address]
        exact freshGrid_none_scalar program logicalWidth (584 + lane.val) source (by omega)
      simp [substitution, SourceSubstitution.form?, hit, missLogical, missWord, missFresh]
  | logical source position =>
      have valueBound : position.val < 617 := position.isLt
      have address : Spartan.sourceToSpartan (Location.logical source position).sourceColumn =
          frameSourceStart + source.val * 3259 + (1996 + position.val) := by
        rw [logicalTarget]
        unfold logicalSourceStart
        omega
      have hit := logicalGrid_form? geometry source position
      have missPoseidon : (poseidonGrid program).form? logicalWidth
          (Spartan.sourceToSpartan (Location.logical source position).sourceColumn) = none := by
        rw [address]
        exact frameGrid_none _ logicalWidth 584 (1996 + position.val)
          rfl ⟨rfl, rfl, rfl, rfl⟩ (by change 588 ≤ 3259; decide) source (by omega) (by right; change 584 + 4 ≤ 1996 + position.val; omega)
      have missWord : (wordGrid program).form? logicalWidth
          (Spartan.sourceToSpartan (Location.logical source position).sourceColumn) = none := by
        rw [address]
        exact frameGrid_none _ logicalWidth 3205 (1996 + position.val)
          rfl ⟨rfl, rfl, rfl, rfl⟩ (by change 3259 ≤ 3259; decide) source (by omega) (by left; change 1996 + position.val < 3205; omega)
      have missFresh : (freshGrid program).form? logicalWidth
          (Spartan.sourceToSpartan (Location.logical source position).sourceColumn) = none := by
        rw [address]
        exact freshGrid_none_scalar program logicalWidth (1996 + position.val) source (by omega)
      simp [substitution, SourceSubstitution.form?, hit, missPoseidon, missWord, missFresh]
  | word source position =>
      have valueBound : position.val < 54 := position.isLt
      have address : Spartan.sourceToSpartan (Location.word source position).sourceColumn =
          frameSourceStart + source.val * 3259 + (3205 + position.val) := by
        rw [wordTarget]
        unfold wordSourceStart
        omega
      have hit := wordGrid_form? geometry source position
      have missPoseidon : (poseidonGrid program).form? logicalWidth
          (Spartan.sourceToSpartan (Location.word source position).sourceColumn) = none := by
        rw [address]
        exact frameGrid_none _ logicalWidth 584 (3205 + position.val)
          rfl ⟨rfl, rfl, rfl, rfl⟩ (by change 588 ≤ 3259; decide) source (by omega) (by right; change 584 + 4 ≤ 3205 + position.val; omega)
      have missLogical : (logicalGrid program).form? logicalWidth
          (Spartan.sourceToSpartan (Location.word source position).sourceColumn) = none := by
        rw [address]
        exact frameGrid_none _ logicalWidth 1996 (3205 + position.val)
          rfl ⟨rfl, rfl, rfl, rfl⟩ (by change 2613 ≤ 3259; decide) source (by omega) (by right; change 1996 + 617 ≤ 3205 + position.val; omega)
      have missFresh : (freshGrid program).form? logicalWidth
          (Spartan.sourceToSpartan (Location.word source position).sourceColumn) = none := by
        rw [address]
        exact freshGrid_none_scalar program logicalWidth (3205 + position.val) source (by omega)
      simp [substitution, SourceSubstitution.form?, hit, missPoseidon, missLogical, missFresh]
  | fresh source position =>
      have hit := freshGrid_form? geometry source position
      have missPoseidon : (poseidonGrid program).form? logicalWidth
          (Spartan.sourceToSpartan (Location.fresh source position).sourceColumn) = none := by
        rw [freshTarget]
        exact frameGrid_none_fresh _ logicalWidth 584 rfl ⟨rfl, rfl⟩
          (by decide) source position
      have missLogical : (logicalGrid program).form? logicalWidth
          (Spartan.sourceToSpartan (Location.fresh source position).sourceColumn) = none := by
        rw [freshTarget]
        exact frameGrid_none_fresh _ logicalWidth 1996 rfl ⟨rfl, rfl⟩
          (by decide) source position
      have missWord : (wordGrid program).form? logicalWidth
          (Spartan.sourceToSpartan (Location.fresh source position).sourceColumn) = none := by
        rw [freshTarget]
        exact frameGrid_none_fresh _ logicalWidth 3205 rfl ⟨rfl, rfl⟩
          (by decide) source position
      simp [substitution, SourceSubstitution.form?, hit, missPoseidon, missLogical, missWord]

theorem substitution_agrees_on_target
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (column : Fin Spartan.spartanColumnCount)
    (support : PiRLCSamplerOrdinaryDirectSource.Target column.val) :
    (substitution program).form? logicalWidth column.val =
      some ((PiRLCSamplerOrdinaryDirectPlan.sourceMap geometry).form column) := by
  rcases support with ⟨source, sourceSupport, mapped⟩
  have sourceBound : source < Spartan.SourceColumnCount := sourceSupport.bounded
  have inverse := Spartan.spartanToSource_sourceToSpartan source sourceBound
  rw [mapped] at inverse
  rcases PiRLCSamplerOrdinaryDirectPlan.classifyTarget_complete
      ⟨source, sourceSupport, mapped⟩ with ⟨decoded, found⟩
  have decodedFound :
      PiRLCSamplerOrdinaryDirectPlan.classifySource source = some decoded := by
    unfold PiRLCSamplerOrdinaryDirectPlan.classifyTarget at found
    rw [inverse] at found
    exact found
  have owns :=
    PiRLCSamplerOrdinaryDirectPlan.classifySource_sound decodedFound
  have target :
      Spartan.sourceToSpartan decoded.sourceColumn = column.val := by
    rw [owns, mapped]
  change (substitution program).form? logicalWidth column.val =
    some (match PiRLCSamplerOrdinaryDirectPlan.classifyTarget column.val with
      | none => .empty
      | some location => location.form geometry)
  rw [found]
  simpa only [target] using substitution_location_form? geometry decoded

variable {relationLogicalWidth : Nat}
  {relationPublicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth relationLogicalWidth}

private theorem programRow_support (index : Fin 38811) :
    (PiRLCSamplerOrdinaryDirectSource.programRow
      (logicalWidth := relationLogicalWidth)
      (publicFits := relationPublicFits) index).VarsSatisfy
        PiRLCSamplerOrdinaryDirectSource.Target := by
  exact PiRLCSamplerOrdinaryDirectSource.sourceRows_varsSatisfy _
    (List.get_mem _
      (PiRLCSamplerOrdinaryDirectSource.sourceListIndex
        (logicalWidth := relationLogicalWidth)
        (publicFits := relationPublicFits) index))

theorem substitution_agrees_on_programRow
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (index : Fin 38811) :
    let row := PiRLCSamplerOrdinaryDirectSource.programRow
      (logicalWidth := relationLogicalWidth)
      (publicFits := relationPublicFits) index
    Ordinary.AgreesOnTerms (substitution program)
        (PiRLCSamplerOrdinaryDirectPlan.sourceMap geometry) row.a.terms ∧
      Ordinary.AgreesOnTerms (substitution program)
        (PiRLCSamplerOrdinaryDirectPlan.sourceMap geometry) row.b.terms ∧
      Ordinary.AgreesOnTerms (substitution program)
        (PiRLCSamplerOrdinaryDirectPlan.sourceMap geometry) row.c.terms := by
  dsimp only
  have scope := programRow_support
    (relationLogicalWidth := relationLogicalWidth)
    (relationPublicFits := relationPublicFits) index
  refine ⟨?_, ?_, ?_⟩
  · intro term member bounded
    exact substitution_agrees_on_target geometry ⟨term.1, bounded⟩
      (scope.1 term member)
  · intro term member bounded
    exact substitution_agrees_on_target geometry ⟨term.1, bounded⟩
      (scope.2.1 term member)
  · intro term member bounded
    exact substitution_agrees_on_target geometry ⟨term.1, bounded⟩
      (scope.2.2 term member)

end NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryMatrixSubstitution
