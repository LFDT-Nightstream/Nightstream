import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryDirectPlan

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryMatrixSubstitution

open NightstreamFPrime.Layout.MatrixProgram NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_2 NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open PiRLCSamplerOrdinaryRetainedBlocks PiRLCSamplerOrdinaryRetainedGeometry
open PiRLCSamplerOrdinaryDirectPlan (Location)

abbrev Program := Lifecycle.Stage1.Application.Program

def frameSourceStart : Nat := PackageOrder.sourceToPackage PiRLCStarts.samplerLogicalStart
def poseidonSourceStart : Nat := frameSourceStart + 1158
def logicalSourceStart : Nat := frameSourceStart + 3022
def wordSourceStart : Nat := frameSourceStart + 4965
def poseidonGrid (program : Program) : SourceGrid :=
  SourceGrid.externalOfSemantic (PiRLCSamplerPoseidonPlan.retainedBlock program)
    (PiRLCSamplerPoseidonPlan.retainedStart program)
    poseidonSourceStart 15 5029 1 5029 5 155 342 0

def logicalGrid (program : Program) : SourceGrid :=
  SourceGrid.ofSemantic (logicalBlock program) (logicalStart program)
    logicalSourceStart 15 5029 1 5029 769 0 769 0

def wordGrid (program : Program) : SourceGrid :=
  SourceGrid.ofSemantic (PiRLCRetainedGeometry.challengeBlock program)
    (PiRLCRetainedGeometry.challengeStart program)
    wordSourceStart 15 5029 1 5029 64 0 64 0

def substitution (program : Program) : SourceSubstitution where
  ranges := []
  grids := [poseidonGrid program, logicalGrid program, wordGrid program]

private theorem samplerLogical_after_piCcs : PackageOrder.piCcsPhaseOffset ≤ PiRLCStarts.samplerLogicalStart := by
  norm_num [PackageOrder.piCcsPhaseOffset, PiRLCStarts.samplerLogicalStart,
    Formal.samplerOffset, PiRLCStarts.phaseLogicalStart_eq]

private theorem scalarTarget (source offset : Nat) :
    PackageOrder.sourceToPackage (PiRLCStarts.samplerLogicalStart + source * 5029 + offset) =
      frameSourceStart + source * 5029 + offset := by
  rw [Nat.add_assoc, PackageOrder.sourceToPackage_add_of_piCcsLocal _ _ samplerLogical_after_piCcs]
  unfold frameSourceStart
  omega

theorem poseidonTarget (source : Fin sourceCount) (lane : Fin 5) :
    PackageOrder.sourceToPackage (Location.poseidon source lane).sourceColumn =
      poseidonSourceStart + source.val * 5029 + lane.val := by
  rw [PiRLCSamplerOrdinaryDirectPlan.poseidonColumn, scalarTarget]
  unfold poseidonSourceStart
  omega

theorem logicalTarget (source : Fin sourceCount) (position : Fin logicalCountPerSource) :
    PackageOrder.sourceToPackage (Location.logical source position).sourceColumn =
      logicalSourceStart + source.val * 5029 + position.val := by
  rw [PiRLCSamplerOrdinaryDirectPlan.logicalColumn, scalarTarget]
  unfold logicalSourceStart
  omega

theorem wordTarget (source : Fin sourceCount) (position : Fin ringDegree) :
    PackageOrder.sourceToPackage (Location.word source position).sourceColumn =
      wordSourceStart + source.val * 5029 + position.val := by
  rw [PiRLCSamplerOrdinaryDirectPlan.wordColumn, scalarTarget]
  unfold wordSourceStart
  omega

theorem logicalGrid_form? {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (source : Fin sourceCount) (position : Fin logicalCountPerSource) :
    (logicalGrid program).form? logicalWidth (PackageOrder.sourceToPackage (Location.logical source position).sourceColumn) =
      some ((Location.logical source position).form geometry) := by
  rw [logicalTarget]
  have sourceLt : source.val < 15 := source.isLt
  have positionLt : position.val < 769 := position.isLt
  have direct := SourceGrid.form?_ofSemantic (logicalBlock program) (logicalStart program)
    logicalSourceStart 15 5029 1 5029 769 0 769 0 (logicalFits geometry)
    (by decide) (by decide) source ⟨0, by decide⟩ position (by omega) (by omega)
    (by rw [logicalBlock_slotCount]; omega)
  simp only [Nat.zero_add, Nat.zero_mul, Nat.mul_zero, Nat.add_zero] at direct
  have slot : (⟨source.val * 769 + position.val, by rw [logicalBlock_slotCount]; omega⟩ :
      Fin (logicalBlock program).slotCount) = logicalSlot source position := by
    apply Fin.ext
    simp [logicalSlot, Fin.encodeProd, Fin.mkDivMod, logicalCountPerSource, Nat.mul_comm]
  rw [slot] at direct
  exact direct

theorem wordGrid_form? {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (source : Fin sourceCount) (position : Fin ringDegree) :
    (wordGrid program).form? logicalWidth (PackageOrder.sourceToPackage (Location.word source position).sourceColumn) =
      some ((Location.word source position).form geometry) := by
  rw [wordTarget]
  have sourceLt : source.val < 15 := source.isLt
  have positionLt : position.val < 64 := position.isLt
  have slotCount : (PiRLCRetainedGeometry.challengeBlock program).slotCount = 960 := rfl
  have direct := SourceGrid.form?_ofSemantic (PiRLCRetainedGeometry.challengeBlock program)
    (PiRLCRetainedGeometry.challengeStart program)
    wordSourceStart 15 5029 1 5029 64 0 64 0
    (PiRLCRetainedGeometry.challengeFits (piRlcGeometry geometry))
    (by decide) (by decide) source ⟨0, by decide⟩ position (by omega) (by omega)
    (by rw [slotCount]; omega)
  simp only [Nat.zero_add, Nat.zero_mul, Nat.mul_zero, Nat.add_zero] at direct
  have slot : (⟨source.val * 64 + position.val, by rw [slotCount]; omega⟩ :
      Fin (PiRLCRetainedGeometry.challengeBlock program).slotCount) =
      PiRLCProductSourceBlocks.challengeIndex source position := by
    apply Fin.ext
    simp [PiRLCProductSourceBlocks.challengeIndex, Fin.encodeProd, Fin.mkDivMod, ringDegree,
      Nat.mul_comm]
  rw [slot] at direct
  exact direct

theorem poseidonGrid_form? {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (source : Fin sourceCount) (lane : Fin 5) :
    (poseidonGrid program).form? logicalWidth (PackageOrder.sourceToPackage (Location.poseidon source lane).sourceColumn) =
      some ((Location.poseidon source lane).form geometry) := by
  rw [poseidonTarget]
  have sourceLt : source.val < 15 := source.isLt
  have laneLt := lane.isLt
  have direct := SourceGrid.form?_externalOfSemantic
    (PiRLCSamplerPoseidonPlan.retainedBlock program) (PiRLCSamplerPoseidonPlan.retainedStart program)
    poseidonSourceStart 15 5029 1 5029 5 155 342 0
    (PiRLCSamplerPoseidonPlan.retainedFits (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry))
    (by decide) (by decide) source ⟨0, by decide⟩ lane (by omega) (by omega) (by omega)
    (by intro selected; have selectedLt := selected.isLt
        rw [PiRLCSamplerPoseidonPlan.retainedBlock_slotCount]; omega)
  simp only [Nat.zero_mul, Nat.mul_zero, Nat.add_zero] at direct
  have outputEq : SparseLayer.external (fun selected : Fin 16 =>
      (PiRLCSamplerPoseidonPlan.retainedBlock program).form (PiRLCSamplerPoseidonPlan.retainedStart program)
        (PiRLCSamplerPoseidonPlan.retainedFits (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry))
        ⟨155 + source.val * 342 + selected.val, by
          have selectedLt := selected.isLt
          rw [PiRLCSamplerPoseidonPlan.retainedBlock_slotCount]; omega⟩)
      ⟨lane.val, by omega⟩ = (Location.poseidon source lane).form geometry := by
    unfold Location.form PiRLCSamplerPoseidonPlan.interface PoseidonRetainedFamily.familyInterface
      PoseidonSboxFamilyPlan.Interface.output
    apply congrArg (fun state => SparseLayer.external state (Sampler.rateLane lane))
    funext selected
    apply congrArg ((PiRLCSamplerPoseidonPlan.retainedBlock program).form
      (PiRLCSamplerPoseidonPlan.retainedStart program)
      (PiRLCSamplerPoseidonPlan.retainedFits (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry)))
    apply Fin.ext
    simp only [PoseidonRetainedFamily.slot, Fin.cast, PoseidonRetainedSlots.rows_length,
      PoseidonRetainedSlots.finalRow_val, Location.poseidonInvocation,
      PiRLCSamplerPoseidonPlan.invocation, Fin.mkDivMod, Fin.encodeProd,
      PiRLCSamplerPoseidonPlan.invocationsPerSource]
    omega
  rw [outputEq] at direct
  exact direct

private theorem frameGrid_none (grid : SourceGrid) (logicalWidth start offset : Nat)
    (origin : grid.sourceStart = frameSourceStart + start)
    (shape : grid.majorCount = 15 ∧ grid.majorSourceStride = 5029 ∧
      grid.minorCount = 1 ∧ grid.minorSourceStride = 5029)
    (endBound : start + grid.runCount ≤ 5029)
    (source : Fin sourceCount) (offsetBound : offset < 5029)
    (outside : offset < start ∨ start + grid.runCount ≤ offset) :
    grid.form? logicalWidth (frameSourceStart + source.val * 5029 + offset) = none := by
  have sourceLt : source.val < 15 := source.isLt
  have gap (major : Fin grid.majorCount) (distance : Nat)
      (distanceBound : distance < 5029) (afterRun : grid.runCount ≤ distance) :
      grid.form? logicalWidth
        (grid.sourceStart + major.val * 5029 + distance) = none := by
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
      have checked := gap previous (5029 + offset - start) (by omega) (by omega)
      have address : grid.sourceStart + previous.val * 5029 + (5029 + offset - start) =
          frameSourceStart + source.val * 5029 + offset := by
        rw [origin]
        dsimp only [previous]
        omega
      rw [address] at checked
      exact checked
  · let major : Fin grid.majorCount := ⟨source.val, by rw [shape.1]; exact sourceLt⟩
    have checked := gap major (offset - start) (by omega) (by omega)
    have address : grid.sourceStart + major.val * 5029 + (offset - start) =
        frameSourceStart + source.val * 5029 + offset := by
      rw [origin]
      dsimp only [major]
      omega
    rw [address] at checked
    exact checked

theorem substitution_location_form? {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (location : Location) :
    (substitution program).form? logicalWidth (PackageOrder.sourceToPackage location.sourceColumn) =
      some (location.form geometry) := by
  cases location with
  | poseidon source lane =>
      have valueBound : lane.val < 5 := lane.isLt
      have address : PackageOrder.sourceToPackage (Location.poseidon source lane).sourceColumn =
          frameSourceStart + source.val * 5029 + (1158 + lane.val) := by
        rw [poseidonTarget]
        unfold poseidonSourceStart
        omega
      have hit := poseidonGrid_form? geometry source lane
      have missLogical : (logicalGrid program).form? logicalWidth
          (PackageOrder.sourceToPackage (Location.poseidon source lane).sourceColumn) = none := by
        rw [address]
        exact frameGrid_none _ logicalWidth 3022 (1158 + lane.val)
          rfl ⟨rfl, rfl, rfl, rfl⟩ (by change 3791 ≤ 5029; decide) source (by omega) (by left; omega)
      have missWord : (wordGrid program).form? logicalWidth
          (PackageOrder.sourceToPackage (Location.poseidon source lane).sourceColumn) = none := by
        rw [address]
        exact frameGrid_none _ logicalWidth 4965 (1158 + lane.val)
          rfl ⟨rfl, rfl, rfl, rfl⟩ (by change 5029 ≤ 5029; decide) source (by omega) (by left; omega)
      simp [substitution, SourceSubstitution.form?, hit, missLogical, missWord]
  | logical source position =>
      have valueBound : position.val < 769 := position.isLt
      have address : PackageOrder.sourceToPackage (Location.logical source position).sourceColumn =
          frameSourceStart + source.val * 5029 + (3022 + position.val) := by
        rw [logicalTarget]
        unfold logicalSourceStart
        omega
      have hit := logicalGrid_form? geometry source position
      have missPoseidon : (poseidonGrid program).form? logicalWidth
          (PackageOrder.sourceToPackage (Location.logical source position).sourceColumn) = none := by
        rw [address]
        exact frameGrid_none _ logicalWidth 1158 (3022 + position.val)
          rfl ⟨rfl, rfl, rfl, rfl⟩ (by change 1163 ≤ 5029; decide) source (by omega) (by right; change 1158 + 5 ≤ 3022 + position.val; omega)
      have missWord : (wordGrid program).form? logicalWidth
          (PackageOrder.sourceToPackage (Location.logical source position).sourceColumn) = none := by
        rw [address]
        exact frameGrid_none _ logicalWidth 4965 (3022 + position.val)
          rfl ⟨rfl, rfl, rfl, rfl⟩ (by change 5029 ≤ 5029; decide) source (by omega) (by left; omega)
      simp [substitution, SourceSubstitution.form?, hit, missPoseidon, missWord]
  | word source position =>
      have valueBound : position.val < 64 := position.isLt
      have address : PackageOrder.sourceToPackage (Location.word source position).sourceColumn =
          frameSourceStart + source.val * 5029 + (4965 + position.val) := by
        rw [wordTarget]
        unfold wordSourceStart
        omega
      have hit := wordGrid_form? geometry source position
      have missPoseidon : (poseidonGrid program).form? logicalWidth
          (PackageOrder.sourceToPackage (Location.word source position).sourceColumn) = none := by
        rw [address]
        exact frameGrid_none _ logicalWidth 1158 (4965 + position.val)
          rfl ⟨rfl, rfl, rfl, rfl⟩ (by change 1163 ≤ 5029; decide) source (by omega) (by right; change 1158 + 5 ≤ 4965 + position.val; omega)
      have missLogical : (logicalGrid program).form? logicalWidth
          (PackageOrder.sourceToPackage (Location.word source position).sourceColumn) = none := by
        rw [address]
        exact frameGrid_none _ logicalWidth 3022 (4965 + position.val)
          rfl ⟨rfl, rfl, rfl, rfl⟩ (by change 3791 ≤ 5029; decide) source (by omega) (by right; change 3022 + 769 ≤ 4965 + position.val; omega)
      simp [substitution, SourceSubstitution.form?, hit, missPoseidon, missLogical]

theorem substitution_agrees_on_target
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (column : Fin PackageOrder.packageColumnCount)
    (support : PiRLCSamplerOrdinaryDirectSource.Target column.val) :
    (substitution program).form? logicalWidth column.val =
      some ((PiRLCSamplerOrdinaryDirectPlan.sourceMap geometry).form column) := by
  rcases support with ⟨source, sourceSupport, mapped⟩
  have sourceBound : source < PackageOrder.SourceColumnCount := sourceSupport.bounded
  have inverse := PackageOrder.packageToSource_sourceToPackage source sourceBound
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
      PackageOrder.sourceToPackage decoded.sourceColumn = column.val := by
    rw [owns, mapped]
  change (substitution program).form? logicalWidth column.val =
    some (match PiRLCSamplerOrdinaryDirectPlan.classifyTarget column.val with
      | none => .empty
      | some location => location.form geometry)
  rw [found]
  simpa only [target] using substitution_location_form? geometry decoded

variable {relationLogicalWidth : Nat}
  {relationPublicFits : ringDegree * publicRingColumns ≤
    RingCarrierLayout.carrierWidth relationLogicalWidth}

private theorem programRow_support (index : Fin 13635) :
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
    (geometry : Geometry program logicalWidth) (index : Fin 13635) :
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
