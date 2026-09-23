import NightstreamFPrime.Export.Stage1.ApplicationPoseidonMatrixRows

/-! Execute the proved application matrix metadata without constructing physical source layouts. -/

namespace NightstreamFPrime.Export.Stage1.ApplicationPoseidonMatrixProgram

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open ApplicationPoseidonRetainedBlock ApplicationPoseidonRetainedGeometry

variable {application : Stage1.Application.Program} {certificate : Certificate application}
  {columns : Nat}

private theorem localStart_value (certificate : Certificate application) : localStart application = 149282421 := by
  have width := completeLogicalWidth_eq application certificate
  rw [completeLogicalWidth, block_coordinateCount] at width
  omega

private theorem inputStart_value : inputStart application = 121294795 := by
  unfold inputStart ApplicationOrdinaryGeometry.inputStart PiRLCPoseidonGeometry.priorInputStart
  rw [PiRLCRetainedGeometry.prefixLogicalWidth_eq]
  rfl

private theorem outputStart_value : outputStart application = 123319908 := by
  unfold outputStart ApplicationOrdinaryGeometry.outputStart PiRLCPoseidonGeometry.outputInputStart
    PiRLCPoseidonGeometry.priorInputStart
  rw [PiRLCRetainedGeometry.prefixLogicalWidth_eq]
  have count : (PiRLCPoseidonGeometry.priorInputBlock application).coordinateCount = 2025113 := by
    simp [PiRLCPoseidonGeometry.priorInputBlock]
  rw [count]
  rfl

def directRetained : RetainedBlock := ⟨.field, 258, 149282421⟩

def directInputProgram : PoseidonInput.Program where
  rules := [
    ⟨⟨1, 2, 0, 8⟩, .external directRetained 78 86⟩,
    constantRule,
    ⟨⟨0, 1, 0, 4⟩, .retained ⟨.field, 4, 121294795⟩ 0 0 1⟩,
    ⟨⟨1, 1, 0, 4⟩, .retained ⟨.field, 4, 149282257⟩ 0 0 1⟩]

def directPoseidonBlock : Poseidon.Block := ⟨3, 0, directRetained, directInputProgram⟩

private theorem retained_eq : RetainedBlock.ofSemantic (block application certificate)
    (localStart application) = directRetained := by
  unfold RetainedBlock.ofSemantic
  rw [block_slotCount, localStart_value certificate]
  rfl

theorem inputProgram_eq_direct : inputProgram application certificate = directInputProgram := by
  unfold inputProgram directInputProgram previousRule priorRule messageRule
  rw [retained_eq]
  have prior : RetainedBlock.ofSemantic (ApplicationRetainedBlocks.inputBlock application)
      (inputStart application) = ⟨.field, 4, 121294795⟩ := by
    unfold RetainedBlock.ofSemantic
    rw [inputStart_value]
    rfl
  have message : RetainedBlock.ofSemantic (ApplicationRetainedBlocks.witnessBlock application)
      (witnessStart application) = ⟨.field, 4, 149282257⟩ := by
    change RetainedBlock.mk .field application.witnessWordCount
      (PiRLCSamplerOrdinaryRetainedGeometry.completeLogicalWidth application) = _
    rw [certificate.wordCount, PiRLCSamplerOrdinaryRetainedGeometry.completeLogicalWidth_eq]
    rfl
  rw [prior, message]

theorem poseidonBlock_eq_direct (geometry : Geometry application certificate columns) :
    poseidonBlock geometry = directPoseidonBlock := by
  change Poseidon.Block.mk 3 0
    (RetainedBlock.ofSemantic (block application certificate) (localStart application))
    (inputProgram application certificate) = _
  rw [retained_eq, inputProgram_eq_direct]
  rfl

private def directDigest : RetainedBlock := ⟨.field, 4, 123319908⟩

private def directRetainedFits (geometry : Geometry application certificate columns) :
    directRetained.start + directRetained.semantic.coordinateCount ≤ columns := by
  have bound := geometry.completeFits
  rw [completeLogicalWidth_eq] at bound
  exact bound

private def directDigestFits (geometry : Geometry application certificate columns) :
    directDigest.start + directDigest.semantic.coordinateCount ≤ columns := by
  have bound := geometry.completeFits
  rw [completeLogicalWidth_eq] at bound
  change 123319908 + 4 * 41 ≤ columns
  omega

def directPins (geometry : Geometry application certificate columns) : PinFamilyPlan.Interface columns 4 where
  oneColumn := oneColumn geometry
  value := fun lane => (directDigest.semantic.form directDigest.start (directDigestFits geometry) lane).add
    (SparseForm.scale (-1) ((SparseLayer.external fun index => directRetained.semantic.form directRetained.start
      (directRetainedFits geometry) ⟨250 + index.val, by have := index.isLt; change 250 + index.val < 258; omega⟩)
      ⟨lane.val, by have := lane.isLt; omega⟩))

theorem pins_eq_direct (geometry : Geometry application certificate columns) :
    Stage1.Poseidon2HashChainCompact.pins (interface geometry) = directPins geometry := by
  have digest (lane : Fin 4) : (interface geometry).digest lane =
      directDigest.semantic.form directDigest.start (directDigestFits geometry) lane := by
    apply LowNormBlock.Block.form_eq_of_coordinates
    · rfl
    · change outputStart application + lane.val * 41 = 123319908 + lane.val * 41
      rw [outputStart_value]
  have sbox (lane : Fin 8) :
      (interface geometry).sbox 2 (PoseidonRetainedSlots.finalRow lane) =
        directRetained.semantic.form directRetained.start (directRetainedFits geometry)
          ⟨250 + lane.val, by have := lane.isLt; change 250 + lane.val < 258; omega⟩ := by
    apply LowNormBlock.Block.form_eq_of_coordinates
    · rfl
    · change localStart application +
        (Fin.encodeProd ((2 : Fin 3), PoseidonRetainedSlots.finalRow lane)).val * 41 =
          149282421 + (250 + lane.val) * 41
      rw [localStart_value (certificate := certificate)]
      simp [Fin.encodeProd, PoseidonRetainedSlots.finalRow_val, PoseidonRetainedSlots.rows_length]
      omega
  have output : Stage1.Poseidon2HashChainCompact.output (interface geometry) 2 =
      SparseLayer.external (fun lane => directRetained.semantic.form directRetained.start
        (directRetainedFits geometry) ⟨250 + lane.val, by have := lane.isLt; change 250 + lane.val < 258; omega⟩) := by
    unfold Stage1.Poseidon2HashChainCompact.output
    exact congrArg SparseLayer.external (funext sbox)
  unfold Stage1.Poseidon2HashChainCompact.pins directPins
  simp only [digest, output]
  rfl

def directMatrixProgram (geometry : Geometry application certificate columns) : MatrixProgram.Program where
  blocks := [.poseidon directPoseidonBlock, .pin (Pin.Block.ofSemantic (directPins geometry))]

theorem matrixProgram_eq_direct (geometry : Geometry application certificate columns) :
    matrixProgram geometry = directMatrixProgram geometry := by
  unfold matrixProgram directMatrixProgram bindingBlock
  rw [poseidonBlock_eq_direct, pins_eq_direct]

@[csimp] theorem matrixProgram_eq_directMatrixProgram : @matrixProgram = @directMatrixProgram := by
  funext application certificate columns geometry
  exact matrixProgram_eq_direct geometry

end NightstreamFPrime.Export.Stage1.ApplicationPoseidonMatrixProgram
